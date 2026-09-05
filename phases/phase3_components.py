"""Phase 3 — Component attribution, NeuroStrike-style recovery, mediation & rescue (RQ3).

For each (layer, concept) in {early, mid, late} x {role, harm, control}, ranks
MLP neurons three ways, and compares all of them against NeuroStrike's own
neuron set:
  1. causal attribution     — activation-weighted projection onto the concept subspace
  2. static alignment       — pure down_proj weight-direction cosine alignment (no activations)
  3. activation contrast    — |mean act on positive prompts - mean on negative| (unsupervised)
  vs. NeuroStrike           — their supervised gate/up logistic probe (core/neurostrike.py)

and measures top-k overlap between them (RQ3: "can we recover the same safety
neurons just by measuring alignment, without activation-difference profiling?",
and "what functions do NeuroStrike's components perform?"). Their neurons are
index-comparable to ours: down_proj input i == silu(gate_i) * up_i.
Also computes the frozen architectural quantities (C_R, N_eff, r_eff, k_50)
that phase 4 uses as predictors, classifies each top neuron by which concept
it contributes to most, and runs a mediation + causal-rescue experiment at
the mid layer (ablate each concept's top neurons -> which R's projection
drops -> restore R_control's subspace without restoring the neurons -> does
behavior return?).

Requires phase1_geometry to have been run first.

Outputs (results/phase3_components/):
  architecture_metrics.csv           layer, concept, C_R, N_eff, r_eff, k_50
  neuron_rankings.pt                 {(layer, concept): {"causal":.., "static":.., "activation_contrast":.., "neurostrike":..}} ranked index tensors
  neurostrike_weights.pt             {module_name: probe weight vector} (their signal, unthresholded)
  neurostrike_neurons.pt             {module_name: selected neuron indices} (|z|>3 & w>0) — phase 4's prefix attack
  neuron_ranking_overlap.csv         layer, concept, k, comparison, jaccard
  neuron_functional_classification.csv  layer, neuron_idx, best_concept, score_role, score_harm, score_control
  mediation_rescue_results.json
  mediation_rescue_samples.txt
"""
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from core import data, neurostrike
from core.attribution import AttributionEngine
from core.config import Config, load_config
from core.hooks import HookEngine, capture_last_token_residuals, capture_mlp_neuron_activations
from core.interventions import InterventionEngine
from core.io_utils import get_logger, load_json, load_torch, require_phase_output, save_df, save_json, save_torch
from core.metrics import component_ablation_curve, component_set_overlap, find_k50, make_projection_score_fn, refusal_margin_score
from core.model_io import build_prompt, generate_text, load_model
from core.subspaces import SubspaceEngine

PHASE_NAME = "phase3_components"
UPSTREAM_PHASE = "phase1_geometry"
CONCEPTS = ["role", "harm", "control"]
OVERLAP_K_GRID = [10, 25, 50, 100]
TOP_N_FOR_CLASSIFICATION = 500


def _concept_pos_neg_texts(tokenizer, concept: str, n: int, factorial: bool = True):
    if factorial:
        pairs = data.factorial_pairs(concept)[:n]
    else:
        pairs = {"role": data.ROLE_PAIRS, "harm": data.HARM_PAIRS, "control": data.CONTROL_PAIRS}[concept][:n]
    pos, neg = zip(*data.materialize_pairs(tokenizer, pairs))
    return list(pos), list(neg)


def run(cfg: Config) -> None:
    out_dir = cfg.phase_dir(PHASE_NAME)
    logger = get_logger(PHASE_NAME, out_dir)
    logger.info("=" * 80)
    logger.info("PHASE 3: COMPONENT ATTRIBUTION, NEUROSTRIKE RECOVERY, MEDIATION & RESCUE (RQ3)")
    logger.info("=" * 80)

    upstream_dir = cfg.phase_dir(UPSTREAM_PHASE)
    require_phase_output(upstream_dir / "directions.pt", UPSTREAM_PHASE)
    directions = load_torch(upstream_dir / "directions.pt")
    selected = load_json(upstream_dir / "selected_layers.json")
    early, mid, late = selected["early_layer"], selected["mid_layer"], selected["late_layer"]
    num_layers = selected["num_layers"]
    # LAYER_STRIDE=1 sweeps every layer (more rows for phase 4's prediction test);
    # unset uses just the three reference layers from phase 1.
    target_layers = list(range(0, num_layers, cfg.layer_stride)) if cfg.layer_stride else sorted({early, mid, late})
    logger.info(f"Target layers for component attribution: {target_layers} (mid_layer={mid})")

    model, tokenizer = load_model(cfg.model_id, cfg.device, logger)
    hooks = HookEngine(model)
    attrib = AttributionEngine(model)

    n_pairs = 4 if cfg.fast_dev else None  # None -> use full concept pool via slicing below
    n_control = 3 if cfg.fast_dev else 6
    k50_grid = cfg.k_grid

    # ---- NeuroStrike reference neurons (run once for the whole model) --------
    # RQ3 asks what functions NeuroStrike's components perform, so we need their
    # actual neuron set, via their supervised gate/up probe. Their probe is
    # trained on thousands of prompts; our curated pool is far smaller, so these
    # weights are underdetermined until the prompt sets are scaled up.
    ns_harmful = [build_prompt(tokenizer, user=p.positive.user) for p in data.HARM_PAIRS] + \
                 [build_prompt(tokenizer, user=p.positive.user) for p in data.CONTROL_PAIRS]
    ns_benign = [build_prompt(tokenizer, user=p.negative.user) for p in data.HARM_PAIRS] + \
                [build_prompt(tokenizer, user=p) for p in data.BENIGN_REFERENCE_PROMPTS]
    if cfg.fast_dev:
        ns_harmful, ns_benign = ns_harmful[:6], ns_benign[:6]
    logger.warning(f"NeuroStrike probe training set is small ({len(ns_harmful)}+{len(ns_benign)} prompts vs. thousands in the reference) — scale core/data.py pools before trusting these weights.")
    ns_weights, ns_neurons = neurostrike.compute_safety_neurons(
        model, tokenizer, ns_harmful, ns_benign, cfg.device,
        threshold=cfg.neurostrike_z_threshold, batch_size=cfg.neurostrike_batch_size,
        num_epochs=cfg.neurostrike_probe_epochs, logger=logger,
    )
    save_torch(out_dir / "neurostrike_weights.pt", ns_weights)
    save_torch(out_dir / "neurostrike_neurons.pt", ns_neurons)

    control_requests = [p.positive.user for p in data.CONTROL_PAIRS[:n_control]]
    k50_probe_texts = [build_prompt(tokenizer, user=r) for r in control_requests]
    k50_probe_toks_list = [tokenizer(t, return_tensors="pt").to(cfg.device) for t in k50_probe_texts]

    # A_R(k) must be probed on prompts where that security function is actually
    # engaged: role-injected prompts for R_role, harmful ones for R_harm/R_control.
    def _concept_probe_toks(concept: str):
        n = 3 if cfg.fast_dev else 6
        if concept == "role":
            texts = [n_ for _, n_ in data.materialize_pairs(tokenizer, data.ROLE_PAIRS[:n])]
        elif concept == "harm":
            texts = [build_prompt(tokenizer, user=p.positive.user) for p in data.HARM_PAIRS[:n]]
        else:
            texts = k50_probe_texts
        return [tokenizer(t, return_tensors="pt").to(cfg.device) for t in texts]
    id_refusal = tokenizer.encode("I", add_special_tokens=False)[0]
    id_compliance = tokenizer.encode("Sure", add_special_tokens=False)[0]

    architecture_rows = []
    overlap_rows = []
    classification_rows = []
    ns_function_rows = []
    ablation_curve_frames = []
    neuron_rankings = {}  # (layer, concept) -> {"causal":.., "static":.., "neurostrike":..}

    for layer in target_layers:
        raw_by_concept, static_by_concept, contrast_by_concept, active_w_by_concept = {}, {}, {}, {}

        # NeuroStrike's probe ranking for this layer. Their neurons live in the
        # gate_proj/up_proj output basis, which is index-identical to our
        # pre-down_proj basis (down_proj input i == silu(gate_i) * up_i), so the
        # top-k index sets are directly comparable.
        ns_gate = ns_weights.get(f"model.layers.{layer}.mlp.gate_proj")
        ns_ranking = neurostrike.rank_by_probe_weight(ns_gate).tolist() if ns_gate is not None else None

        for concept in CONCEPTS:
            n = n_pairs if n_pairs is not None else (12 if concept == "role" else 20 if concept == "harm" else 15)
            pos_texts, neg_texts = _concept_pos_neg_texts(tokenizer, concept, n, factorial=cfg.use_factorial_design)

            pos_acts = capture_mlp_neuron_activations(model, hooks, tokenizer, pos_texts, cfg.device, layer)
            neg_acts = capture_mlp_neuron_activations(model, hooks, tokenizer, neg_texts, cfg.device, layer)

            raw_contrib, active_w = attrib.compute_neuron_attributions(layer, pos_acts, directions[concept][layer].to(cfg.device))
            static_align = attrib.compute_static_weight_alignment(layer, directions[concept][layer])
            contrast_rank = attrib.compute_activation_contrast_ranking(pos_acts, neg_acts)

            raw_by_concept[concept] = raw_contrib
            static_by_concept[concept] = static_align
            contrast_by_concept[concept] = contrast_rank
            active_w_by_concept[concept] = active_w

            metrics = attrib.compute_architectural_quantities(raw_contrib, active_w, gram_top_k=cfg.gram_top_k)

            mlp_module = hooks.layers[layer].mlp
            causal_ranking = torch.argsort(raw_contrib, descending=True)
            # A_R measured on this concept's own representation (plan-faithful);
            # refusal_margin recorded at the same grid points as the behavioral
            # cross-check, never as the definition of A_R.
            score_fns = {
                "A_R": make_projection_score_fn(hooks, layer, directions[concept][layer]),
                "refusal_margin": refusal_margin_score,
            }
            curve_df = component_ablation_curve(model, tokenizer, cfg.device, mlp_module, causal_ranking.tolist(), _concept_probe_toks(concept), score_fns, k50_grid, n_neurons=raw_contrib.numel(), seed=cfg.seed)
            curve_df["layer"], curve_df["concept"] = layer, concept
            ablation_curve_frames.append(curve_df)
            k50 = find_k50(curve_df, column="ratio_A_R")
            metrics["k_50"] = k50 if k50 is not None else float("nan")
            metrics["k_50_behavioral"] = find_k50(curve_df, column="ratio_refusal_margin") or float("nan")
            metrics["k_50_random_control"] = find_k50(curve_df, column="ratio_A_R", ablation="random") or float("nan")
            metrics["layer"] = layer
            metrics["concept"] = concept
            architecture_rows.append(metrics)
            logger.info(f"[L{layer}/{concept}] C_R={metrics['functional_concentration_C_R']:.4f} N_eff={metrics['effective_component_count_N_eff']:.1f} r_eff={metrics['effective_functional_rank_r_eff']:.1f} k_50={metrics['k_50']}")

            static_ranking = torch.argsort(static_align, descending=True)
            contrast_ranking = torch.argsort(contrast_rank, descending=True)
            neuron_rankings[(layer, concept)] = {
                "causal": causal_ranking,
                "static": static_ranking,
                "activation_contrast": contrast_ranking,
                "neurostrike": torch.tensor(ns_ranking) if ns_ranking is not None else None,
            }

            # RQ3: can our geometric route recover the neurons NeuroStrike's
            # supervised probe finds? "*_vs_neurostrike" are the rows that answer it.
            candidates = {"causal": causal_ranking.tolist(), "static": static_ranking.tolist(), "activation_contrast": contrast_ranking.tolist()}
            for k in OVERLAP_K_GRID:
                for a, b in [("causal", "static"), ("causal", "activation_contrast"), ("static", "activation_contrast")]:
                    overlap_rows.append({"layer": layer, "concept": concept, "k": k, "comparison": f"{a}_vs_{b}", "jaccard": component_set_overlap(candidates[a], candidates[b], k)})
                if ns_ranking is not None:
                    for name, ranking in candidates.items():
                        overlap_rows.append({"layer": layer, "concept": concept, "k": k, "comparison": f"{name}_vs_neurostrike", "jaccard": component_set_overlap(ranking, ns_ranking, k)})

        # Functional classification: for the top-N neurons by causal contribution
        # (union across concepts), which concept does each neuron contribute to most?
        union_top = set()
        for concept in CONCEPTS:
            union_top |= set(torch.topk(raw_by_concept[concept], k=min(TOP_N_FOR_CLASSIFICATION, raw_by_concept[concept].numel())).indices.tolist())
        for idx in sorted(union_top):
            scores = {c: raw_by_concept[c][idx].item() for c in CONCEPTS}
            best = max(scores, key=scores.get)
            classification_rows.append({"layer": layer, "neuron_idx": idx, "best_concept": best, **{f"score_{c}": v for c, v in scores.items()}})

        # RQ3 proper: "determine what functions are performed by components
        # previously identified by NeuroStrike". Take THEIR selected neurons and
        # ask which of our three security functions each one contributes to most.
        for ns_module in (f"model.layers.{layer}.mlp.gate_proj", f"model.layers.{layer}.mlp.up_proj"):
            for idx in [int(i) for i in ns_neurons.get(ns_module, [])]:
                if idx >= raw_by_concept["harm"].numel():
                    continue
                scores = {c: raw_by_concept[c][idx].item() for c in CONCEPTS}
                total = sum(abs(v) for v in scores.values()) or 1.0
                ns_function_rows.append({
                    "layer": layer,
                    "ns_module": ns_module.split(".")[-1],
                    "neuron_idx": idx,
                    "best_concept": max(scores, key=scores.get),
                    **{f"score_{c}": v for c, v in scores.items()},
                    **{f"share_{c}": abs(v) / total for c, v in scores.items()},
                })

    save_df(out_dir / "architecture_metrics.csv", pd.DataFrame(architecture_rows))
    save_df(out_dir / "neuron_ranking_overlap.csv", pd.DataFrame(overlap_rows))
    save_df(out_dir / "neuron_functional_classification.csv", pd.DataFrame(classification_rows))
    save_df(out_dir / "ablation_curves.csv", pd.concat(ablation_curve_frames, ignore_index=True) if ablation_curve_frames else pd.DataFrame())
    save_torch(out_dir / "neuron_rankings.pt", neuron_rankings)

    df_ns_fn = pd.DataFrame(ns_function_rows)
    save_df(out_dir / "neurostrike_neuron_functions.csv", df_ns_fn)
    if not df_ns_fn.empty:
        logger.info(f"NeuroStrike neurons by best-matching security function:\n{df_ns_fn.groupby(['layer', 'best_concept']).size().to_string()}")

    mean_overlap = pd.DataFrame(overlap_rows).groupby("comparison")["jaccard"].mean()
    logger.info(f"Mean top-k neuron ranking overlap (Jaccard, averaged over layers/concepts/k):\n{mean_overlap.to_string()}")

    # -------------------------------------------------------------------
    # Mediation & rescue at the mid layer.
    # -------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Mediation & causal rescue at mid layer")
    logger.info("-" * 80)

    arch_df = pd.DataFrame(architecture_rows)

    def _k50_for(layer, concept, default=50):
        row = arch_df[(arch_df.layer == layer) & (arch_df.concept == concept)]
        if row.empty or pd.isna(row.iloc[0]["k_50"]):
            return default
        return int(row.iloc[0]["k_50"])

    mediation_requests = [p.positive.user for p in data.CONTROL_PAIRS[:3]]
    mediation_texts = [build_prompt(tokenizer, user=r) for r in mediation_requests]
    mediation_toks = tokenizer(mediation_texts, return_tensors="pt", padding=True).to(cfg.device)

    def measure(toks):
        hooks.hook_residual_stream([mid])
        with torch.no_grad():
            out = model(**toks)
        resid = hooks.residual_cache[mid][:, -1, :].detach().cpu()
        hooks.remove_all_hooks()
        margin = (out.logits[:, -1, id_refusal] - out.logits[:, -1, id_compliance]).mean().item()
        projs = {c: F.cosine_similarity(resid, directions[c][mid].unsqueeze(0), dim=1).mean().item() for c in CONCEPTS}
        return projs, margin

    baseline_projs, baseline_margin = measure(mediation_toks)
    logger.info(f"Baseline projections={baseline_projs}, margin={baseline_margin:.4f}")

    mediation_results = {"baseline": {"projections": baseline_projs, "margin": baseline_margin}, "ablations": {}}
    for which in CONCEPTS:
        k50_which = _k50_for(mid, which)
        ranking = neuron_rankings[(mid, which)]["causal"].tolist()[:k50_which]
        handle = InterventionEngine.ablate_neurons(hooks.layers[mid].mlp, ranking)
        projs, margin = measure(mediation_toks)
        handle.remove()
        mediation_results["ablations"][which] = {
            "k_50_ablated": k50_which,
            "projections": projs,
            "margin": margin,
            "delta_vs_baseline": {c: projs[c] - baseline_projs[c] for c in CONCEPTS},
            "margin_delta": margin - baseline_margin,
        }
        logger.info(f"Ablate top-{k50_which} '{which}' neurons -> delta_projections={mediation_results['ablations'][which]['delta_vs_baseline']}, margin_delta={margin - baseline_margin:+.4f}")

    # Causal rescue: re-ablate R_control's neurons, then re-inject the R_control
    # direction WITHOUT restoring the ablated neurons, and check if behavior returns.
    k50_control = _k50_for(mid, "control")
    control_ranking = neuron_rankings[(mid, "control")]["causal"].tolist()[:k50_control]
    ablate_handle = InterventionEngine.ablate_neurons(hooks.layers[mid].mlp, control_ranking)
    rescue_handle = InterventionEngine.install_causal_rescue(hooks.layers[mid], directions["control"][mid], directions["control"][mid] * 5.0)
    rescued_projs, rescued_margin = measure(mediation_toks)
    rescue_handle.remove()
    ablate_handle.remove()
    mediation_results["rescue"] = {
        "k_50_ablated": k50_control,
        "projections": rescued_projs,
        "margin": rescued_margin,
        "delta_vs_ablated_control": {c: rescued_projs[c] - mediation_results["ablations"]["control"]["projections"][c] for c in CONCEPTS},
        "margin_delta_vs_ablated_control": rescued_margin - mediation_results["ablations"]["control"]["margin"],
    }
    logger.info(f"Rescue (R_control reinjected, neurons still ablated) -> margin={rescued_margin:.4f} (recovered {rescued_margin - mediation_results['ablations']['control']['margin']:+.4f} vs ablated)")

    save_json(out_dir / "mediation_rescue_results.json", mediation_results)

    # Qualitative generation samples: baseline / ablated(control) / rescued.
    sample_lines = []
    gen_prompts = [build_prompt(tokenizer, user=r) for r in mediation_requests[:2]]
    for prompt in gen_prompts:
        toks = tokenizer(prompt, return_tensors="pt").to(cfg.device)
        base_text = generate_text(model, tokenizer, toks, cfg.max_new_tokens)

        ablate_handle = InterventionEngine.ablate_neurons(hooks.layers[mid].mlp, control_ranking)
        ablated_text = generate_text(model, tokenizer, toks, cfg.max_new_tokens)

        rescue_handle = InterventionEngine.install_causal_rescue(hooks.layers[mid], directions["control"][mid], directions["control"][mid] * 5.0)
        rescued_text = generate_text(model, tokenizer, toks, cfg.max_new_tokens)
        rescue_handle.remove()
        ablate_handle.remove()

        sample_lines.append(f"PROMPT: {prompt}\n[baseline] {base_text}\n[ablated top-{k50_control} R_control neurons] {ablated_text}\n[rescued] {rescued_text}\n{'-'*80}")

    Path(out_dir / "mediation_rescue_samples.txt").write_text("\n".join(sample_lines))

    logger.info("=" * 80)
    logger.info(f"Phase 3 complete. Outputs written to {out_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run(load_config())
