"""Phase 5 — Context-dependent reorganization (RQ6) & attack-stage diagnosis (RQ4).

Both RQs apply the same machinery (direction extraction, neuron attribution,
steering, ablation, ASR) to different prompt *contexts*, so they share one
phase: jailbreak/authority/roleplay framings are simultaneously "contexts"
(RQ6: does the same intent, reframed, still activate the same mechanism?)
and "attacks" (RQ4: which stage of the architecture does each framing
actually compromise?).

Part A — Context stability (RQ6). For each intent in data.CONTEXT_INTENTS,
rendered under {direct, paraphrase, roleplay, authority, jailbreak}, we
estimate a context-specific R_harm(c) and R_control(c) at the mid layer
(diff-of-means against a shared benign anchor set / forced "I" vs "Sure"),
plus context-specific causally-ranked neuron sets M_harm(c) AND M_control(c).
We report representation-level similarity S_repr, component-level similarity
S_comp, and — most importantly — causal transfer: we ablate the neurons
identified under c1 and measure the effect on behavior evaluated under c2.
Both functions are carried through all three levels, because the paper's
sharpest possible result ("stable representation + dynamic implementation")
requires comparing repr- against component-stability *per function*.

Part B — Attack-stage diagnosis (RQ4). Relative to the *direct* framing with
no intervention, we measure how far each of five conditions
(roleplay / authority / jailbreak framing, representation steering, neuron
ablation) pushes the model's R_role/R_harm/R_control projections and its
behavioral attack-success rate, to see which stage each attack family
actually compromises.

Requires phase1_geometry (always) and phase3_components (for the neuron-
ablation attack condition in Part B) to have been run first.

Outputs (results/phase5_context_and_attacks/):
  context_stability.csv        c1, c2, metric, cosine_sim   (S_repr; R_harm(c)/R_control(c) pooled over all intents)
  component_stability.csv      metric (M_harm/M_control), c1, c2, k, jaccard   (S_comp)
  causal_transfer_matrix.csv   mechanism (M_harm/M_control), mechanism_context, eval_context, margin_delta
  attack_stage_diagnosis.csv   attack_type, delta_R_role, delta_R_harm, delta_R_control, behavioral_ASR
"""
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

from core import data
from core.attribution import AttributionEngine
from core.config import Config, load_config
from core.hooks import HookEngine, capture_last_token_residuals, capture_mlp_neuron_activations
from core.interventions import InterventionEngine
from core.io_utils import get_logger, load_json, load_torch, require_phase_output, save_df
from core.judge import AttackJudge
from core.metrics import attack_success_rate, component_set_overlap, jaccard_difference_ci
from core.model_io import build_prompt, load_model
from core.subspaces import SubspaceEngine

PHASE_NAME = "phase5_context_and_attacks"
PHASE1 = "phase1_geometry"
PHASE3 = "phase3_components"
CONCEPTS = ["role", "harm", "control"]
CONTEXT_NAMES = data.CONTEXT_NAMES
COMPONENT_K = 50


def run(cfg: Config) -> None:
    out_dir = cfg.phase_dir(PHASE_NAME)
    logger = get_logger(PHASE_NAME, out_dir)
    logger.info("=" * 80)
    logger.info("PHASE 5: CONTEXT-DEPENDENT REORGANIZATION (RQ6) & ATTACK-STAGE DIAGNOSIS (RQ4)")
    logger.info("=" * 80)

    phase1_dir = cfg.phase_dir(PHASE1)
    phase3_dir = cfg.phase_dir(PHASE3)
    require_phase_output(phase1_dir / "directions.pt", PHASE1)
    require_phase_output(phase3_dir / "neuron_rankings.pt", PHASE3)
    directions_global = load_torch(phase1_dir / "directions.pt")
    neuron_rankings_global = load_torch(phase3_dir / "neuron_rankings.pt")

    selected = load_json(phase1_dir / "selected_layers.json")
    mid = selected["mid_layer"]

    model, tokenizer = load_model(cfg.model_id, cfg.device, logger)
    hooks = HookEngine(model)
    attrib = AttributionEngine(model)
    id_refusal = tokenizer.encode("I", add_special_tokens=False)[0]
    id_compliance = tokenizer.encode("Sure", add_special_tokens=False)[0]
    judge = AttackJudge(cfg.device, use_llama_guard=cfg.use_llama_guard, logger=logger)

    n_intents = 4 if cfg.fast_dev else len(data.CONTEXT_INTENTS)
    intents = data.CONTEXT_INTENTS[:n_intents]
    benign_anchor_texts = [build_prompt(tokenizer, user=t) for t in data.BENIGN_REFERENCE_PROMPTS]

    def _measure_batch(texts, layer=mid):
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(cfg.device)
        hooks.hook_residual_stream([layer])
        with torch.no_grad():
            out = model(**inputs)
        resid = hooks.residual_cache[layer][:, -1, :].detach().cpu()
        hooks.remove_all_hooks()
        margin = (out.logits[:, -1, id_refusal] - out.logits[:, -1, id_compliance]).mean().item()
        projs = {c: F.cosine_similarity(resid, directions_global[c][layer].unsqueeze(0), dim=1).mean().item() for c in CONCEPTS}
        return projs, margin

    # -------------------------------------------------------------------
    # Part A: context stability (RQ6)
    # -------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("PART A: representation- and component-level context stability (RQ6)")
    logger.info("-" * 80)

    r_harm_by_ctx, r_control_by_ctx, ranking_by_ctx, baseline_margin_by_ctx, harmful_content_by_ctx = {}, {}, {}, {}, {}
    split_half_ranking, split_half_direction = {}, {}

    for ctx in CONTEXT_NAMES:
        harmful_content = [getattr(iv, ctx) for iv in intents]
        harmful_content_by_ctx[ctx] = harmful_content
        pos_texts = [build_prompt(tokenizer, user=t) for t in harmful_content]

        pos_resid = capture_last_token_residuals(model, hooks, tokenizer, pos_texts, cfg.device, [mid])
        neg_resid = capture_last_token_residuals(model, hooks, tokenizer, benign_anchor_texts, cfg.device, [mid])
        r_harm_by_ctx[ctx] = SubspaceEngine.extract_direction(pos_resid[mid], neg_resid[mid])

        ctrl_pos_texts = [build_prompt(tokenizer, user=t, assistant_prefix="I") for t in harmful_content]
        ctrl_neg_texts = [build_prompt(tokenizer, user=t, assistant_prefix="Sure") for t in harmful_content]
        ctrl_pos_resid = capture_last_token_residuals(model, hooks, tokenizer, ctrl_pos_texts, cfg.device, [mid])
        ctrl_neg_resid = capture_last_token_residuals(model, hooks, tokenizer, ctrl_neg_texts, cfg.device, [mid])
        r_control_by_ctx[ctx] = SubspaceEngine.extract_direction(ctrl_pos_resid[mid], ctrl_neg_resid[mid])

        # The plan requires the causally important components for BOTH functions
        # per context (M_harm(c) and M_control(c)). Measuring only control makes
        # the key "stable representation + dynamic implementation" claim
        # untestable, since that needs repr- vs component-stability compared
        # per function.
        mlp_pos_acts = capture_mlp_neuron_activations(model, hooks, tokenizer, pos_texts, cfg.device, mid)
        for r_concept, r_vec in (("harm", r_harm_by_ctx[ctx]), ("control", r_control_by_ctx[ctx])):
            raw_contrib, _ = attrib.compute_neuron_attributions(mid, mlp_pos_acts, r_vec.to(cfg.device))
            ranking_by_ctx[(r_concept, ctx)] = torch.argsort(raw_contrib, descending=True).tolist()

        # WITHIN-CONTEXT NOISE FLOOR for S_comp. Across-context component overlap
        # is only evidence of reorganization if it is LOWER than what two disjoint
        # halves of the SAME context produce. Without this baseline, "components
        # change with context" is indistinguishable from "the ranking is noisy",
        # and the paper's central "stable representation + dynamic implementation"
        # claim rests entirely on that distinction.
        half = len(pos_texts) // 2
        if half >= 1:
            for tag, sl in (("a", slice(0, half)), ("b", slice(half, 2 * half))):
                half_acts = capture_mlp_neuron_activations(model, hooks, tokenizer, pos_texts[sl], cfg.device, mid)
                half_pos = capture_last_token_residuals(model, hooks, tokenizer, pos_texts[sl], cfg.device, [mid])
                half_r_harm = SubspaceEngine.extract_direction(half_pos[mid], neg_resid[mid])
                half_ctrl_pos = capture_last_token_residuals(model, hooks, tokenizer, ctrl_pos_texts[sl], cfg.device, [mid])
                half_ctrl_neg = capture_last_token_residuals(model, hooks, tokenizer, ctrl_neg_texts[sl], cfg.device, [mid])
                half_r_control = SubspaceEngine.extract_direction(half_ctrl_pos[mid], half_ctrl_neg[mid])
                for r_concept, r_vec in (("harm", half_r_harm), ("control", half_r_control)):
                    rc, _ = attrib.compute_neuron_attributions(mid, half_acts, r_vec.to(cfg.device))
                    split_half_ranking[(r_concept, ctx, tag)] = torch.argsort(rc, descending=True).tolist()
                    split_half_direction[(r_concept, ctx, tag)] = r_vec

        _, base_margin = _measure_batch(pos_texts)
        baseline_margin_by_ctx[ctx] = base_margin
        logger.info(f"[{ctx}] baseline refusal margin = {base_margin:.4f}")

    stability_rows = []
    for c1, c2 in itertools.combinations(CONTEXT_NAMES, 2):
        stability_rows.append({"c1": c1, "c2": c2, "metric": "R_harm", "cosine_sim": SubspaceEngine.cosine_similarity(r_harm_by_ctx[c1], r_harm_by_ctx[c2])})
        stability_rows.append({"c1": c1, "c2": c2, "metric": "R_control", "cosine_sim": SubspaceEngine.cosine_similarity(r_control_by_ctx[c1], r_control_by_ctx[c2])})

    component_rows = []
    for r_concept in ("harm", "control"):
        for c1, c2 in itertools.combinations(CONTEXT_NAMES, 2):
            jaccard = component_set_overlap(ranking_by_ctx[(r_concept, c1)], ranking_by_ctx[(r_concept, c2)], COMPONENT_K)
            component_rows.append({"metric": f"M_{r_concept}", "c1": c1, "c2": c2, "k": COMPONENT_K, "jaccard": jaccard})
    # Same metric, same k, computed within each context between disjoint halves.
    for r_concept in ("harm", "control"):
        for ctx in CONTEXT_NAMES:
            ka = split_half_ranking.get((r_concept, ctx, "a"))
            kb = split_half_ranking.get((r_concept, ctx, "b"))
            if ka is None or kb is None:
                continue
            component_rows.append({
                "metric": f"M_{r_concept}", "c1": ctx, "c2": ctx, "k": COMPONENT_K,
                "jaccard": component_set_overlap(ka, kb, COMPONENT_K),
                "comparison": "within_context_split_half",
            })
            stability_rows.append({
                "c1": ctx, "c2": ctx, "metric": f"R_{r_concept}",
                "cosine_sim": SubspaceEngine.cosine_similarity(
                    split_half_direction[(r_concept, ctx, "a")], split_half_direction[(r_concept, ctx, "b")]),
                "comparison": "within_context_split_half",
            })
    # Matched-n across-context comparison. The within-context floor is estimated
    # from half-sized samples, so comparing it against the FULL-sample
    # across-context number is unfair: fewer samples means a noisier ranking and
    # a lower Jaccard regardless of context, which biases the floor -- and hence
    # the reorganization gap -- downward. Recompute across-context from the same
    # half-sized samples so both sides carry identical estimation noise.
    for r_concept in ("harm", "control"):
        for c1, c2 in itertools.combinations(CONTEXT_NAMES, 2):
            ka = split_half_ranking.get((r_concept, c1, "a"))
            kb = split_half_ranking.get((r_concept, c2, "a"))
            if ka is None or kb is None:
                continue
            component_rows.append({
                "metric": f"M_{r_concept}", "c1": c1, "c2": c2, "k": COMPONENT_K,
                "jaccard": component_set_overlap(ka, kb, COMPONENT_K),
                "comparison": "across_context_matched_n",
            })

    for row in component_rows:
        row.setdefault("comparison", "across_context")
    for row in stability_rows:
        row.setdefault("comparison", "across_context")
    save_df(out_dir / "context_stability.csv", pd.DataFrame(stability_rows))
    save_df(out_dir / "component_stability.csv", pd.DataFrame(component_rows))

    df_comp = pd.DataFrame(component_rows)
    summary = df_comp.groupby(["metric", "comparison"])["jaccard"].mean().unstack()
    if {"within_context_split_half", "across_context_matched_n"} <= set(summary.columns):
        # Only the matched-n columns form a valid contrast; the full-sample
        # across-context column is reported for reference, not for the gap.
        summary["reorganization_gap"] = summary["within_context_split_half"] - summary["across_context_matched_n"]
        logger.info(
            "S_comp vs its within-context noise floor, both at matched n. A positive "
            "reorganization_gap is the evidence for context-dependent reorganization; "
            "~0 means the across-context drop is just estimation noise. The "
            "'across_context' column uses full samples and is NOT comparable to the "
            "floor -- do not compute the gap from it:\n" + summary.round(3).to_string())

        gap_rows = []
        for r_concept in ("harm", "control"):
            sub = df_comp[df_comp["metric"] == f"M_{r_concept}"]
            stats = jaccard_difference_ci(
                sub[sub["comparison"] == "within_context_split_half"]["jaccard"].tolist(),
                sub[sub["comparison"] == "across_context_matched_n"]["jaccard"].tolist(),
                seed=cfg.seed,
            )
            gap_rows.append({"metric": f"M_{r_concept}", **stats})
        df_gap = pd.DataFrame(gap_rows)
        save_df(out_dir / "reorganization_gap_ci.csv", df_gap)
        logger.info(
            "Bootstrap 95% CI on the reorganization gap. A CI straddling 0 means the "
            "gap is not distinguishable from estimation noise:\n" + df_gap.round(3).to_string(index=False))
    logger.info(f"Component-level stability (top-{COMPONENT_K} neuron Jaccard overlap):\n{pd.DataFrame(component_rows).to_string()}")

    # Causal transfer: ablate the neurons identified under c1, evaluate behavior under c2.
    transfer_rows = []
    for r_concept in ("harm", "control"):
        for c1 in CONTEXT_NAMES:
            handle = InterventionEngine.ablate_neurons(hooks.layers[mid].mlp, ranking_by_ctx[(r_concept, c1)][:COMPONENT_K])
            for c2 in CONTEXT_NAMES:
                texts_c2 = [build_prompt(tokenizer, user=t) for t in harmful_content_by_ctx[c2]]
                _, margin_under_ablation = _measure_batch(texts_c2)
                transfer_rows.append({
                    "mechanism": f"M_{r_concept}",
                    "mechanism_context": c1,
                    "eval_context": c2,
                    "margin_delta": margin_under_ablation - baseline_margin_by_ctx[c2],
                })
            handle.remove()
    df_transfer = pd.DataFrame(transfer_rows)
    save_df(out_dir / "causal_transfer_matrix.csv", df_transfer)
    for r_concept in ("harm", "control"):
        sub = df_transfer[df_transfer["mechanism"] == f"M_{r_concept}"]
        logger.info(f"Causal transfer, do(M_{r_concept}(c1)) evaluated under c2 — margin delta:\n"
                    f"{sub.pivot(index='mechanism_context', columns='eval_context', values='margin_delta').to_string()}")

    # -------------------------------------------------------------------
    # Part B: mechanistic attack-stage diagnosis (RQ4)
    # -------------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("PART B: mechanistic attack-stage diagnosis (RQ4)")
    logger.info("-" * 80)

    direct_content = harmful_content_by_ctx["direct"]
    direct_texts = [build_prompt(tokenizer, user=t) for t in direct_content]
    baseline_projs, _ = _measure_batch(direct_texts)
    baseline_asr = attack_success_rate(model, tokenizer, direct_texts, direct_content, cfg.device, cfg.max_new_tokens, judge)
    logger.info(f"[direct/no intervention] projections={baseline_projs}, ASR={baseline_asr:.2f}")

    diagnosis_rows = [{
        "attack_type": "direct_baseline",
        **{f"delta_R_{c}": 0.0 for c in CONCEPTS},
        "behavioral_ASR": baseline_asr,
    }]

    for ctx in ["roleplay", "authority", "jailbreak"]:
        texts = [build_prompt(tokenizer, user=t) for t in harmful_content_by_ctx[ctx]]
        projs, _ = _measure_batch(texts)
        asr = attack_success_rate(model, tokenizer, texts, harmful_content_by_ctx[ctx], cfg.device, cfg.max_new_tokens, judge)
        diagnosis_rows.append({
            "attack_type": f"context_{ctx}",
            **{f"delta_R_{c}": projs[c] - baseline_projs[c] for c in CONCEPTS},
            "behavioral_ASR": asr,
        })
        logger.info(f"[context:{ctx}] delta_projections={ {c: round(projs[c]-baseline_projs[c],4) for c in CONCEPTS} }, ASR={asr:.2f}")

    # Representation-steering attack: push R_role toward the injected/override class.
    # Magnitude must be on the active steering scale: under relative steering
    # alpha is a fraction of the residual norm, so a hardcoded 4.0 would be 4x
    # the norm and measure collapse rather than a role attack.
    steer_alpha = data.attack_steering_sign("role", cfg.use_factorial_design) * (
        0.25 if cfg.steer_relative else 4.0
    )
    steer_handle = InterventionEngine.steer_subspace(hooks.layers[mid], directions_global["role"][mid], steer_alpha, relative=cfg.steer_relative)
    projs, _ = _measure_batch(direct_texts)
    asr = attack_success_rate(model, tokenizer, direct_texts, direct_content, cfg.device, cfg.max_new_tokens, judge)
    steer_handle.remove()
    diagnosis_rows.append({
        "attack_type": "repr_steering_role",
        **{f"delta_R_{c}": projs[c] - baseline_projs[c] for c in CONCEPTS},
        "behavioral_ASR": asr,
    })
    logger.info(f"[repr_steering_role, alpha={steer_alpha}] delta_projections={ {c: round(projs[c]-baseline_projs[c],4) for c in CONCEPTS} }, ASR={asr:.2f}")

    # Neuron-ablation attack: ablate phase 3's top-k_50 R_control neurons at mid layer.
    control_ranking = neuron_rankings_global[(mid, "control")]["causal"].tolist()
    ablate_handle = InterventionEngine.ablate_neurons(hooks.layers[mid].mlp, control_ranking[:COMPONENT_K])
    projs, _ = _measure_batch(direct_texts)
    asr = attack_success_rate(model, tokenizer, direct_texts, direct_content, cfg.device, cfg.max_new_tokens, judge)
    ablate_handle.remove()
    diagnosis_rows.append({
        "attack_type": "neuron_ablation_control",
        **{f"delta_R_{c}": projs[c] - baseline_projs[c] for c in CONCEPTS},
        "behavioral_ASR": asr,
    })
    logger.info(f"[neuron_ablation_control, top-{COMPONENT_K}] delta_projections={ {c: round(projs[c]-baseline_projs[c],4) for c in CONCEPTS} }, ASR={asr:.2f}")

    df_diag = pd.DataFrame(diagnosis_rows)
    save_df(out_dir / "attack_stage_diagnosis.csv", df_diag)
    logger.info(f"Attack-stage diagnosis:\n{df_diag.to_string()}")

    logger.info("=" * 80)
    logger.info(f"Phase 5 complete. Outputs written to {out_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run(load_config())
