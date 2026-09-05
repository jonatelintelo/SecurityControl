"""Phase 2 — Cross-layer causal structure (RQ2).

Loads the frozen per-layer R_role/R_harm/R_control directions from phase 1
and builds a cross-intervention matrix: for each (source layer, source
concept) we additively steer the residual stream, then read out both the
representation-level effect (cosine-similarity shift toward each concept's
direction at a downstream target layer) and the behavioral effect (refusal
logit-margin shift) relative to an unsteered baseline. The resulting matrix
is the evidence used to argue for a sequential, parallel, or overlapping
causal structure between R_role, R_harm, and R_control.

Requires phase1_geometry to have been run first.

Outputs (results/phase2_causal_structure/):
  cross_intervention_matrix.csv   source_layer, source_concept, alpha, target_layer,
                                   behavior_margin_delta, cos_delta_role, cos_delta_harm, cos_delta_control
"""
import pandas as pd
import torch
import torch.nn.functional as F

from core import data
from core.config import Config, load_config
from core.hooks import HookEngine
from core.interventions import InterventionEngine
from core.io_utils import get_logger, load_json, load_torch, require_phase_output, save_df
from core.model_io import load_model

PHASE_NAME = "phase2_causal_structure"
UPSTREAM_PHASE = "phase1_geometry"
CONCEPTS = ["role", "harm", "control"]
STEER_ALPHA = 3.0


def run(cfg: Config) -> None:
    out_dir = cfg.phase_dir(PHASE_NAME)
    logger = get_logger(PHASE_NAME, out_dir)
    logger.info("=" * 80)
    logger.info("PHASE 2: CROSS-LAYER CAUSAL STRUCTURE (RQ2)")
    logger.info("=" * 80)

    upstream_dir = cfg.phase_dir(UPSTREAM_PHASE)
    require_phase_output(upstream_dir / "directions.pt", UPSTREAM_PHASE)
    directions = load_torch(upstream_dir / "directions.pt")
    selected = load_json(upstream_dir / "selected_layers.json")
    early, mid, late = selected["early_layer"], selected["mid_layer"], selected["late_layer"]
    logger.info(f"Loaded directions from {UPSTREAM_PHASE}. early={early}, mid={mid}, late={late}")

    model, tokenizer = load_model(cfg.model_id, cfg.device, logger)
    hooks = HookEngine(model)

    n_probes = 3 if cfg.fast_dev else 6
    probe_pairs = data.HARM_PAIRS[:n_probes]
    probe_texts = [pos for pos, _ in data.materialize_pairs(tokenizer, probe_pairs)]
    probe_inputs = tokenizer(probe_texts, return_tensors="pt", padding=True).to(cfg.device)
    id_refusal = tokenizer.encode("I", add_special_tokens=False)[0]
    id_compliance = tokenizer.encode("Sure", add_special_tokens=False)[0]

    layer_pairs = sorted({(early, mid), (early, late), (mid, late)})
    target_layers = sorted({t for _, t in layer_pairs})
    logger.info(f"(source_layer, target_layer) pairs: {layer_pairs}")

    # Unsteered baseline: residual at every target layer + refusal margin.
    hooks.hook_residual_stream(target_layers)
    with torch.no_grad():
        base_out = model(**probe_inputs)
    baseline_margin = (base_out.logits[:, -1, id_refusal] - base_out.logits[:, -1, id_compliance]).mean().item()
    baseline_residual = {l: hooks.residual_cache[l][:, -1, :].detach().cpu() for l in target_layers}
    hooks.remove_all_hooks()
    logger.info(f"Baseline refusal-margin (mean over probes): {baseline_margin:.4f}")

    rows = []
    for L_src, L_tgt in layer_pairs:
        for concept in CONCEPTS:
            direction = directions[concept][L_src]

            # Steering hook must be registered BEFORE the residual-capture hook: when
            # L_src == L_tgt both hooks land on the same module, and forward hooks
            # fire in registration order — capture-before-steer would read the
            # pre-steering value and silently zero out every cos_delta column.
            steer_handle = InterventionEngine.steer_subspace(hooks.layers[L_src], direction, STEER_ALPHA)
            hooks.hook_residual_stream([L_tgt])
            with torch.no_grad():
                out = model(**probe_inputs)
            steer_handle.remove()
            steered_residual = hooks.residual_cache[L_tgt][:, -1, :].detach().cpu()
            hooks.remove_all_hooks()

            steered_margin = (out.logits[:, -1, id_refusal] - out.logits[:, -1, id_compliance]).mean().item()

            row = {
                "source_layer": L_src,
                "source_concept": concept,
                "alpha": STEER_ALPHA,
                "target_layer": L_tgt,
                "behavior_margin_delta": steered_margin - baseline_margin,
            }
            for target_concept in CONCEPTS:
                td = directions[target_concept][L_tgt]
                cos_steered = F.cosine_similarity(steered_residual, td.unsqueeze(0), dim=1)
                cos_base = F.cosine_similarity(baseline_residual[L_tgt], td.unsqueeze(0), dim=1)
                row[f"cos_delta_{target_concept}"] = (cos_steered - cos_base).mean().item()
            rows.append(row)

            summary = ", ".join(f"cos_delta_{c}={row[f'cos_delta_{c}']:.4f}" for c in CONCEPTS)
            logger.info(f"steer[{concept}@L{L_src}] -> read@L{L_tgt}: behavior_delta={row['behavior_margin_delta']:+.4f}  {summary}")

    df = pd.DataFrame(rows)
    save_df(out_dir / "cross_intervention_matrix.csv", df)
    logger.info(f"Full cross-intervention matrix:\n{df.to_string()}")

    logger.info("=" * 80)
    logger.info(f"Phase 2 complete. Outputs written to {out_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run(load_config())
