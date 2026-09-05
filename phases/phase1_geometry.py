"""Phase 1 — Latent variable extraction & geometry (RQ1).

Extracts per-layer R_role, R_harm, R_control diff-of-means directions from
core.data's contrastive pairs, validates each direction with a held-out
sign-classification accuracy, computes the full pairwise cosine-similarity
geometry across layers, and projects a labeled probe set onto every
direction to check projection correlations. Freezes directions.pt + the
selected reference layers for every later phase to consume.

Outputs (results/phase1_geometry/):
  directions.pt            {concept: {layer: unit_direction_tensor}}
  selected_layers.json      {mid_layer, early_layer, late_layer, num_layers}
  probe_accuracy.csv        layer, concept, accuracy
  geometry_cosine.csv       layer, concept_a, concept_b, cosine_sim
  projections.csv           prompt_id, label, layer, concept, projection
  projection_correlations.csv  layer, concept_a, concept_b, pearson_r
"""
from typing import List

import pandas as pd
import torch

from core import data
from core.attribution import AttributionEngine  # noqa: F401 (re-exported for phase2 convenience imports)
from core.config import Config, load_config
from core.hooks import HookEngine, capture_last_token_residuals
from core.io_utils import get_logger, save_df, save_json, save_torch
from core.model_io import build_prompt, load_model
from core.subspaces import SubspaceEngine

PHASE_NAME = "phase1_geometry"
CONCEPTS = ["role", "harm", "control"]


def _shrink(pairs: List, cfg: Config, n: int) -> List:
    return pairs[:n] if cfg.fast_dev else pairs


def _concept_pairs(cfg: Config):
    return {
        "role": _shrink(data.ROLE_PAIRS, cfg, 4),
        "harm": _shrink(data.HARM_PAIRS, cfg, 6),
        "control": _shrink(data.CONTROL_PAIRS, cfg, 6),
    }


def run(cfg: Config) -> None:
    out_dir = cfg.phase_dir(PHASE_NAME)
    logger = get_logger(PHASE_NAME, out_dir)
    logger.info("=" * 80)
    logger.info("PHASE 1: LATENT VARIABLE EXTRACTION & GEOMETRY (RQ1)")
    logger.info("=" * 80)

    model, tokenizer = load_model(cfg.model_id, cfg.device, logger)
    hooks = HookEngine(model)
    all_layers = list(range(hooks.num_layers))

    pairs_by_concept = _concept_pairs(cfg)

    directions = {}          # concept -> {layer: tensor}
    probe_acc_rows = []       # layer, concept, accuracy
    concept_texts = {}        # concept -> (pos_texts, neg_texts) over the FULL pair set (for projections)

    for concept, pairs in pairs_by_concept.items():
        train_pairs, heldout_pairs = data.split_pairs(pairs, cfg.train_fraction, seed=cfg.seed)
        logger.info(f"[{concept}] {len(pairs)} pairs total -> {len(train_pairs)} train / {len(heldout_pairs)} held-out")

        train_pos = [p for p, _ in data.materialize_pairs(tokenizer, train_pairs)]
        train_neg = [n for _, n in data.materialize_pairs(tokenizer, train_pairs)]
        pos_acts = capture_last_token_residuals(model, hooks, tokenizer, train_pos, cfg.device, all_layers)
        neg_acts = capture_last_token_residuals(model, hooks, tokenizer, train_neg, cfg.device, all_layers)

        directions[concept] = {
            l: SubspaceEngine.extract_direction(pos_acts[l], neg_acts[l]) for l in all_layers
        }

        if heldout_pairs:
            held_pos = [p for p, _ in data.materialize_pairs(tokenizer, heldout_pairs)]
            held_neg = [n for _, n in data.materialize_pairs(tokenizer, heldout_pairs)]
            held_pos_acts = capture_last_token_residuals(model, hooks, tokenizer, held_pos, cfg.device, all_layers)
            held_neg_acts = capture_last_token_residuals(model, hooks, tokenizer, held_neg, cfg.device, all_layers)
            for l in all_layers:
                acc = SubspaceEngine.probe_validation_accuracy(directions[concept][l], held_pos_acts[l], held_neg_acts[l])
                probe_acc_rows.append({"layer": l, "concept": concept, "accuracy": acc})

        full_pos, full_neg = zip(*data.materialize_pairs(tokenizer, pairs))
        concept_texts[concept] = (list(full_pos), list(full_neg))

    save_torch(out_dir / "directions.pt", directions)
    save_df(out_dir / "probe_accuracy.csv", pd.DataFrame(probe_acc_rows))
    if probe_acc_rows:
        mean_acc = pd.DataFrame(probe_acc_rows).groupby("concept")["accuracy"].mean()
        logger.info(f"Mean held-out probe accuracy per concept:\n{mean_acc.to_string()}")

    # --- Geometry: pairwise cosine similarity across all layers ---------------
    geometry_rows = []
    for l in all_layers:
        for i, a in enumerate(CONCEPTS):
            for b in CONCEPTS[i + 1:]:
                cos = SubspaceEngine.cosine_similarity(directions[a][l], directions[b][l])
                geometry_rows.append({"layer": l, "concept_a": a, "concept_b": b, "cosine_sim": cos})
    df_geom = pd.DataFrame(geometry_rows)
    save_df(out_dir / "geometry_cosine.csv", df_geom)

    harm_control = df_geom[(df_geom.concept_a == "harm") & (df_geom.concept_b == "control")].set_index("layer")["cosine_sim"]
    mid_layer = int(harm_control.abs().idxmax())
    early_layer = max(0, hooks.num_layers // 3)
    late_layer = min(hooks.num_layers - 1, (2 * hooks.num_layers) // 3)
    selected = {"mid_layer": mid_layer, "early_layer": early_layer, "late_layer": late_layer, "num_layers": hooks.num_layers}
    save_json(out_dir / "selected_layers.json", selected)
    logger.info(f"Selected layers: {selected} (mid_layer = argmax|cos(R_harm, R_control)| = {harm_control.abs().max():.4f})")

    # --- Projections onto each direction, across a labeled probe set ----------
    probe_texts, probe_ids, probe_labels = [], [], []
    for concept, (pos_texts, neg_texts) in concept_texts.items():
        for i, t in enumerate(pos_texts):
            probe_texts.append(t); probe_ids.append(f"{concept}_pos_{i}"); probe_labels.append(f"{concept}_positive")
        for i, t in enumerate(neg_texts):
            probe_texts.append(t); probe_ids.append(f"{concept}_neg_{i}"); probe_labels.append(f"{concept}_negative")
    for i, p in enumerate(data.BENIGN_REFERENCE_PROMPTS):
        probe_texts.append(build_prompt(tokenizer, user=p)); probe_ids.append(f"benign_{i}"); probe_labels.append("benign")

    if cfg.fast_dev:
        probe_texts, probe_ids, probe_labels = probe_texts[:16], probe_ids[:16], probe_labels[:16]

    logger.info(f"Projecting {len(probe_texts)} probe prompts onto {len(CONCEPTS)} concepts across {len(all_layers)} layers...")
    probe_acts = capture_last_token_residuals(model, hooks, tokenizer, probe_texts, cfg.device, all_layers)

    projection_rows = []
    proj_by_layer_concept = {}  # (layer, concept) -> list[float] aligned with probe_ids
    for l in all_layers:
        for concept in CONCEPTS:
            d = directions[concept][l]
            projs = (probe_acts[l] @ (d / torch.norm(d))).tolist()
            proj_by_layer_concept[(l, concept)] = projs
            for pid, label, proj in zip(probe_ids, probe_labels, projs):
                projection_rows.append({"prompt_id": pid, "label": label, "layer": l, "concept": concept, "projection": proj})
    save_df(out_dir / "projections.csv", pd.DataFrame(projection_rows))

    corr_rows = []
    for l in all_layers:
        for i, a in enumerate(CONCEPTS):
            for b in CONCEPTS[i + 1:]:
                x = pd.Series(proj_by_layer_concept[(l, a)])
                y = pd.Series(proj_by_layer_concept[(l, b)])
                r = x.corr(y) if x.std() > 0 and y.std() > 0 else float("nan")
                corr_rows.append({"layer": l, "concept_a": a, "concept_b": b, "pearson_r": r})
    save_df(out_dir / "projection_correlations.csv", pd.DataFrame(corr_rows))

    logger.info("=" * 80)
    logger.info(f"Phase 1 complete. Outputs written to {out_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run(load_config())
