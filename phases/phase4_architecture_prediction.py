"""Phase 4 — Structural characterization & vulnerability prediction (RQ5, the
central prediction experiment).

The architectural quantities {C_R, N_eff, r_eff, k_50} (phase 3) and probe
accuracy (phase 1) were estimated exclusively on the representation/causal-
analysis datasets (ROLE_PAIRS/HARM_PAIRS/CONTROL_PAIRS) and are treated here
as FROZEN. On a topically disjoint attack-intent pool (data.ATTACK_PROBE_PROMPTS),
we run two attack families per (layer, concept):

  - sparse neuron ablation, sweeping k -> k* = min k with ASR(k) >= tau
  - representation steering, sweeping alpha -> alpha* = min |alpha| with ASR(alpha) >= tau

ASR is measured behaviorally (generation + refusal-keyword classifier) on the
disjoint attack pool. We then correlate the frozen architecture metrics
against {k*, alpha*} across all (layer, concept) rows.

Scope note: with a single model, this is a *within-model* correlation across
many (layer, concept) architectural configurations, not true leave-one-model-out
generalization across models (that requires >= 2 models and is intentionally
out of scope for this single-model pipeline — see project plan RQ5). The
prediction table's row schema (layer, concept, metrics..., k*, alpha*) is
designed so leave-one-model-out becomes a matter of appending a `model_id`
column and more rows once a second model is run, not a redesign.

Requires phase1_geometry and phase3_components to have been run first.

Outputs (results/phase4_architecture_prediction/):
  prediction_table.csv          layer, concept, C_R, N_eff, r_eff, k_50, probe_accuracy, k_star, alpha_star
  prediction_correlations.csv   predictor, outcome, pearson_r, spearman_rho, n_obs
  k_star_curves.csv             layer, concept, k, ASR
  alpha_star_curves.csv         layer, concept, alpha, ASR
"""
import pandas as pd

from core import data
from core.config import Config, load_config
from core.hooks import HookEngine
from core.io_utils import get_logger, load_df, load_torch, require_phase_output, save_df
from core.metrics import find_min_alpha_for_asr, find_min_k_for_asr, simple_correlation
from core.model_io import build_prompt, load_model

PHASE_NAME = "phase4_architecture_prediction"
PHASE1 = "phase1_geometry"
PHASE3 = "phase3_components"
CONCEPTS = ["role", "harm", "control"]
PREDICTORS = ["functional_concentration_C_R", "effective_component_count_N_eff", "effective_functional_rank_r_eff", "k_50", "probe_accuracy"]
OUTCOMES = ["k_star", "alpha_star"]


def run(cfg: Config) -> None:
    out_dir = cfg.phase_dir(PHASE_NAME)
    logger = get_logger(PHASE_NAME, out_dir)
    logger.info("=" * 80)
    logger.info("PHASE 4: STRUCTURAL CHARACTERIZATION & VULNERABILITY PREDICTION (RQ5)")
    logger.info("=" * 80)

    phase1_dir = cfg.phase_dir(PHASE1)
    phase3_dir = cfg.phase_dir(PHASE3)
    require_phase_output(phase1_dir / "directions.pt", PHASE1)
    require_phase_output(phase3_dir / "architecture_metrics.csv", PHASE3)

    directions = load_torch(phase1_dir / "directions.pt")
    probe_acc_df = load_df(phase1_dir / "probe_accuracy.csv")
    arch_df = load_df(phase3_dir / "architecture_metrics.csv")
    neuron_rankings = load_torch(phase3_dir / "neuron_rankings.pt")

    probe_acc_mean = probe_acc_df.groupby(["layer", "concept"])["accuracy"].mean().rename("probe_accuracy")
    arch_df = arch_df.merge(probe_acc_mean, on=["layer", "concept"], how="left")

    model, tokenizer = load_model(cfg.model_id, cfg.device, logger)
    hooks = HookEngine(model)

    n_attack_prompts = 5 if cfg.fast_dev else len(data.ATTACK_PROBE_PROMPTS)
    attack_prompts = [build_prompt(tokenizer, user=p) for _, p in data.ATTACK_PROBE_PROMPTS[:n_attack_prompts]]
    logger.info(f"Disjoint attack-intent pool: {len(attack_prompts)} prompts (topics never used in phase 1-3)")

    k_grid = [1, 5, 10, 20, 50] if cfg.fast_dev else [1, 5, 10, 20, 50, 100, 250, 500]
    alpha_magnitudes = [1, 3, 6] if cfg.fast_dev else [0.5, 1, 2, 3, 4, 6, 8, 12]
    tau = cfg.attack_success_threshold

    prediction_rows = []
    k_curve_rows, alpha_curve_rows = [], []

    for _, row in arch_df.iterrows():
        layer, concept = int(row["layer"]), row["concept"]
        logger.info(f"[L{layer}/{concept}] running disjoint-attack sweeps (k*, alpha*)...")

        ranking = neuron_rankings[(layer, concept)]["causal"].tolist()
        mlp_module = hooks.layers[layer].mlp
        k_star, k_curve = find_min_k_for_asr(model, tokenizer, cfg.device, mlp_module, ranking, attack_prompts, k_grid, tau, cfg.max_new_tokens)
        k_curve["layer"], k_curve["concept"] = layer, concept
        k_curve_rows.append(k_curve)

        sign = data.ATTACK_STEERING_SIGN[concept]
        alpha_grid = [sign * m for m in alpha_magnitudes]
        direction = directions[concept][layer]
        alpha_star, alpha_curve = find_min_alpha_for_asr(model, tokenizer, cfg.device, hooks.layers[layer], direction, attack_prompts, alpha_grid, tau, cfg.max_new_tokens)
        alpha_curve["layer"], alpha_curve["concept"] = layer, concept
        alpha_curve_rows.append(alpha_curve)

        logger.info(f"[L{layer}/{concept}] k*={k_star}  alpha*={alpha_star}")
        prediction_rows.append({
            "layer": layer,
            "concept": concept,
            "functional_concentration_C_R": row["functional_concentration_C_R"],
            "effective_component_count_N_eff": row["effective_component_count_N_eff"],
            "effective_functional_rank_r_eff": row["effective_functional_rank_r_eff"],
            "k_50": row["k_50"],
            "probe_accuracy": row.get("probe_accuracy", float("nan")),
            "k_star": k_star if k_star is not None else float("nan"),
            "alpha_star": abs(alpha_star) if alpha_star is not None else float("nan"),
        })

    pred_df = pd.DataFrame(prediction_rows)
    save_df(out_dir / "prediction_table.csv", pred_df)
    save_df(out_dir / "k_star_curves.csv", pd.concat(k_curve_rows, ignore_index=True))
    save_df(out_dir / "alpha_star_curves.csv", pd.concat(alpha_curve_rows, ignore_index=True))
    logger.info(f"Prediction table:\n{pred_df.to_string()}")

    corr_rows = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            sub = pred_df[[predictor, outcome]].dropna()
            pearson_r, spearman_rho = simple_correlation(sub[predictor].tolist(), sub[outcome].tolist())
            corr_rows.append({"predictor": predictor, "outcome": outcome, "pearson_r": pearson_r, "spearman_rho": spearman_rho, "n_obs": len(sub)})
    corr_df = pd.DataFrame(corr_rows)
    save_df(out_dir / "prediction_correlations.csv", corr_df)
    logger.info(f"Architecture -> vulnerability correlations (within-model, across (layer,concept) rows):\n{corr_df.to_string()}")
    logger.info("NOTE: single-model scope -> this is a within-model correlation, not leave-one-model-out generalization (see module docstring).")

    logger.info("=" * 80)
    logger.info(f"Phase 4 complete. Outputs written to {out_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run(load_config())
