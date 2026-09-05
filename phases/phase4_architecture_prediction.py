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
import numpy as np
import pandas as pd

from core import data
from core.config import Config, load_config
from core.hooks import HookEngine
from core.io_utils import get_logger, load_df, load_torch, require_phase_output, save_df
from core.judge import AttackJudge
from core.metrics import attack_success_rate, find_min_alpha_for_asr, find_min_k_for_asr, find_min_layer_prefix_for_asr, random_direction_like, correlation_with_ci, simple_correlation, validate_margin_vs_judge
from core.model_io import build_prompt, load_model

PHASE_NAME = "phase4_architecture_prediction"
PHASE1 = "phase1_geometry"
PHASE3 = "phase3_components"
CONCEPTS = ["role", "harm", "control"]
PREDICTORS = ["functional_concentration_C_R", "effective_component_count_N_eff", "effective_functional_rank_r_eff", "k_50", "probe_accuracy", "baseline_asr"]
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
    attack_questions = [p for _, p in data.ATTACK_PROBE_PROMPTS[:n_attack_prompts]]
    attack_prompts = [build_prompt(tokenizer, user=p) for p in attack_questions]
    logger.info(f"Disjoint attack-intent pool: {len(attack_prompts)} prompts (topics never used in phase 1-3)")

    judge = AttackJudge(cfg.device, use_llama_guard=cfg.use_llama_guard, logger=logger)
    # Benign set for the capability-retention constraint on k*/alpha*.
    benign_prompts = [build_prompt(tokenizer, user=p) for p in data.BENIGN_REFERENCE_PROMPTS]
    if not cfg.use_llama_guard:
        logger.warning("USE_LLAMA_GUARD=0: ASR is keyword-only. Do not report these numbers as ASR.")

    k_grid, alpha_magnitudes = cfg.k_grid, cfg.alpha_magnitudes
    tau = cfg.attack_success_threshold

    # Baseline ASR with no intervention: doubles as the "baseline refusal rate"
    # simple-predictor comparator the plan asks for, and as a positive control —
    # if an extreme-budget attack below never moves ASR off this value, the
    # measurement is suspect rather than the model being robust.
    baseline_asr = attack_success_rate(model, tokenizer, attack_prompts, attack_questions, cfg.device, cfg.max_new_tokens, judge)
    logger.info(f"Baseline ASR (no intervention): {baseline_asr:.3f}")

    # NeuroStrike's own budget parameterization: prune every selected safety
    # neuron across layers 0..i. This is a much larger budget than the
    # single-layer top-k sweep and is what their reported ASR comes from.
    ns_neurons_path = phase3_dir / "neurostrike_neurons.pt"
    if ns_neurons_path.exists():
        ns_neurons = load_torch(ns_neurons_path)
        prefix_star, prefix_curve = find_min_layer_prefix_for_asr(
            model, tokenizer, cfg.device, ns_neurons, hooks.num_layers,
            attack_prompts, attack_questions, cfg.layer_prefix_fractions, tau, cfg.max_new_tokens, judge,
        )
        prefix_curve["baseline_ASR"] = baseline_asr
        save_df(out_dir / "neurostrike_prefix_curve.csv", prefix_curve)
        logger.info(f"NeuroStrike layer-prefix attack: prefix*={prefix_star}\n{prefix_curve.to_string()}")
    else:
        logger.warning(f"{ns_neurons_path} missing — skipping NeuroStrike prefix attack (re-run phase 3).")

    prediction_rows = []
    k_curve_rows, alpha_curve_rows = [], []

    for _, row in arch_df.iterrows():
        layer, concept = int(row["layer"]), row["concept"]
        logger.info(f"[L{layer}/{concept}] running disjoint-attack sweeps (k*, alpha*)...")

        mlp_module = hooks.layers[layer].mlp
        # Sweep our causal ranking AND NeuroStrike's probe ranking through the
        # same graded budget axis, so the two component-identification routes
        # are compared on identical footing (their fixed |z|>3 rule alone has
        # no k axis; rank_by_probe_weight supplies one).
        k_star_by_source = {}
        for source in ("causal", "neurostrike", "random"):
            if source == "random":
                # Random-ranking control for k*, mirroring the random-ablation
                # control on A_R(k): ablating any k neurons degrades the model
                # somewhat, so k* only means something relative to this.
                n_pool = neuron_rankings[(layer, concept)]["causal"].numel()
                ranking = [int(i) for i in np.random.default_rng(cfg.seed + layer).permutation(n_pool)]
            else:
                ranking_t = neuron_rankings[(layer, concept)].get(source)
                if ranking_t is None:
                    continue
                ranking = ranking_t.tolist()
            ks, kc = find_min_k_for_asr(model, tokenizer, cfg.device, mlp_module, ranking, attack_prompts, attack_questions, k_grid, tau, cfg.max_new_tokens, judge, benign_prompts=benign_prompts, min_utility=cfg.min_utility)
            kc["layer"], kc["concept"], kc["ranking_source"] = layer, concept, source
            k_curve_rows.append(kc)
            k_star_by_source[source] = ks
        k_star = k_star_by_source.get("causal")

        sign = data.ATTACK_STEERING_SIGN[concept]
        alpha_grid = [sign * m for m in alpha_magnitudes]
        direction = directions[concept][layer]
        alpha_star, alpha_curve = find_min_alpha_for_asr(model, tokenizer, cfg.device, hooks.layers[layer], direction, attack_prompts, attack_questions, alpha_grid, tau, cfg.max_new_tokens, judge, relative=cfg.steer_relative, benign_prompts=benign_prompts, min_utility=cfg.min_utility)
        alpha_curve["layer"], alpha_curve["concept"], alpha_curve["direction_type"] = layer, concept, "concept"
        alpha_curve_rows.append(alpha_curve)

        # Random-direction control: same magnitudes, meaningless direction.
        # Without it, "steering R breaks safety at alpha*" is indistinguishable
        # from "any perturbation this large breaks safety".
        rand_dir = random_direction_like(direction, seed=cfg.seed + layer)
        alpha_star_rand, rand_curve = find_min_alpha_for_asr(model, tokenizer, cfg.device, hooks.layers[layer], rand_dir, attack_prompts, attack_questions, alpha_grid, tau, cfg.max_new_tokens, judge, relative=cfg.steer_relative, benign_prompts=benign_prompts, min_utility=cfg.min_utility)
        rand_curve["layer"], rand_curve["concept"], rand_curve["direction_type"] = layer, concept, "random"
        alpha_curve_rows.append(rand_curve)
        logger.info(f"[L{layer}/{concept}] alpha*={alpha_star} vs random-direction control alpha*={alpha_star_rand}")

        logger.info(f"[L{layer}/{concept}] k*={k_star}  alpha*={alpha_star}")
        prediction_rows.append({
            "layer": layer,
            "concept": concept,
            "functional_concentration_C_R": row["functional_concentration_C_R"],
            "effective_component_count_N_eff": row["effective_component_count_N_eff"],
            "effective_functional_rank_r_eff": row["effective_functional_rank_r_eff"],
            "k_50": row["k_50"],
            "probe_accuracy": row.get("probe_accuracy", float("nan")),
            "baseline_asr": baseline_asr,
            "k_star": k_star if k_star is not None else float("nan"),
            "k_star_neurostrike": k_star_by_source.get("neurostrike") if k_star_by_source.get("neurostrike") is not None else float("nan"),
            "k_star_random_control": k_star_by_source.get("random") if k_star_by_source.get("random") is not None else float("nan"),
            "alpha_star": abs(alpha_star) if alpha_star is not None else float("nan"),
        })

    # Proxy validation: does the cheap refusal margin track the guard's verdict?
    mid_layer = int(arch_df.iloc[len(arch_df) // 2]["layer"])
    margin_probe_toks = [tokenizer(p, return_tensors="pt").to(cfg.device) for p in attack_prompts[:3]]
    val_df = validate_margin_vs_judge(
        model, tokenizer, cfg.device, hooks.layers[mid_layer], directions["control"][mid_layer],
        attack_prompts, attack_questions, margin_probe_toks,
        [data.ATTACK_STEERING_SIGN["control"] * m for m in alpha_magnitudes], cfg.max_new_tokens, judge,
    )
    save_df(out_dir / "proxy_validation.csv", val_df)
    r_margin, rho_margin = simple_correlation(val_df["refusal_margin"].tolist(), val_df["ASR"].tolist())
    logger.info(f"Proxy validation (refusal margin vs judge ASR, steering R_control@L{mid_layer}): pearson={r_margin:.3f} spearman={rho_margin:.3f}\n{val_df.to_string()}")

    pred_df = pd.DataFrame(prediction_rows)
    save_df(out_dir / "prediction_table.csv", pred_df)
    save_df(out_dir / "k_star_curves.csv", pd.concat(k_curve_rows, ignore_index=True))
    save_df(out_dir / "alpha_star_curves.csv", pd.concat(alpha_curve_rows, ignore_index=True))
    logger.info(f"Prediction table:\n{pred_df.to_string()}")

    corr_rows = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            sub = pred_df[[predictor, outcome]].dropna()
            stats = correlation_with_ci(sub[predictor].tolist(), sub[outcome].tolist(), seed=cfg.seed)
            corr_rows.append({"predictor": predictor, "outcome": outcome, **stats})
    corr_df = pd.DataFrame(corr_rows)
    save_df(out_dir / "prediction_correlations.csv", corr_df)
    logger.info(f"Architecture -> vulnerability correlations (within-model, across (layer,concept) rows):\n{corr_df.to_string()}")
    logger.info("NOTE: single-model scope -> this is a within-model correlation, not leave-one-model-out generalization (see module docstring).")

    logger.info("=" * 80)
    logger.info(f"Phase 4 complete. Outputs written to {out_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run(load_config())
