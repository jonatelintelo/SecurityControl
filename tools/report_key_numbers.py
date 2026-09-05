#!/usr/bin/env python
"""Print every quantity the README cites, straight from results/.

Run this after any full pipeline run and reconcile the output against the
README's Glossary and Status sections. Numbers have already drifted once
(a pre-redesign ablation figure was quoted as current for several rounds),
and that class of error is not caught by tests — only by re-deriving the
cited values from the artifacts.

    python tools/report_key_numbers.py [results_dir]
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results")


def _read(rel):
    p = ROOT / rel
    return pd.read_csv(p) if p.exists() else None


def _section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main() -> None:
    _section("PROVENANCE")
    for man in sorted(ROOT.glob("*/run_manifest.json")):
        m = json.load(open(man))
        commit = (m.get("git_commit") or "none")[:8]
        print(f"  {m['phase']:<32} commit={commit} dirty={m.get('git_dirty')} job={m.get('slurm_job_id')} {m['timestamp_utc'][:19]}")
    cfgs = list(ROOT.glob("*/run_manifest.json"))
    if cfgs:
        commits = {json.load(open(c)).get("git_commit") for c in cfgs}
        print(f"  -> {'CONSISTENT' if len(commits) == 1 else 'MIXED CODE STATES — do not quote jointly'}: {len(commits)} distinct commit(s)")

    _section("STEP 1 — do the variables exist? (RQ1)")
    if (df := _read("phase1_geometry/probe_accuracy.csv")) is not None:
        print("  probe_accuracy (held-out):", df.groupby("concept")["accuracy"].mean().round(3).to_dict())
    if (df := _read("phase1_geometry/concept_dimensionality.csv")) is not None:
        agg = df.groupby("concept")[["r_eff_spectrum", "top1_variance_share"]].mean().round(3)
        print("  r_eff_spectrum:          ", agg["r_eff_spectrum"].to_dict())
        print("  top1_variance_share:     ", agg["top1_variance_share"].to_dict())
        print("  n_pairs (caps r_eff):    ", df.groupby("concept")["n_pairs"].max().to_dict())
    if (df := _read("phase1_geometry/geometry_cosine.csv")) is not None:
        print("  cross-concept cosine:    ", df.groupby(["concept_a", "concept_b"])["cosine_sim"].mean().round(3).to_dict())
    if (df := _read("phase1_geometry/control_transfer_test.csv")) is not None:
        s = df["transfer_accuracy_unforced"]
        print(f"  R_control transfer:       mean={s.mean():.3f} best={s.max():.3f} (0.5=chance)")

    _section("STEP 2 — causal structure (RQ2)")
    if (df := _read("phase2_causal_structure/cross_intervention_matrix.csv")) is not None:
        alpha = df["alpha"].iloc[0] if "alpha" in df else "?"
        rel = df["steer_relative"].iloc[0] if "steer_relative" in df else "?"
        print(f"  steer_alpha={alpha} relative={rel}  (>1.0 relative == destructive, treat as invalid)")
        print(df.groupby("source_concept")[[c for c in df.columns if c.startswith("cos_delta") or c == "behavior_margin_delta"]].mean().round(4).to_string())

    _section("STEP 4 — structural quantities (RQ5 predictors)")
    if (df := _read("phase3_components/architecture_metrics.csv")) is not None:
        cols = [c for c in ["layer", "concept", "functional_concentration_C_R", "effective_component_count_N_eff",
                            "effective_functional_rank_r_eff", "k_50", "k_50_random_control", "k_50_behavioral"] if c in df]
        print(df[cols].to_string(index=False))
    if (df := _read("phase3_components/ablation_curves.csv")) is not None:
        piv = df.groupby(["ablation", "k"])["ratio_A_R"].mean().unstack(0).round(3)
        print("\n  A_R(k) ranked vs random (the key control):")
        print(piv.to_string())
        rk = df[(df.ablation == "ranked") & (df.k == df.k.min())]
        print("  per-concept at k=%d (ranked): %s" % (df.k.min(), rk.groupby("concept")["ratio_A_R"].mean().round(3).to_dict()))

    _section("STEP 5 — attack budgets (RQ5 outcome)")
    if (df := _read("phase4_architecture_prediction/prediction_table.csv")) is not None:
        cols = [c for c in ["layer", "concept", "k_50", "baseline_asr", "k_star", "k_star_random_control", "k_star_neurostrike", "alpha_star"] if c in df]
        print(df[cols].to_string(index=False))
    if (df := _read("phase4_architecture_prediction/alpha_star_curves.csv")) is not None:
        print("\n  ASR / utility by |alpha| and direction type:")
        df["abs_alpha"] = df["alpha"].abs()
        cols = [c for c in ["ASR", "utility", "viable_attack"] if c in df]
        print(df.groupby(["direction_type", "abs_alpha"])[cols].mean().round(3).to_string())
    if (df := _read("phase4_architecture_prediction/neurostrike_prefix_curve.csv")) is not None:
        print("\n  NeuroStrike layer-prefix attack (positive control):")
        print(df.to_string(index=False))
    if (df := _read("phase4_architecture_prediction/proxy_validation.csv")) is not None:
        print("\n  proxy validation (margin vs judge):")
        print(df.to_string(index=False))

    _section("STEP 3 / 6 — components, NeuroStrike, thesis test")
    if (df := _read("phase3_components/neuron_ranking_overlap.csv")) is not None:
        print("  ranking overlap (Jaccard):", df.groupby("comparison")["jaccard"].mean().round(4).to_dict())
    if (df := _read("phase3_components/neurostrike_neuron_functions.csv")) is not None:
        counts = df.groupby("best_concept").size().to_dict()
        print(f"  NeuroStrike neurons by function: {counts}  (near-uniform => probe is noise)")
    if (df := _read("phase4_architecture_prediction/prediction_correlations.csv")) is not None:
        print("\n  THE THESIS TEST (architecture -> attack budget):")
        print(df.to_string(index=False))

    _section("STEP 7 / 8 — context stability & attack taxonomy")
    for rel, label in [("phase5_context_and_attacks/context_stability.csv", "S_repr"),
                       ("phase5_context_and_attacks/component_stability.csv", "S_comp"),
                       ("phase5_context_and_attacks/attack_stage_diagnosis.csv", "attack stage")]:
        if (df := _read(rel)) is not None:
            print(f"\n  {label}:")
            print(df.to_string(index=False))


if __name__ == "__main__":
    main()
