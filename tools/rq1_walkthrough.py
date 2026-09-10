#!/usr/bin/env python
"""End-to-end audit: every RQ1 experiment, its artifacts, and its verification."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

R = Path("results"); C = R/"e1_0_corpus"
MODELS = ["qwen2.5-7b", "qwen3.5-9b", "qwen3.5-35b-a3b"]

def art(m, f): return (R/"rq1"/m/f).exists()
def ok(b): return "ok " if b else "MISS"

STAGES = [
    ("E1.0",  "corpus + transfer corpus, frozen",
     lambda: (C/"instructions.jsonl").exists() and (C/"transfer_corpus.jsonl").exists()
             and "FROZEN" in json.load(open(C/"corpus_meta.json"))["status"],
     "verification.json (22 checks)"),
    ("E1.1a", "refusal labelling, both models",
     lambda: all(art(m,"refusal_labels.csv") and art(m,"budget_ladder.json") for m in MODELS),
     "labels_checks.json: cell size + undetermined rate"),
    ("E1.1b", "extraction: directions + role probe, all layers",
     lambda: all(art(m,"directions.pt") and art(m,"role_probes.pt") and art(m,"direction_validation.csv") for m in MODELS),
     "held-out AUC + cluster-bootstrap CI + length baseline + split-half"),
    ("E1.1c", "fidelity: pre_mlp + cross-corpus transfer",
     lambda: all(art(m,"fidelity.json") for m in MODELS),
     "transfer above chance in BOTH directions"),
    ("E1.1d", "R_harm with refusal held constant",
     lambda: all(art(m,"harm_controls.csv") and art(m,"harm_controls_geometry.csv") for m in MODELS),
     "in_refused / in_complied vs pooled; layer-MATCHED cosines only"),
    ("E1.2a", "geometry pass 1 — pairwise cosines",
     lambda: all(art(m,"geometry_cosines.csv") and art(m,"geometry_null_band.json") for m in MODELS),
     "null band vs analytic 1/sqrt(d); split-half floor"),
    ("E1.2b", "geometry pass 2 — subspaces at E1.4b's k",
     lambda: all(art(m,"geometry_subspace.csv") for m in MODELS),
     "principal angles / projection metric / data CCA vs matched subspace null"),
    ("E1.3",  "projections + correlations",
     lambda: all(art(m,"projections.csv") and art(m,"projection_correlations.csv") for m in MODELS),
     "correlation null from random directions"),
    ("E1.4a", "dimensionality, spectral half",
     lambda: all(art(m,"dimensionality.csv") for m in MODELS),
     "r_eff over stratified dirs; AUC after removing top-1"),
    ("E1.4b", "dimensionality, behavioural k",
     lambda: all(art(m,"behavioural_k.json") for m in MODELS),
     "k* vs behavioural null (k UNDEFINED below it)"),
    ("E1.5",  "emergence + onset CIs",
     lambda: all(art(m,"emergence_summary.csv") for m in MODELS),
     "onset90 bootstrap CI; conclusions across 80/90/95%"),
    ("E1.6",  "causal gate (both control variants)",
     lambda: art("qwen2.5-7b","causal_gate__under.json") and art("qwen2.5-7b","causal_gate__over.json")
             and art("qwen3.5-9b","causal_gate__over.json"),
     "G1/G2/G3 vs alpha-matched nulls, FDR, KL ladder"),
    ("E1.7b", "Level 2: tag vs register, crossed on generated corpus",
     lambda: all(art(m,"style_level2.csv") for m in MODELS),
     "cos(tag, register) vs split-half floor; generated arm + template control"),
    ("E1.6b", "stage B: alpha x steered-token-position profile",
     lambda: all(any((R/"rq1"/m).glob("causal_stage_b*.csv")) for m in MODELS),
     "4 token sets x full alpha grid, on TRAIN; picks Stage C's layers"),
    ("E1.6s", "gate sensitivity sweep (O-12)",
     lambda: (R/"gate_sensitivity.csv").exists(),
     "216 settings x 3 runs"),
    ("E1.7",  "metadata vs style, Level 1",
     lambda: all(art(m,"style_vs_metadata.csv") for m in MODELS),
     "within-level split-half floor"),
]
def missing_models(f):
    """Which models lack this artifact — a bare MISS does not say."""
    return [m for m in MODELS if not art(m, f)]


print(f"{'stage':<7} {'':<4} {'what':<46} verification")
print("-"*118)
for tag, what, chk, ver in STAGES:
    try:
        good = chk()
    except Exception as e:                      # a malformed artifact is not "present"
        good = False
        ver = f"{ver}   [check raised {type(e).__name__}]"
    print(f"{tag:<7} [{ok(good)}] {what:<46} {ver}")

print("\n" + "="*118)
print("HEADLINE NUMBERS")
print("="*118)
for m in MODELS:
    # Degrade per model rather than dying. This runs unattended after a multi-hour
    # chain, and a crash on the first incomplete model would discard the report
    # for every model that DID finish.
    need = ["direction_validation.csv", "emergence_summary.csv", "behavioural_k.json"]
    absent = [f for f in need if not art(m, f)]
    if absent:
        print(f"\n{m}\n   INCOMPLETE — missing {absent}")
        continue
    v = pd.read_csv(R/"rq1"/m/"direction_validation.csv")
    e = pd.read_csv(R/"rq1"/m/"emergence_summary.csv")
    k = json.load(open(R/"rq1"/m/"behavioural_k.json"))["k_star"]
    print(f"\n{m}")
    for c in ["R_harm","R_harm_at_post","R_control","R_control_harmless"]:
        s = v[v.concept==c]
        if not len(s): continue
        b = s.loc[s.train_auc.idxmax()]
        er = e[e.concept==c]
        on = f"{er.onset_90.iloc[0]:.2f}[{er.onset_90_ci_low.iloc[0]:.2f},{er.onset_90_ci_high.iloc[0]:.2f}]" if len(er) else "-"
        print(f"   {c:<20} AUC {b.auc:.3f}  len-only {b.length_only_auc:.3f}  "
              f"split-half {b.split_half_cos:.3f}  onset90 {on:<20} k*={k.get(c)}")
