#!/usr/bin/env python
"""Adjudicate every gate matrix with identical logic, and compare like-for-like.

The three runs differ only in (model, control variant), so this is the matched
cross-model test the earlier comparison could not be: previously Qwen2.5 used the
under-refusal control and Qwen3.5 the over-refusal one, which are different
directions (cos 0.72 vs a split-half floor of 0.944).
"""
import json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd
from experiments.rq1 import _adjudicate


def _post(m):
    """Keep only the `t_post_inst` read position.

    `causal_matrix*.csv` carries a second read position (`t_inst`), added for the
    token-resolved readout. Every gate-level quantity is defined at
    `t_post_inst`; consuming both would double-count each intervention and mix
    two incomparable residual bases.

    It deliberately does NOT restrict read LAYERS. The matrix carries every layer
    downstream of the steer layer, and which of those form the test family is an
    adjudication option (`gate_layers`) applied inside `_adjudicate_at`, so that
    it can be swept. Filtering here would silently pin that sweep to one arm.
    Older matrices lack the column.
    """
    if "read_position" in m.columns:
        m = m[m.read_position == "t_post_inst"]
    return m



log = logging.getLogger("cmp"); log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout); h.setFormatter(logging.Formatter("%(message)s")); log.addHandler(h)

RUNS = [("qwen2.5-7b", "under"), ("qwen2.5-7b", "over"), ("qwen3.5-9b", "over")]
BOUND = "0.5"

summary = []
for slug, variant in RUNS:
    p = Path(f"results/rq1/{slug}/causal_matrix__{variant}.csv")
    if not p.exists():
        print(f"missing {p}"); continue
    print(f"\n{'='*76}\n{slug}  control={variant}\n{'='*76}")
    mat = _post(pd.read_csv(p))
    v = _adjudicate(mat, log, f"{slug}/{variant}")
    Path(f"results/rq1/{slug}/causal_gate__{variant}.json").write_text(
        json.dumps(v, indent=2, default=str))
    b = v["per_bound"][BOUND]
    dc = b["G2_asymmetry"].get("direction_counts", {})
    summary.append({"model": slug, "control": variant,
                    "G1": b["G1_diagonal_dominance"]["rate"],
                    "G2_qualifying": b["G2_asymmetry"]["n_asymmetric"],
                    "family": b["G2_asymmetry"]["family_size"],
                    "G3": b["G3_behavioural_dissociation"]["n"],
                    "verdict": v["verdict"]})
    if dc:
        tot = sum(dc.values())
        print(f"\n  direction of the stronger influence (KL<={BOUND}, FDR-surviving):")
        pairs = {}
        for k, n in dc.items():
            a, b_ = k.split("->")
            pairs.setdefault(frozenset((a, b_)), {})[k] = n
        for key, d in pairs.items():
            t = sum(d.values())
            parts = "  ".join(f"{k}: {n}/{t} ({n/t:.0%})" for k, n in sorted(d.items(), key=lambda x: -x[1]))
            top = max(d.values()) / t
            flag = "CONSISTENT" if top >= 0.8 else ("leaning" if top >= 0.65 else "MIXED")
            print(f"    {' <-> '.join(sorted(key)):<28} {parts:<46} [{flag}]")

print(f"\n{'='*76}\nMATCHED COMPARISON (KL_harmless <= {BOUND})\n{'='*76}")
print(pd.DataFrame(summary).to_string(index=False))
