#!/usr/bin/env python
"""Which directed influences are asymmetric, and which way do they point?

G2 establishes that asymmetry exists. This asks whether it forms a consistent
directed structure — the difference between 'the variables are distinguishable'
and 'here is the architecture'.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd


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



BOUND = 0.5

# Gate artifacts carry a `__under` / `__over` suffix naming the control variant.
# Globbing, not naming one file: an earlier version read the pre-rename
# `causal_matrix.csv` and `continue`d when it was absent, so it printed nothing
# at all while still exiting 0 — a silent skip that reads exactly like a real
# "no qualifying pairs" result.
mats = sorted(Path("results/rq1").glob("*/causal_matrix*.csv"))
if not mats:
    raise SystemExit("no causal_matrix*.csv under results/rq1 — nothing to analyse")

for p in mats:
    slug = p.parent.name
    variant = p.stem.replace("causal_matrix", "") or "(default)"
    m = _post(pd.read_csv(p))
    rnd = m[m.source == "random"]
    band = {k: float(np.percentile(g.delta.abs().dropna(), 95))
            for k, g in rnd.groupby(["target", "abs_alpha"])}
    live = m[(m.source != "random") & (m.kl_harmless <= BOUND) & (m.alpha > 0)]

    print(f"\n{'='*74}\n{slug} {variant}  (KL_harmless <= {BOUND}, alpha > 0)\n{'='*74}")

    rows = []
    for (L, rl, a), g in live.groupby(["steer_layer", "read_layer", "abs_alpha"]):
        srcs = sorted(g.source.unique())
        for i, A in enumerate(srcs):
            for B in srcs[i+1:]:
                ab = g[(g.source == A) & (g.target == B)]
                ba = g[(g.source == B) & (g.target == A)]
                if not len(ab) or not len(ba):
                    continue
                r1, r2 = ab.iloc[0], ba.iloc[0]
                disj = (r1.ci_low > r2.ci_high) or (r2.ci_low > r1.ci_high)
                big = abs(r1.delta) > band.get((B, a), np.inf) or abs(r2.delta) > band.get((A, a), np.inf)
                if disj and big:
                    stronger = f"{A}->{B}" if abs(r1.delta) > abs(r2.delta) else f"{B}->{A}"
                    rows.append({"pair": f"{A} <-> {B}", "stronger": stronger,
                                 "d_ab": r1.delta, "d_ba": r2.delta, "alpha": a,
                                 "steer_layer": L, "read_layer": rl})
    if not rows:
        print("  no qualifying asymmetric pairs")
        continue
    df = pd.DataFrame(rows)
    print(f"  {len(df)} qualifying asymmetric cells\n")
    print("  direction of the STRONGER influence, per unordered pair:")
    for pair, g in df.groupby("pair"):
        vc = g.stronger.value_counts()
        tot = int(vc.sum())
        parts = "  ".join(f"{k}: {v}/{tot} ({v/tot:.0%})" for k, v in vc.items())
        winner = vc.index[0]
        consistency = vc.iloc[0] / tot
        flag = "CONSISTENT" if consistency >= 0.8 else ("leaning" if consistency >= 0.65 else "MIXED")
        print(f"    {pair:<28} {parts:<44} [{flag}]")
    print("\n  mean |delta| by ordered direction (qualifying cells only):")
    md = {}
    for _, r in df.iterrows():
        A, B = r.pair.split(" <-> ")
        md.setdefault(f"{A}->{B}", []).append(abs(r.d_ab))
        md.setdefault(f"{B}->{A}", []).append(abs(r.d_ba))
    for k in sorted(md):
        print(f"    {k:<28} {np.mean(md[k]):.3f}  (n={len(md[k])})")
