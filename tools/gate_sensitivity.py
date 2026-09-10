#!/usr/bin/env python
"""O-12 — is Gate 1's verdict robust to the adjudication's own discretionary choices?

The adjudicator was corrected four times during RQ1 and every correction moved or
reversed the verdict. That history is the reason for this: a conclusion that holds
only at the settings we happened to land on is not a conclusion. Here every
remaining discretionary choice is swept and the verdict recomputed.

Adjudication is analysis, not measurement — the saved `causal_matrix*.csv` files
are the measurement — so the whole sweep is CPU-only and costs no GPU time.

Swept:
    null_q       percentile defining the random-direction null band
    null_group   how that null is matched: by alpha, by alpha AND layer, or pooled
    g2_rule      one direction beyond the null, or both
    fdr_q        Benjamini-Hochberg level
    beh_null_q   percentile for the behavioural null band

Reported three ways, because they answer different questions:
    1. the full grid — how often each qualitative verdict appears
    2. one-at-a-time — which single choice moves the result most
    3. the headline claim — does `R_harm -> R_control` stay directionally
       consistent across the whole grid?
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from experiments.rq1 import ADJUDICATION_DEFAULTS, _adjudicate_at  # noqa: E402


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



GRID = {
    "null_q": [90.0, 95.0, 97.5, 99.0],
    "null_group": ["alpha", "alpha_layer", "pooled"],
    "g2_rule": ["any", "both"],
    "fdr_q": [0.01, 0.05, 0.10],
    "beh_null_q": [90.0, 95.0, 99.0],
    # Which read layers form the BH family. E1.6 reads every layer downstream of
    # the steer layer; testing all of them versus three relative depths is a free
    # choice, so it is swept rather than argued.
    "gate_layers": ["all", "relative3"],
}
BOUND = 0.5          # the mid capability bound; the ladder is swept separately
DEFAULTS = dict(ADJUDICATION_DEFAULTS)


def _matrices():
    for mp in sorted(Path("results/rq1").glob("*/causal_matrix*.csv")):
        tag = mp.stem.replace("causal_matrix", "").lstrip("_") or "default"
        yield f"{mp.parent.name}/{tag}", _post(pd.read_csv(mp))


def _summarise(v: dict) -> dict:
    g2 = v["G2_asymmetry"]
    g3 = v["G3_behavioural_dissociation"]
    return {"n_live": v["n_live"], "family": g2["family_size"],
            "g2": g2["n_asymmetric"], "g3": g3["n"], "verdict": v["verdict"],
            "g2_holds": g2["n_asymmetric"] > 0, "g3_holds": g3["n"] > 0,
            "direction_counts": g2.get("direction_counts", {})}


def _harm_to_control(dc: dict) -> tuple:
    """(consistent?, share) for R_harm <-> R_control among qualifying asymmetries."""
    fwd = dc.get("R_harm->R_control", 0)
    rev = dc.get("R_control->R_harm", 0)
    tot = fwd + rev
    return (tot > 0 and fwd / tot >= 0.8, (fwd / tot) if tot else float("nan"), tot)


def main() -> int:
    keys = list(GRID)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(GRID[k] for k in keys))]
    print(f"sweeping {len(combos)} adjudication settings x {BOUND} capability bound\n")

    rows = []
    for name, mat in _matrices():
        for opts in combos:
            s = _summarise(_adjudicate_at(mat, BOUND, opts))
            cons, share, tot = _harm_to_control(s.pop("direction_counts"))
            rows.append({"run": name, **opts, **s,
                         "hc_consistent": cons, "hc_share": share, "hc_n": tot})
    df = pd.DataFrame(rows)
    Path("results/gate_sensitivity.csv").write_text(df.to_csv(index=False))

    print("=" * 78)
    print("1. FULL GRID — how often each qualitative outcome appears")
    print("=" * 78)
    for run, g in df.groupby("run"):
        n = len(g)
        print(f"\n{run}  ({n} settings)")
        print(f"   G2 holds (asymmetry present) : {g.g2_holds.mean():6.1%}   "
              f"G2 count median {int(g.g2.median()):>3}  range {int(g.g2.min())}-{int(g.g2.max())}")
        print(f"   G3 holds (dissociation)      : {g.g3_holds.mean():6.1%}   "
              f"G3 count median {int(g.g3.median()):>3}  range {int(g.g3.min())}-{int(g.g3.max())}")
        vc = g.verdict.value_counts()
        print(f"   verdicts: {', '.join(f'{k} {v}/{n}' for k, v in vc.items())}")
        hc = g.dropna(subset=["hc_share"])
        if len(hc):
            print(f"   R_harm->R_control consistent (>=80%): {hc.hc_consistent.mean():6.1%}"
                  f"   median share {hc.hc_share.median():.2f}")

    print("\n" + "=" * 78)
    print("2. ONE-AT-A-TIME — vary one choice, hold the rest at the pre-registered value")
    print("=" * 78)
    for run, g in df.groupby("run"):
        print(f"\n{run}")
        for k in keys:
            mask = np.ones(len(g), dtype=bool)
            for other in keys:
                if other != k:
                    mask &= (g[other] == DEFAULTS[other]).values
            sub = g[mask].sort_values(k)
            if not len(sub):
                continue
            cells = "  ".join(f"{v}:G2={int(r.g2)},G3={int(r.g3)}"
                              for v, r in zip(sub[k], sub.itertuples()))
            print(f"   {k:<12} {cells}")

    print("\n" + "=" * 78)
    print("3. HEADLINE ROBUSTNESS")
    print("=" * 78)
    print(f"   G2 > 0 in {df.g2_holds.mean():.1%} of all {len(df)} run x setting combinations")
    print(f"   G3 > 0 in {df.g3_holds.mean():.1%}")
    hc = df.dropna(subset=["hc_share"])
    print(f"   R_harm->R_control directionally consistent in {hc.hc_consistent.mean():.1%} "
          f"of {len(hc)} combinations where the pair was testable")
    worst = df.loc[df.g2.idxmin()]
    print(f"\n   weakest setting for G2: {worst.run} "
          f"({', '.join(f'{k}={worst[k]}' for k in keys)}) -> G2={int(worst.g2)} G3={int(worst.g3)}")
    Path("results/gate_sensitivity_summary.json").write_text(json.dumps({
        "n_settings": len(combos), "bound": BOUND,
        "g2_holds_share": float(df.g2_holds.mean()),
        "g3_holds_share": float(df.g3_holds.mean()),
        "harm_to_control_consistent_share": float(hc.hc_consistent.mean()) if len(hc) else None,
        "grid": GRID, "defaults": DEFAULTS,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
