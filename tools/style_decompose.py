#!/usr/bin/env python
"""Is R_harm's style-dependence driven by the harmful side or the harmless side?

Each level is a (harmful_source | harmless_source) pair, so a pair-to-pair
comparison varies both. Splitting by what the two pairs SHARE separates the two
explanations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.config import RQ1_MODELS
import numpy as np, pandas as pd

for slug in RQ1_MODELS:
    p = Path(f"results/rq1/{slug}/style_vs_metadata.csv")
    if not p.exists():
        continue
    df = pd.read_csv(p)
    sp = df[(df.concept == "R_harm") & (df.factor == "source_pair")].copy()
    if not len(sp):
        continue
    def parts(s): return s.split("|")
    sp["h_a"], sp["s_a"] = zip(*sp.level_a.map(parts))
    sp["h_b"], sp["s_b"] = zip(*sp.level_b.map(parts))
    sp["shares"] = np.where(sp.h_a == sp.h_b, "same harmful source",
                    np.where(sp.s_a == sp.s_b, "same harmless source", "nothing shared"))
    floor = float(sp.split_half_floor.iloc[0])
    print(f"\n=== {slug} — R_harm across source pairs (split-half floor {floor:.3f}) ===")
    g = sp.groupby("shares").cosine.agg(["count", "min", "median", "max"]).round(3)
    print(g.to_string())
    print("\n  interpretation: whichever grouping keeps the cosine HIGH is the side")
    print("  that does NOT drive the variation.")
