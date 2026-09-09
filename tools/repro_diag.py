#!/usr/bin/env python
"""Quantify run-to-run agreement of RQ1 directions between two results roots."""
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

A_ROOT = Path(os.environ.get("ROOT_A", "./results"))
B_ROOT = Path(os.environ.get("ROOT_B", "./results_verify"))

for m in ["qwen2.5-7b", "qwen3.5-9b"]:
    pa, pb = A_ROOT/"rq1"/m/"direction_validation.csv", B_ROOT/"rq1"/m/"direction_validation.csv"
    if not (pa.exists() and pb.exists()):
        print(f"=== {m}: missing ==="); continue
    A, B = pd.read_csv(pa), pd.read_csv(pb)
    j = A.merge(B, on=["concept", "layer"], suffixes=("_a", "_b"))
    j["d"] = (j.auc_a - j.auc_b).abs()
    print(f"=== {m} — |delta AUC| over all layers ===")
    print(j.groupby("concept").d.agg(["max", "mean", "count"]).sort_values("max", ascending=False).round(5).to_string())
    rows = []
    for c in A.concept.unique():
        if c not in set(B.concept):
            rows.append({"concept": c, "note": "absent in B"}); continue
        a, b = A[A.concept == c], B[B.concept == c]
        ba, bb = a.loc[a.train_auc.idxmax()], b.loc[b.train_auc.idxmax()]
        rows.append({"concept": c, "layer_a": int(ba.layer), "layer_b": int(bb.layer),
                     "auc_a": round(ba.auc, 4), "auc_b": round(bb.auc, 4),
                     "delta": round(abs(ba.auc - bb.auc), 4)})
    print("\n  reported values (at each run's own train-selected layer):")
    print(pd.DataFrame(rows).to_string(index=False))
    print()
