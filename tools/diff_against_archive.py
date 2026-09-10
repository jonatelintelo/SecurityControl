#!/usr/bin/env python
"""Compare a fresh results root against an archived one, quantity by quantity.

The point of a blank-slate rerun is not that it produces numbers — it is that the
numbers can be *compared* to what was reported before. Three outcomes, and all
three are informative:

  MATCH      the earlier number survives; it was not an artifact of stale code.
  DRIFT      same quantity, different value — either a real code fix landed, or
             something is nondeterministic that should not be.
  NEW / GONE the artifact set changed; usually a rename, occasionally a stage
             that silently stopped producing output.

Only the *evidence* files are compared (CSV/JSON). `activations.pt` is a cache and
is excluded from the archive by construction.

    python tools/diff_against_archive.py [archive_root] [--live results]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pandas as pd

ARCHIVE = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
               else "results_archive/pre_blank_slate_20260909/results")
LIVE = Path("results")

# Quantities worth naming individually, because a claim in RQ1_FINDINGS.md rests
# on each. (file, key columns, value columns)
TRACKED = [
    ("direction_validation.csv", ["concept", "layer"],
     ["auc", "train_auc", "length_only_auc", "split_half_cos"]),
    ("harm_controls.csv", ["concept", "position"],
     ["auc", "layer", "split_half_cos"]),
    ("harm_controls_geometry.csv", ["variant", "position"],
     ["cos_with_pooled_harm", "cos_R_control_refctrl", "cos_R_control_harmless_refctrl",
      "cos_R_control_pooled", "cos_R_control_harmless_pooled"]),
    ("geometry_cosines.csv", ["concept_a", "concept_b", "layer"], ["cosine"]),
    ("emergence_summary.csv", ["concept"], ["onset_90"]),
    ("dimensionality.csv", ["concept"], ["r_eff"]),
]

TOL = 5e-4          # below this two floats are "the same number"


def _fmt(x) -> str:
    return f"{x:.4f}" if isinstance(x, (float, np.floating)) else str(x)


def compare_csv(rel: str, keys: list[str], vals: list[str], model: str) -> None:
    a, b = ARCHIVE / "rq1" / model / rel, LIVE / "rq1" / model / rel
    if not b.exists():
        print(f"  {rel:<32} GONE     (not produced by the new run)")
        return
    if not a.exists():
        print(f"  {rel:<32} NEW      (no archived counterpart)")
        return
    A, B = pd.read_csv(a), pd.read_csv(b)
    keys = [k for k in keys if k in A.columns and k in B.columns]
    vals = [v for v in vals if v in A.columns and v in B.columns]
    if not keys or not vals:
        print(f"  {rel:<32} SCHEMA   keys/values changed; columns "
              f"added={sorted(set(B.columns) - set(A.columns))} "
              f"removed={sorted(set(A.columns) - set(B.columns))}")
        return
    j = A.merge(B, on=keys, suffixes=("_a", "_b"))
    worst, n_drift = ("", 0.0), 0
    for v in vals:
        d = (pd.to_numeric(j[f"{v}_a"], errors="coerce")
             - pd.to_numeric(j[f"{v}_b"], errors="coerce")).abs()
        n_drift += int((d > TOL).sum())
        if len(d) and d.max() > worst[1]:
            worst = (v, float(d.max()))
    verdict = "MATCH" if n_drift == 0 else "DRIFT"
    print(f"  {rel:<32} {verdict:<8} rows={len(j)}/{len(A)}  "
          f"cells>tol={n_drift}  worst: {worst[0]} Δ={worst[1]:.4f}")

    # Name the biggest movers, so a DRIFT is actionable rather than just a flag.
    if n_drift:
        rowdiff = []
        for _, r in j.iterrows():
            for v in vals:
                x, y = pd.to_numeric(r[f"{v}_a"], errors="coerce"), pd.to_numeric(r[f"{v}_b"], errors="coerce")
                if pd.notna(x) and pd.notna(y) and abs(x - y) > TOL:
                    rowdiff.append((abs(x - y), " ".join(_fmt(r[k]) for k in keys), v, x, y))
        for d, k, v, x, y in sorted(rowdiff, reverse=True)[:6]:
            print(f"      {k:<28} {v:<22} {x:.4f} -> {y:.4f}   (Δ {d:+.4f})")


def compare_gate(model: str) -> None:
    for gp in sorted((LIVE / "rq1" / model).glob("causal_gate*.json")):
        ap = ARCHIVE / "rq1" / model / gp.name
        if not ap.exists():
            print(f"  {gp.name:<32} NEW")
            continue
        A, B = json.loads(ap.read_text()), json.loads(gp.read_text())
        for bound in sorted(set(A.get("per_bound", {})) | set(B.get("per_bound", {}))):
            va, vb = A["per_bound"].get(bound, {}), B["per_bound"].get(bound, {})
            sa, sb = va.get("verdict", "-"), vb.get("verdict", "-")
            g2a = va.get("G2_asymmetry", {}).get("n_fdr_reject", "-")
            g2b = vb.get("G2_asymmetry", {}).get("n_fdr_reject", "-")
            flag = "MATCH" if (sa == sb and g2a == g2b) else "DRIFT"
            print(f"  {gp.name}  KL<={bound:<5} {flag:<6} verdict {sa} -> {sb}   "
                  f"G2 FDR-reject {g2a} -> {g2b}")


def main() -> int:
    if not ARCHIVE.exists():
        print(f"archive not found: {ARCHIVE}")
        return 1
    print(f"archive : {ARCHIVE}\nlive    : {LIVE}\ntolerance: {TOL}\n")
    for model in ("qwen2.5-7b", "qwen3.5-9b", "qwen3.5-35b-a3b"):
        if not (LIVE / "rq1" / model).exists():
            print(f"\n{'='*78}\n{model}: NOT PRESENT in the live run\n{'='*78}")
            continue
        print(f"\n{'='*78}\n{model}\n{'='*78}")
        for rel, keys, vals in TRACKED:
            compare_csv(rel, keys, vals, model)
        compare_gate(model)
    print("\nDRIFT is not automatically bad: several corrections landed after the "
          "archived run.\nEach one should be explainable by a specific change.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
