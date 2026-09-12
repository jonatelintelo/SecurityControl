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
from core.config import RQ1_MODELS
import numpy as np
import pandas as pd

# There is currently NO default baseline. `results_archive/` was deleted on
# 2026-09-12 once it had become inert: it was built on a 200/200 corpus against
# the live 500/500, so the corpus-scale guard below refused it anyway and every
# quantity would have differed for that reason alone.
#
# The tool is kept because it is generic — pass any two roots. The useful
# comparison today is `./results` against `./results_verify`, and the next real
# baseline is whatever RQ2 freezes. A missing archive is a clean SKIP (exit 0),
# not a failure: this runs inside `blank_slate_post.sh`, where a non-zero exit
# would abort the rest of the post pass over a comparison nobody can make.
#
# Whatever is passed should be the IMMEDIATELY PRECEDING state: drift against a
# state two code-generations back mixes several explanations and makes "every
# DRIFT must be explainable by a specific change" unenforceable.
ARCHIVE = Path(sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
               else "results_archive/pre_5model_20260911/results")
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
    # A drift comparison is only meaningful between runs of the SAME corpus.
    # Every archive predates the 500/500 scale-up, so every quantity would read
    # DRIFT and the one signal this tool exists for — "an unexplained change is a
    # bug" — would be buried under a thousand explained ones. Detected and said
    # out loud rather than left for the reader to infer from a wall of red.
    try:
        import json as _j
        a_meta = _j.loads((ARCHIVE / "e1_0_corpus" / "corpus_meta.json").read_text())
        l_meta = _j.loads((LIVE / "e1_0_corpus" / "corpus_meta.json").read_text())
        a_n = (a_meta.get("n_harmful"), a_meta.get("n_harmless"))
        l_n = (l_meta.get("n_harmful"), l_meta.get("n_harmless"))
        if a_n != l_n:
            print("=" * 78)
            print("SKIPPING the archive diff: the baseline is a DIFFERENT CORPUS.")
            print(f"  archive {ARCHIVE}: {a_n[0]} harmful / {a_n[1]} harmless")
            print(f"  live    {LIVE}: {l_n[0]} harmful / {l_n[1]} harmless")
            print("Every quantity would differ for that reason alone, so a DRIFT")
            print("verdict here would carry no information. Re-point at an archive")
            print("of the same scale (argv[1]) once one exists.")
            print("=" * 78)
            return 0
    except Exception:
        pass    # missing metadata: fall through and diff as before

    if not ARCHIVE.exists():
        # A missing baseline is a legitimate state, not a failure: the archives
        # were deleted once they became uninformative (different corpus scale),
        # and this tool runs inside `blank_slate_post.sh`, where a non-zero exit
        # would abort the rest of the post pass over a comparison nobody can make.
        # Exit 0 and say so, exactly as the different-corpus skip above does.
        print("=" * 78)
        print(f"SKIPPING the archive diff: no baseline at {ARCHIVE}.")
        print("Nothing to compare against. Pass an archive root as argv[1] to")
        print("compare two roots (e.g. ./results vs ./results_verify).")
        print("=" * 78)
        return 0
    print(f"archive : {ARCHIVE}\nlive    : {LIVE}\ntolerance: {TOL}\n")
    for model in RQ1_MODELS:
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
