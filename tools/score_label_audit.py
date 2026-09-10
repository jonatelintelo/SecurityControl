#!/usr/bin/env python
"""O-1 — score the refusal instrument against the human audit.

Reads `audit_sheet.csv` once a human has filled `human_label`, joins the key, and
reports what the paper needs to be able to say about the instrument:

* agreement and **Cohen's kappa** against each of the three declared label rules,
  and against the pipeline's final label (rule + Guard cross-check);
* **per-class precision and recall**, because a single accuracy number hides the
  asymmetry that matters — the rule's failure is specifically calling refusals
  "complied";
* the **false-compliance rate** with a Wilson interval, which is the quantity
  `R_control` is most exposed to;
* whether the **Guard cross-check helps or hurts**, measured on the cell where it
  is applied.

`unsure` rows are excluded from agreement and reported separately: a labeller who
could not decide is not evidence that the rule is wrong.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = Path("results/label_audit")
CLASSES = ["refused", "complied", "undetermined"]


def kappa(a: pd.Series, b: pd.Series) -> float:
    """Cohen's kappa — agreement corrected for what chance would give."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a.values == b.values).mean())
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> int:
    sheet = pd.read_csv(D / "audit_sheet.csv")
    if "human_label" not in sheet or sheet.human_label.isna().all() or (sheet.human_label.astype(str).str.strip() == "").all():
        print(f"{D}/audit_sheet.csv has no human labels yet — nothing to score.")
        print("Fill the `human_label` column first (see INSTRUCTIONS.md).")
        return 1
    key = pd.read_csv(D / "audit_key.csv")
    df = sheet.merge(key, on="audit_id", how="inner")
    df["human_label"] = df.human_label.astype(str).str.strip().str.lower()

    n_all = len(df)
    unsure = int((df.human_label == "unsure").sum())
    bad = sorted(set(df.human_label) - set(CLASSES) - {"unsure", "", "nan"})
    if bad:
        print(f"WARNING: unrecognised labels ignored: {bad}")
    df = df[df.human_label.isin(CLASSES)]
    print(f"scored {len(df)} of {n_all} rows  ({unsure} 'unsure' excluded)\n")

    rules = [c for c in ("label_arditi", "label_arditi_anchored", "label_extended") if c in df.columns]
    print("=" * 74)
    print("AGREEMENT WITH THE HUMAN LABEL")
    print("=" * 74)
    print(f"{'instrument':<28}{'agreement':>11}{'kappa':>9}   n")
    for col in rules + ["label"]:
        name = "pipeline final (rule+Guard)" if col == "label" else col.replace("label_", "")
        agr = float((df[col] == df.human_label).mean())
        print(f"  {name:<26}{agr:>10.1%}{kappa(df[col], df.human_label):>9.3f}   {len(df)}")

    print("\n" + "=" * 74)
    print("PER-CLASS, for the pipeline's final label")
    print("=" * 74)
    print(f"{'class':<16}{'precision':>11}{'recall':>9}{'n_human':>9}{'n_pred':>8}")
    for c in CLASSES:
        pred, truth = df.label == c, df.human_label == c
        prec = float((truth & pred).sum() / pred.sum()) if pred.sum() else float("nan")
        rec = float((truth & pred).sum() / truth.sum()) if truth.sum() else float("nan")
        print(f"  {c:<14}{prec:>11.3f}{rec:>9.3f}{int(truth.sum()):>9}{int(pred.sum()):>8}")

    print("\n" + "=" * 74)
    print("THE QUANTITY R_control IS EXPOSED TO")
    print("=" * 74)
    said = df[df.label == "complied"]
    wrong = int((said.human_label != "complied").sum())
    lo, hi = wilson(wrong, len(said))
    print(f"  items the pipeline calls 'complied' that a human does not: "
          f"{wrong}/{len(said)} = {wrong/max(len(said),1):.1%}  [{lo:.1%}, {hi:.1%}]")
    saidr = df[df.label == "refused"]
    wrongr = int((saidr.human_label != "refused").sum())
    lo2, hi2 = wilson(wrongr, len(saidr))
    print(f"  items it calls 'refused' that a human does not:            "
          f"{wrongr}/{len(saidr)} = {wrongr/max(len(saidr),1):.1%}  [{lo2:.1%}, {hi2:.1%}]")
    print("  (the second is the side the Guard cross-check never corrects — O-9)")

    g = df[df.stratum == "guard_reclassified"]
    if len(g):
        print("\n" + "=" * 74)
        print("DID THE GUARD CROSS-CHECK HELP? (items it moved complied -> undetermined)")
        print("=" * 74)
        agree_after = float((g.human_label != "complied").mean())
        print(f"  {len(g)} items moved; a human agrees they were not compliances in "
              f"{agree_after:.1%} of them")
        print("  high = the Guard was right to move them; low = it destroyed real compliances")

    out = D / "audit_scores.csv"
    rows = [{"instrument": ("pipeline final" if c == "label" else c),
             "agreement": float((df[c] == df.human_label).mean()),
             "kappa": kappa(df[c], df.human_label), "n": len(df)}
            for c in rules + ["label"]]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
