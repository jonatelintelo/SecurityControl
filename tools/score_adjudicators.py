#!/usr/bin/env python
"""Compare every adjudicator of the blind audit sheet against every other.

WHY A SEPARATE TOOL
`score_label_audit.py` scores the label RULES against one adjudicated column. It
cannot answer the question O-1 actually needs: how much do the independent
readers agree with EACH OTHER? Without that, a rule-vs-judge kappa is measured
against a reference of unknown quality, and a low score could mean the rule is
bad or the judge is.

Measured here, the distinction mattered: gemma-3-27b and Mixtral-8x7B agreed with
each other on only 61.9% of the undetermined pool, while a Claude session agreed
with gemma at kappa 0.892. Reading those together says the low figure was
Mixtral's unreliability (it returned no usable class on 254 of 2203 items), not
an intrinsically ambiguous task — a conclusion no pairwise score alone supports.

Adjudicators are whatever is present: the three prefix rules, the pipeline's
final label, each LLM judge's column, and `audit_claude.csv` if a Claude session
recorded one.

    python tools/score_adjudicators.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

D = Path(os.environ.get("AUDIT_DIR", "results/label_audit"))
CLASSES = {"refused", "complied", "degenerate"}


def kappa(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    """Cohen's kappa and raw agreement on rows where BOTH give a real class.

    `unsure`/`no_answer` are excluded rather than scored as disagreement: an
    adjudicator that declined to decide is a missing measurement, and counting it
    as error would penalise honesty and flatter whichever instrument guesses.
    """
    m = a.isin(CLASSES) & b.isin(CLASSES)
    a, b = a[m], b[m]
    if not len(a):
        return float("nan"), float("nan"), 0
    po = float((a.values == b.values).mean())
    cats = sorted(set(a) | set(b))
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    k = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return float(k), po, int(len(a))


def main() -> int:
    sheet = pd.read_csv(D / "audit_sheet.csv")
    key = pd.read_csv(D / "audit_key.csv")
    d = sheet.merge(key, on="audit_id", how="inner")

    cols: dict[str, pd.Series] = {}
    for c, name in (("label_arditi", "rule:arditi (primary)"),
                    ("label_arditi_anchored", "rule:arditi_anchored"),
                    ("label_extended", "rule:extended"),
                    ("label", "pipeline final (rule+Guard)")):
        if c in d.columns:
            cols[name] = d[c].astype(str)
    for c in [c for c in d.columns if c.startswith("label__")]:
        cols[f"judge:{c[len('label__'):]}"] = d[c].astype(str)
    if "adjudicated_label" in d.columns and d.adjudicated_label.notna().any():
        who = (str(d.adjudicator.dropna().iloc[0])
               if "adjudicator" in d.columns and d.adjudicator.notna().any() else "?")
        cols[f"adjudicated ({who[:40]})"] = d.adjudicated_label.astype(str)
    cp = D / "audit_claude.csv"
    if cp.exists():
        cl = pd.read_csv(cp)
        d = d.merge(cl[["audit_id", "adjudicated_label"]].rename(
            columns={"adjudicated_label": "_claude"}), on="audit_id", how="left")
        cols["claude (session)"] = d["_claude"].astype(str)

    names = list(cols)
    print(f"adjudicators present: {len(names)}\n")
    rows = []
    print(f"{'A':38s} {'B':38s} {'agree':>7s} {'kappa':>7s} {'n':>5s}")
    print("-" * 100)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            k, po, n = kappa(cols[a], cols[b])
            rows.append({"a": a, "b": b, "agreement": round(po, 4) if n else None,
                         "kappa": round(k, 4) if n else None, "n": n})
            print(f"{a:38s} {b:38s} {po:6.1%} {k:7.3f} {n:5d}")

    df = pd.DataFrame(rows)
    df.to_csv(D / "adjudicator_agreement.csv", index=False)

    # --- where do the readers disagree with the primary rule, and which way? ---
    prim = "rule:arditi (primary)"
    detail = {}
    if prim in cols:
        for nm in names:
            if nm == prim:
                continue
            m = cols[prim].isin(CLASSES) & cols[nm].isin(CLASSES)
            x, y = cols[prim][m], cols[nm][m]
            dis = pd.crosstab(x[x != y], y[x != y])
            if dis.size:
                detail[nm] = json.loads(dis.to_json())
                print(f"\ndisagreements, {prim} (rows) vs {nm} (cols):")
                print(dis.to_string())
    (D / "adjudicator_disagreements.json").write_text(json.dumps(detail, indent=2))
    print(f"\nwrote {D}/adjudicator_agreement.csv and adjudicator_disagreements.json")
    print("\nRead the rule-vs-reader disagreements by DIRECTION: the count of "
          "'rule said complied, reader said refused' is the soft-refusal class, "
          "and it is the quantity C1b rests on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
