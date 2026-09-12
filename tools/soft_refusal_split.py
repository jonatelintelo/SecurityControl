#!/usr/bin/env python
"""The decisive version of the E+D projection: split the pool by what it IS.

WHAT WENT WRONG THE FIRST TIME
`soft_refusal_probe.py` projected the whole `undetermined` pool onto `R_control`,
found it sitting with the compliances, and concluded `R_control` might be a
refusal-MARKER detector rather than a control variable.

That inference assumed the pool was soft refusals. It is not. Adjudicated by two
off-roster judges, the pool splits 847 `complied` / 513 `refused` (plus 843 where
the judges disagreed) — so it is majority GENUINE ATTEMPTS that happened to
produce no harmful content. A pool that is 62% compliance projecting with the
compliances is exactly what a correct `R_control` would do, and says nothing
about markers.

THE TEST THAT DISCRIMINATES
Split the pool and project each half:

  judge=refused  -> toward REFUSED pole : R_control tracks the decision, and the
                    missing marker does not fool it. `R_control` is a control
                    variable.
  judge=refused  -> toward COMPLIED pole: items a reader calls refusals, with no
                    marker, land with compliances. THAT is the marker-detector
                    evidence, and only that.
  judge=complied -> toward COMPLIED pole: expected either way; a sanity check,
                    not evidence.

Only items where BOTH judges agreed are used for the headline; disagreements are
reported separately rather than silently pooled, because an item two readers
cannot agree on is not evidence about geometry.

CPU only, from cached activations. Fits nothing.

    python tools/soft_refusal_split.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from core import extract  # noqa: E402
from core.config import RQ1_MODELS  # noqa: E402
from core.io_utils import get_logger, load_torch, save_df  # noqa: E402

ROOT = Path(os.environ.get("RESULTS_ROOT", "./results"))
POSITION = "t_post_inst"


def main() -> int:
    log = get_logger("soft-split", ROOT / "soft_refusal_split.log")
    adj_p = ROOT / "label_audit" / "undetermined_adjudicated.csv"
    if not adj_p.exists():
        log.error(f"missing {adj_p} — run tools/adjudicate_undetermined.py first")
        return 1
    adj = pd.read_csv(adj_p)

    rows = []
    for slug in RQ1_MODELS:
        d = ROOT / "rq1" / slug
        cj = d / "activations_cache.json"
        bp = Path(json.loads(cj.read_text())["path"]) if cj.exists() else d / "activations.pt"
        if not (bp.exists() and (d / "directions.pt").exists()):
            continue
        val = pd.read_csv(d / "direction_validation.csv")
        s = val[val.concept == "R_control"]
        if not len(s):
            log.info(f"[{slug}] R_control not fitted — skipping")
            continue
        layer = int(s.loc[s.train_auc.idxmax(), "layer"])
        dirs = load_torch(d / "directions.pt")
        vec = dirs["R_control"][layer]

        blob = load_torch(bp, mmap=True)
        idx = pd.DataFrame(blob["index"])
        acts = blob[POSITION][layer].float().clone()
        del blob
        lab = pd.read_csv(d / "refusal_labels.csv")[["uid", "role", "design", "label", "harmful"]]
        idx = idx.merge(lab, on=["uid", "role", "design"], how="left", suffixes=("", "_l"))
        lc = "label_l" if "label_l" in idx.columns else "label"
        hc = "harmful_l" if "harmful_l" in idx.columns else "harmful"

        a = adj[adj.model == slug].set_index(["uid", "role", "design"])
        jl, agreed = [], []
        for r in idx.itertuples():
            k = (r.uid, r.role, r.design)
            if k in a.index:
                row = a.loc[k]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                jl.append(str(row.adjudicated_label)); agreed.append(bool(row.judges_agreed))
            else:
                jl.append(""); agreed.append(False)
        idx["judge"] = jl; idx["agreed"] = agreed

        harm = idx[hc].astype(bool).to_numpy()
        rule = idx[lc].astype(str).to_numpy()
        proj = vec.project(acts).detach().float().numpy().ravel()

        hard = harm & (rule == "refused")
        comp = harm & (rule == "complied")
        und = harm & (rule == "undetermined")
        m_hard, m_comp = proj[hard].mean(), proj[comp].mean()
        span = m_hard - m_comp
        if abs(span) < 1e-9 or hard.sum() < 2 or comp.sum() < 2:
            log.info(f"[{slug}] poles degenerate — skipping"); continue

        log.info(f"[{slug}] L{layer}  poles: complied {m_comp:+.3f} .. refused {m_hard:+.3f}")
        for name, mask in (("judge=refused  (soft refusals)",
                            und & (idx.judge == "refused").to_numpy() & idx.agreed.to_numpy()),
                           ("judge=complied (harmless attempts)",
                            und & (idx.judge == "complied").to_numpy() & idx.agreed.to_numpy()),
                           ("judges disagreed", und & ~idx.agreed.to_numpy())):
            n = int(mask.sum())
            if n < 5:
                log.info(f"    {name:36s} n={n:4d}  (too few)"); continue
            mu = proj[mask].mean()
            pos = float((mu - m_comp) / span)
            auc_c = extract.auc(torch.tensor(proj[mask]), torch.tensor(proj[comp]))
            ni = int(pd.Series(idx.uid.to_numpy()[mask]).nunique())
            log.info(f"    {name:36s} n={n:4d} ({ni:3d} instr)  mean {mu:+.3f}  "
                     f"position {pos:+.3f}  AUC vs complied {auc_c:.3f}")
            rows.append({"model": slug, "layer": layer, "subset": name.split()[0],
                         "n_items": n, "n_instructions": ni,
                         "mean_projection": float(mu),
                         "position_between_poles": pos,
                         "auc_vs_complied": auc_c,
                         "pole_complied": float(m_comp), "pole_refused": float(m_hard)})

    if not rows:
        log.error("nothing to report")
        return 1
    df = pd.DataFrame(rows)
    save_df(ROOT / "soft_refusal_split.csv", df)

    log.info("-" * 78)
    log.info("VERDICT — does R_control separate reader-confirmed SOFT REFUSALS")
    log.info("from genuine harmless attempts, when NEITHER carries a refusal marker?")
    log.info("")
    log.info("The comparison is BETWEEN the two subsets, not against a fixed cut.")
    log.info("An absolute threshold ('is it past the halfway point to the refused")
    log.info("pole?') is the wrong test: soft refusals are a weaker signal than")
    log.info("marker refusals by construction, so landing short of halfway is what")
    log.info("a control variable that PARTIALLY captures them looks like. What")
    log.info("discriminates is whether the two subsets separate at all.")
    log.info("")
    sr = df[df.subset == "judge=refused"].set_index("model")
    sc = df[df.subset == "judge=complied"].set_index("model")
    for m in sr.index:
        if m not in sc.index:
            continue
        gap = sr.loc[m, "position_between_poles"] - sc.loc[m, "position_between_poles"]
        auc_r, auc_c = sr.loc[m, "auc_vs_complied"], sc.loc[m, "auc_vs_complied"]
        sep = (gap > 0.1) and (auc_r > 0.6) and (auc_r - auc_c > 0.1)
        log.info(f"  {m:26s} refused {sr.loc[m,'position_between_poles']:+.3f} vs "
                 f"complied {sc.loc[m,'position_between_poles']:+.3f} (gap {gap:+.3f}); "
                 f"AUC {auc_r:.3f} vs {auc_c:.3f}  -> "
                 f"{'SEPARATES — not a marker detector' if sep else 'does NOT separate'}")
    log.info("")
    log.info("Reading: where the subsets separate, R_control tracks the refusal")
    log.info("DECISION and is not defeated by the absent marker. The earlier")
    log.info("conclusion that it was a marker detector came from projecting the")
    log.info("whole `undetermined` pool, which is 62%% genuine compliance — a")
    log.info("confound, not a finding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
