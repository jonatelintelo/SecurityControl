#!/usr/bin/env python
"""How many instructions does R_control actually need? Measure, don't assert.

THE QUESTION A REVIEWER WILL ASK
`R_harm` and `R_role` are fitted on 500 instructions per class — about 4x the 128
Arditi et al. use. `R_control` is not: it is fitted on whatever minority cell the
model's own behaviour produces, which across this roster ranges from 6
instructions (qwen3.5-35b-a3b, `under`) to 127 (yi-6b-chat, `under`). At the low
end that is indefensible, and a percentile bootstrap over 6 clusters cannot
resolve a 95% interval at all — its 2.5th percentile IS the minimum observation.

So where is the floor? Picking 100 because the literature uses 128, or 20 because
a bootstrap needs it, are both assertions. This measures it instead: refit
`R_control` on random subsets of increasing MINORITY-class instruction count and
track held-out AUC and split-half stability against n.

  * if both plateau by n ~ 40, fits above that are adequate and the curve shows it
  * if they are still climbing at the largest n available, they are not, and the
    floor belongs at the top of the range

Split-half stability is the more informative of the two and must be read
alongside AUC. A direction can score well on held-out AUC while being unstable
across data splits — that is overfitting, and only the stability curve reveals it.

Subsets are drawn BY INSTRUCTION, never by item: the 8 renderings of one
instruction are near-duplicates, so sampling items would inflate the effective
sample and flatten the curve artificially.

CPU only, from cached activations. Refits nothing that the pipeline reports.

    python tools/control_learning_curve.py
    python tools/control_learning_curve.py --model yi-6b-chat --concept R_control
"""
from __future__ import annotations

import argparse
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
# minority-class instruction counts to probe; the largest available is added
GRID = [5, 10, 20, 30, 40, 60, 80, 100, 125]
N_REPEATS = 5      # random draws per n, so each point has a spread not a point


def _blob(d: Path) -> Path:
    cj = d / "activations_cache.json"
    return Path(json.loads(cj.read_text())["path"]) if cj.exists() else d / "activations.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    ap.add_argument("--concept", default="R_control",
                    choices=["R_control", "R_control_harmless"])
    args = ap.parse_args()
    log = get_logger("learning-curve", ROOT / "control_learning_curve.log")

    CONF = {"R_control": (True, "refused", "complied"),
            "R_control_harmless": (False, "refused", "complied")}
    within_harmful, pos_lab, neg_lab = CONF[args.concept]

    rows = []
    for slug in (args.model or RQ1_MODELS):
        d = ROOT / "rq1" / slug
        bp = _blob(d)
        if not (bp.exists() and (d / "direction_validation.csv").exists()):
            log.info(f"[{slug}] artifacts incomplete — skipping")
            continue
        val = pd.read_csv(d / "direction_validation.csv")
        s = val[val.concept == args.concept]
        if not len(s):
            log.info(f"[{slug}] {args.concept} not fitted on this model — skipping")
            continue
        layer = int(s.loc[s.train_auc.idxmax(), "layer"])

        blob = load_torch(bp, mmap=True)
        idx = pd.DataFrame(blob["index"])
        acts = blob[POSITION][layer].float().clone()
        del blob
        lab = pd.read_csv(d / "refusal_labels.csv")[
            ["uid", "role", "design", "label", "split", "harmful"]]
        idx = idx.merge(lab, on=["uid", "role", "design"], how="left", suffixes=("", "_l"))
        lc = "label_l" if "label_l" in idx.columns else "label"
        hc = "harmful_l" if "harmful_l" in idx.columns else "harmful"
        sc = "split_l" if "split_l" in idx.columns else "split"

        side = idx[hc].astype(bool).to_numpy() if within_harmful else ~idx[hc].astype(bool).to_numpy()
        train = (idx[sc] == "train").to_numpy()
        pos = side & (idx[lc] == pos_lab).to_numpy()
        neg = side & (idx[lc] == neg_lab).to_numpy()
        roles = idx.role.to_numpy(); uids = idx.uid.to_numpy()

        # the minority class is what limits the fit
        mino_mask, majo_mask = (pos, neg) if pos.sum() < neg.sum() else (neg, pos)
        mino_train_uids = sorted(set(uids[mino_mask & train]))
        n_max = len(mino_train_uids)
        log.info(f"[{slug}] {args.concept} @L{layer}: {n_max} minority-class train instructions")
        if n_max < 5:
            log.info(f"[{slug}] too few to draw a curve"); continue

        grid = sorted({g for g in GRID if g <= n_max} | {n_max})
        rng = np.random.default_rng(0)
        for n in grid:
            aucs, shs = [], []
            for rep in range(N_REPEATS if n < n_max else 1):
                keep = set(rng.choice(mino_train_uids, n, replace=False).tolist())
                sub_m = mino_mask & train & np.isin(uids, list(keep))
                sub_M = majo_mask & train
                if sub_m.sum() < 4 or sub_M.sum() < 4:
                    continue
                p_, n_ = (sub_m, sub_M) if mino_mask is pos else (sub_M, sub_m)
                try:
                    dv, _ = extract.stratum_balanced_diff_of_means(
                        acts[p_], acts[n_], list(roles[p_]), list(roles[n_]),
                        args.concept, layer, "residual", POSITION, pos_lab, neg_lab)
                except Exception:
                    continue
                te = ~train
                proj = dv.project(acts)
                sel = te & (pos | neg)
                cb = extract.cluster_bootstrap_auc(
                    proj[torch.tensor(sel)], list(pos[sel]), list(uids[sel]),
                    n_boot=400, seed=rep)
                sh = extract.split_half_stability(acts[p_], acts[n_], n_splits=12, seed=rep)
                sh = sh if isinstance(sh, float) else sh.get("mean_cosine", sh.get("mean"))
                if np.isfinite(cb["auc"]):
                    aucs.append(cb["auc"]); shs.append(sh)
            if not aucs:
                continue
            rows.append({"model": slug, "concept": args.concept, "layer": layer,
                         "n_minority_instructions": n, "n_draws": len(aucs),
                         "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
                         "auc_min": float(np.min(aucs)),
                         "split_half_mean": float(np.mean(shs)),
                         "split_half_sd": float(np.std(shs)),
                         "is_full_sample": bool(n == n_max)})
            log.info(f"  n={n:4d}  AUC {np.mean(aucs):.3f} +-{np.std(aucs):.3f} "
                     f"(min {np.min(aucs):.3f})   split-half {np.mean(shs):.3f} "
                     f"+-{np.std(shs):.3f}   [{len(aucs)} draws]")

    if not rows:
        log.error("no curve produced")
        return 1
    df = pd.DataFrame(rows)
    save_df(ROOT / f"control_learning_curve__{args.concept}.csv", df)

    # ---- where does it stabilise? -------------------------------------------
    log.info("-" * 78)
    log.info("PLATEAU: smallest n whose AUC and split-half are both within 0.02 of "
             "the full-sample value")
    for m, g in df.groupby("model"):
        full = g[g.is_full_sample].iloc[-1]
        ok = g[(abs(g.auc_mean - full.auc_mean) <= 0.02)
               & (abs(g.split_half_mean - full.split_half_mean) <= 0.02)]
        plateau = int(ok.n_minority_instructions.min()) if len(ok) else None
        log.info(f"  {m:26s} full n={int(full.n_minority_instructions):4d} "
                 f"(AUC {full.auc_mean:.3f}, split-half {full.split_half_mean:.3f})"
                 f"  -> plateau at n>={plateau}")
    log.info("Read the SPLIT-HALF column as the primary signal: a direction can hold "
             "its held-out AUC while becoming unstable across splits, and only the "
             "stability curve exposes that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
