#!/usr/bin/env python
"""Is the prefix rule the right refusal detector? Refit R_control and measure.

THE QUESTION
`R_control` is fitted on `refused` vs `complied`, where `refused` means the
response contains a prefix from Arditi's list. That rule demonstrably misses a
large, systematic class: soft refusals that decline in substance without any
listed marker ("that is illegal and unethical... instead, let's..."). They land
in `undetermined` — 3.5% to 22.2% of the harmful side depending on the model.

Adding more prefixes does not fix it. The `extended` rule, which already carries
a longer marker list, recovers 0-7.4% of that pool: soft refusals are SEMANTIC,
not lexical, so there is no prefix to add.

So the question is whether a semantic detector — the off-roster LLM judge — makes
a BETTER `R_control`, and that is an empirical question, not a judgement call.

WHAT THIS DOES
Refits `R_control` under alternative label sets, holding the estimator, the
layer-selection rule, the role balancing and the splits fixed. Only the labels
change. Then compares what actually matters:

  * held-out AUC and its cluster-bootstrap CI — is it better separated?
  * split-half stability — is the direction more reproducible, or just fitted to
    more noise?
  * the length-only baseline — is any gain just longer responses?
  * cos with R_harm — does it drift toward being the harm direction?

Two outcomes, both informative:
  better  -> grounds to promote the judge to primary, with evidence to show.
  worse   -> the prefix rule is not merely traditional but correct here, and it
             strengthens C1b: what `R_control` tracks really is EXPLICIT-refusal
             control, and soft refusals genuinely do not share that state.

THE PRIMARY LABELS ARE NOT MODIFIED. The prefix rule is pre-registered and is the
literature's instrument; this writes a comparison table and nothing else.

    python tools/refit_control_labels.py
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


def _blob_path(d: Path) -> Path:
    cj = d / "activations_cache.json"
    return Path(json.loads(cj.read_text())["path"]) if cj.exists() else d / "activations.pt"


def main() -> int:
    log = get_logger("refit-labels", ROOT / "refit_control_labels.log")
    rows = []

    for slug in RQ1_MODELS:
        d = ROOT / "rq1" / slug
        bp = _blob_path(d)
        if not (bp.exists() and (d / "refusal_labels.csv").exists()
                and (d / "direction_validation.csv").exists()):
            log.info(f"[{slug}] skipping — artifacts incomplete")
            continue

        adj_p = ROOT / "label_audit" / "undetermined_adjudicated.csv"
        if not adj_p.exists():
            log.warning(f"[{slug}] no undetermined_adjudicated.csv — run "
                        f"tools/adjudicate_undetermined.py first; skipping")
            continue
        adj = pd.read_csv(adj_p)
        adj = adj[adj.model == slug]
        if not len(adj):
            log.info(f"[{slug}] no adjudicated rows for this model")
            continue

        val = pd.read_csv(d / "direction_validation.csv")
        s = val[val.concept == "R_control"]
        if not len(s):
            log.info(f"[{slug}] R_control was not fitted (cell too thin) — nothing to compare")
            continue
        layer = int(s.loc[s.train_auc.idxmax(), "layer"])

        blob = load_torch(bp, mmap=True)
        idx = pd.DataFrame(blob["index"])
        acts = blob[POSITION][layer].float().clone()
        del blob

        lab = pd.read_csv(d / "refusal_labels.csv")[
            ["uid", "role", "design", "label", "split", "harmful"]]
        idx = idx.merge(lab, on=["uid", "role", "design"], how="left", suffixes=("", "_l"))
        lcol = "label_l" if "label_l" in idx.columns else "label"
        hcol = "harmful_l" if "harmful_l" in idx.columns else "harmful"
        scol = "split_l" if "split_l" in idx.columns else "split"

        # --- label set A: the pre-registered prefix rule (baseline)
        base = idx[lcol].astype(str).to_numpy()
        # --- label set B: undetermined items resolved by the off-roster judge
        key = adj.set_index(["uid", "role", "design"]).adjudicated_label.to_dict()
        judged = base.copy()
        n_moved = 0
        for i, r in enumerate(idx.itertuples()):
            if base[i] == "undetermined":
                v = key.get((r.uid, r.role, r.design))
                if v in ("refused", "complied"):
                    judged[i] = v
                    n_moved += 1

        harm = idx[hcol].astype(bool).to_numpy()
        train = (idx[scol] == "train").to_numpy()
        roles = idx.role.to_numpy()
        lens = idx.n_tokens.to_numpy() if "n_tokens" in idx.columns else None

        for name, labels in (("prefix_rule (primary)", base), ("llm_judge", judged)):
            pos = harm & (labels == "refused")
            neg = harm & (labels == "complied")
            ntr_p, ntr_n = int((pos & train).sum()), int((neg & train).sum())
            if min(ntr_p, ntr_n) < 25:
                log.info(f"[{slug}] {name}: minority train side {min(ntr_p, ntr_n)} < 25 — skipped")
                continue
            try:
                dvec, meta = extract.stratum_balanced_diff_of_means(
                    acts[pos & train], acts[neg & train],
                    list(roles[pos & train]), list(roles[neg & train]),
                    "R_control_refit", layer, "residual", POSITION, "refused", "complied")
            except Exception as e:
                log.warning(f"[{slug}] {name}: fit failed {type(e).__name__}: {e}")
                continue
            te = ~train
            proj = dvec.project(acts)
            cb = extract.cluster_bootstrap_auc(
                proj[torch.tensor(te & (pos | neg))],
                list(pos[te & (pos | neg)]),
                list(idx.uid.to_numpy()[te & (pos | neg)]), n_boot=1000, seed=0)
            sh = extract.split_half_stability(acts[pos & train], acts[neg & train],
                                              n_splits=20, seed=0)
            sh = sh if isinstance(sh, float) else sh.get("mean_cosine", sh.get("mean"))
            lb = (extract.length_only_baseline(list(lens[te & (pos | neg)]),
                                               list(pos[te & (pos | neg)]))["auc"]
                  if lens is not None else float("nan"))
            cos_h = float("nan")
            dirs = load_torch(d / "directions.pt")
            if "R_harm_at_post" in dirs:
                hv = dirs["R_harm_at_post"][layer]
                hv = (hv.vector if hasattr(hv, "vector") else hv).float()
                cos_h = abs(float(torch.dot(dvec.vector.float(), hv)))
            rows.append({
                "model": slug, "label_set": name, "layer": layer,
                "n_train_pos": ntr_p, "n_train_neg": ntr_n,
                "n_test_instructions": cb.get("n_clusters"),
                "auc": cb["auc"], "ci_low": cb["ci_low"], "ci_high": cb["ci_high"],
                "ci_width": round(cb["ci_high"] - cb["ci_low"], 4),
                "split_half_cos": sh, "length_only_auc": lb,
                "abs_cos_with_R_harm_at_post": cos_h,
                "n_undetermined_relabelled": n_moved if name == "llm_judge" else 0})
            log.info(f"[{slug}] {name}: AUC {cb['auc']:.3f} "
                     f"[{cb['ci_low']:.3f},{cb['ci_high']:.3f}] "
                     f"split-half {sh:.3f} length-only {lb:.3f} "
                     f"cos(R_harm) {cos_h:.3f}  (train {ntr_p}/{ntr_n})")

    if not rows:
        log.error("nothing refitted — needs adjudicated labels and a fitted R_control")
        return 1
    df = pd.DataFrame(rows)
    save_df(ROOT / "refit_control_labels.csv", df)

    log.info("-" * 78)
    log.info("VERDICT per model (does the semantic detector beat the prefix rule?)")
    for m, g in df.groupby("model"):
        if g.label_set.nunique() < 2:
            log.info(f"  {m}: only one label set fitted — no comparison")
            continue
        a = g[g.label_set.str.startswith("prefix")].iloc[0]
        b = g[g.label_set == "llm_judge"].iloc[0]
        better = (b.auc > a.auc) and (b.split_half_cos >= a.split_half_cos - 0.01)
        log.info(f"  {m}: AUC {a.auc:.3f} -> {b.auc:.3f}, split-half "
                 f"{a.split_half_cos:.3f} -> {b.split_half_cos:.3f}  => "
                 f"{'JUDGE BETTER' if better else 'prefix rule holds'}")
    log.info("A judge that wins on AUC while LOSING split-half stability is fitting "
             "noise, not finding signal — both must move the right way.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
