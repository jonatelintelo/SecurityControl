#!/usr/bin/env python
"""E + D — what does `R_control` actually encode, and is it one axis?

SUPERSEDED FOR THE HEADLINE CLAIM — see `tools/soft_refusal_split.py`.
This script projects the WHOLE `undetermined` pool and concludes "MARKER
DETECTOR (soft refusals side with compliances)". That conclusion assumed the pool
was soft refusals; two off-roster judges say it is 847 complied / 513 refused, so
a pool that is 62% compliance projecting with the compliances is exactly what a
CORRECT `R_control` does. Split by adjudicated label, the refused half projects
toward the REFUSED pole (AUC 0.67-0.99 vs 0.43-0.57) and the marker-detector
reading is refuted.

`soft_refusal_summary.json`'s `verdict` field is therefore NOT the current finding.
This script is kept because the pooled projection is the control that motivated the
split, and because the reversal is reportable evidence about the method — not
because its verdict stands.

THE THREAT THIS ADDRESSES
`R_control` is fitted on harmful prompts as `refused` vs `complied`, where
`refused` means the response contains a prefix from Arditi's refusal list. So the
direction could be separating a **decision** (control) or a **vocabulary**
(whether the response opens with "I cannot"). RQ1 claims the former. Nothing in
the pipeline currently distinguishes them, and if it is the latter, C1 and C2 are
measuring a lexical feature and the whole control arm is mis-described.

The `undetermined` pool is the natural experiment that separates them. Those
items are harmful prompts where the prefix rule found NO refusal marker (so the
rule said `complied`) but Llama-Guard, reading the full generation, found no
harmful content. Reading them shows soft refusals — "that is illegal and
unethical... instead, let's...". They are:

  * behaviourally non-compliant  — Guard confirms no harmful content was produced
  * lexically non-refusing       — no refusal marker, by construction
  * EXCLUDED FROM THE FIT        — so they are genuinely held out, not leakage

That makes them a discriminating test:

    if R_control is a CONTROL variable  -> soft refusals project toward `refused`
    if R_control is a MARKER detector   -> soft refusals project toward `complied`

(E) measures where they land. (D) then asks whether soft and hard refusals differ
along an axis of their own, which would make `R_control` more than
one-dimensional — an RQ1 dimensionality result (C4), not a caveat.

NOTHING HERE REFITS OR RELABELS ANYTHING. It reads cached activations and the
frozen directions. The primary labels stay rule-based; see
`tools/adjudicate_undetermined.py` for why.

    python tools/soft_refusal_probe.py
    RESULTS_ROOT=./results_verify python tools/soft_refusal_probe.py
"""
from __future__ import annotations

import gc
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
from core.io_utils import get_logger, load_torch, save_df, save_json  # noqa: E402

ROOT = Path(os.environ.get("RESULTS_ROOT", "./results"))
POSITION = "t_post_inst"      # where R_control is fitted and read
CTRL = "R_control"            # the harmful-side (`under`) contrast


def _cache_path(d: Path) -> Path:
    cj = d / "activations_cache.json"
    return Path(json.loads(cj.read_text())["path"]) if cj.exists() else d / "activations.pt"


def main() -> int:
    log = get_logger("soft-refusal", ROOT / "soft_refusal_probe.log")
    rows_E, rows_D, summary = [], [], {}

    for slug in RQ1_MODELS:
        d = ROOT / "rq1" / slug
        blob_p = _cache_path(d)
        need = [d / "directions.pt", d / "direction_validation.csv",
                d / "refusal_labels.csv", blob_p]
        if not all(p.exists() for p in need):
            log.info(f"[{slug}] skipping — missing {[str(p.name) for p in need if not p.exists()]}")
            continue

        val = pd.read_csv(d / "direction_validation.csv")
        dirs = load_torch(d / "directions.pt")
        if CTRL not in dirs:
            log.warning(f"[{slug}] no {CTRL} direction (the `under` contrast was not "
                        f"estimable on this model) — E needs it; skipping")
            continue

        # The blob is every layer x both positions — 1.3-1.8 GB per model. Keep
        # only the one layer this probe reads and release the rest immediately,
        # or running the roster in one process needs ~8 GB of resident memory.
        blob = load_torch(blob_p, mmap=True)
        idx = pd.DataFrame(blob["index"])
        lab = pd.read_csv(d / "refusal_labels.csv")[
            ["uid", "role", "design", "label", "label_preguard", "truncated"]]
        idx = idx.merge(lab, on=["uid", "role", "design"], how="left",
                        suffixes=("", "_lab"))
        lcol = "label_lab" if "label_lab" in idx.columns else "label"

        # Layer chosen the same way the pipeline chooses it: TRAIN separation.
        sv = val[val.concept == CTRL]
        if sv.empty:
            log.warning(f"[{slug}] {CTRL} absent from direction_validation; skipping")
            continue
        layer = int(sv.loc[sv.train_auc.idxmax(), "layer"])
        direction = dirs[CTRL][layer]
        acts = blob[POSITION][layer].float().clone()
        del blob
        gc.collect()

        harm = idx.harmful.astype(bool).to_numpy()
        lb = idx[lcol].astype(str).to_numpy()
        g_hard = harm & (lb == "refused")
        g_comp = harm & (lb == "complied")
        g_soft = harm & (lb == "undetermined")
        if g_soft.sum() == 0 or g_comp.sum() < 2 or g_hard.sum() < 2:
            log.info(f"[{slug}] group sizes hard={g_hard.sum()} comp={g_comp.sum()} "
                     f"soft={g_soft.sum()} — nothing to test")
            continue

        proj = direction.project(acts).detach().float().numpy().ravel()
        m_hard, m_comp, m_soft = proj[g_hard].mean(), proj[g_comp].mean(), proj[g_soft].mean()

        # Where do the held-out soft refusals sit BETWEEN the two fitted poles?
        # 1.0 = indistinguishable from explicit refusals (control variable)
        # 0.0 = indistinguishable from harmful compliances (marker detector)
        span = m_hard - m_comp
        pos_soft = float((m_soft - m_comp) / span) if abs(span) > 1e-9 else float("nan")

        # AUCs with cluster bootstrap by instruction — the honest sample size,
        # because each instruction appears under 4 roles x 2 designs.
        def cb(maskA, maskB, seed=0):
            sel = maskA | maskB
            return extract.cluster_bootstrap_auc(
                torch.tensor(proj[sel]), list(maskA[sel]),
                list(idx.uid.to_numpy()[sel]), n_boot=1000, seed=seed)

        a_soft_vs_comp = cb(g_soft, g_comp)
        a_hard_vs_comp = cb(g_hard, g_comp)
        a_soft_vs_hard = cb(g_soft, g_hard)

        # Role composition: the soft group is not role-balanced against the poles
        # (soft refusals concentrate in `system` on Qwen2.5), and role has its own
        # direction, so an unbalanced comparison could move the projection for a
        # reason that is not control. Reported as TV, and the test repeated on a
        # role-matched subsample.
        def tv(mA, mB):
            a = pd.Series(idx.role.to_numpy()[mA]).value_counts(normalize=True)
            b = pd.Series(idx.role.to_numpy()[mB]).value_counts(normalize=True)
            k = sorted(set(a.index) | set(b.index))
            return float(0.5 * sum(abs(a.get(r, 0.0) - b.get(r, 0.0)) for r in k))

        rng = np.random.default_rng(0)
        def role_matched(mA, mB):
            """Subsample both groups to a common role distribution."""
            keepA, keepB = np.zeros(len(mA), bool), np.zeros(len(mB), bool)
            roles = idx.role.to_numpy()
            for r in sorted(set(roles)):
                ia = np.flatnonzero(mA & (roles == r))
                ib = np.flatnonzero(mB & (roles == r))
                k = min(len(ia), len(ib))
                if k:
                    keepA[rng.choice(ia, k, replace=False)] = True
                    keepB[rng.choice(ib, k, replace=False)] = True
            return keepA, keepB

        kA, kB = role_matched(g_soft, g_comp)
        a_soft_vs_comp_rb = cb(kA, kB, seed=1) if kA.sum() >= 2 and kB.sum() >= 2 else {}

        verdict = ("CONTROL VARIABLE (soft refusals side with explicit refusals)"
                   if pos_soft > 0.5 else
                   "MARKER DETECTOR (soft refusals side with compliances)")
        log.info(f"[{slug}] L{layer}  n: hard={int(g_hard.sum())} comp={int(g_comp.sum())} "
                 f"soft={int(g_soft.sum())}")
        log.info(f"[{slug}]   mean projection  complied={m_comp:+.4f}  "
                 f"soft={m_soft:+.4f}  refused={m_hard:+.4f}")
        log.info(f"[{slug}]   soft position between poles = {pos_soft:.3f}  -> {verdict}")
        log.info(f"[{slug}]   AUC soft vs complied = {a_soft_vs_comp['auc']:.3f} "
                 f"[{a_soft_vs_comp['ci_low']:.3f},{a_soft_vs_comp['ci_high']:.3f}] "
                 f"(n_instr={a_soft_vs_comp.get('n_clusters')})")
        log.info(f"[{slug}]   AUC soft vs refused  = {a_soft_vs_hard['auc']:.3f} "
                 f"(1.0 would mean soft refusals are their own class)")

        rows_E.append({
            "model": slug, "layer": layer, "position": POSITION,
            "n_hard_refused": int(g_hard.sum()), "n_complied": int(g_comp.sum()),
            "n_soft_refused": int(g_soft.sum()),
            "mean_proj_complied": float(m_comp), "mean_proj_soft": float(m_soft),
            "mean_proj_refused": float(m_hard),
            "soft_position_between_poles": pos_soft,
            "auc_soft_vs_complied": a_soft_vs_comp["auc"],
            "auc_soft_vs_complied_lo": a_soft_vs_comp["ci_low"],
            "auc_soft_vs_complied_hi": a_soft_vs_comp["ci_high"],
            "n_instructions_soft_vs_complied": a_soft_vs_comp.get("n_clusters"),
            "auc_soft_vs_complied_role_matched": a_soft_vs_comp_rb.get("auc", float("nan")),
            "auc_hard_vs_complied": a_hard_vs_comp["auc"],
            "auc_soft_vs_hard": a_soft_vs_hard["auc"],
            "role_tv_soft_vs_complied": tv(g_soft, g_comp),
            "role_tv_soft_vs_refused": tv(g_soft, g_hard),
            "verdict": verdict,
        })

        # ---- (D) is soft-vs-hard refusal an axis of its own? ---------------
        # Both sides are refusals; they differ in whether a marker was emitted.
        # Role-balanced, because the soft group skews toward `system`.
        try:
            d_soft, meta = extract.stratum_balanced_diff_of_means(
                acts[g_soft], acts[g_hard],
                list(idx.role.to_numpy()[g_soft]), list(idx.role.to_numpy()[g_hard]),
                "R_softrefusal", layer, "residual", POSITION, "soft", "hard")
            cos_ctrl = float(torch.dot(d_soft.vector.float(), direction.vector.float()))
            band = extract.random_cosine_band(int(acts.shape[1]), n_samples=2000, seed=0)
            lens = idx.n_tokens.to_numpy() if "n_tokens" in idx.columns else None
            lb_auc = (extract.length_only_baseline(
                list(lens[g_soft | g_hard]), list(g_soft[g_soft | g_hard]))["auc"]
                if lens is not None else float("nan"))
            sp = d_soft.project(acts)
            auc_sh = extract.cluster_bootstrap_auc(
                sp[torch.tensor(g_soft | g_hard)],
                list(g_soft[g_soft | g_hard]),
                list(idx.uid.to_numpy()[g_soft | g_hard]), n_boot=1000, seed=0)
            distinct = abs(cos_ctrl) < band["p97.5"]
            log.info(f"[{slug}]   (D) soft-vs-hard axis: AUC={auc_sh['auc']:.3f} "
                     f"(length-only {lb_auc:.3f}), cos with R_control={cos_ctrl:+.3f} "
                     f"vs null |cos| p97.5={band['p97.5']:.3f} -> "
                     f"{'a DISTINCT axis' if distinct else 'not distinguishable from R_control'}")
            rows_D.append({
                "model": slug, "layer": layer,
                "auc_soft_vs_hard_refusal": auc_sh["auc"],
                "auc_ci_low": auc_sh["ci_low"], "auc_ci_high": auc_sh["ci_high"],
                "n_instructions": auc_sh.get("n_clusters"),
                "length_only_auc": lb_auc,
                "cos_with_R_control": cos_ctrl,
                "null_cos_p97.5": band["p97.5"],
                "is_distinct_axis": bool(distinct),
                "role_balanced": True,
                "shared_strata": ",".join(map(str, meta.get("shared_strata", []))),
            })
        except Exception as e:
            log.warning(f"[{slug}] (D) failed: {type(e).__name__}: {e}")

        summary[slug] = {"layer": layer, "soft_position": pos_soft, "verdict": verdict}

    if not rows_E:
        log.error("no model had the artifacts required (needs R_control, i.e. the "
                  "`under` contrast, plus its activation cache)")
        return 1

    save_df(ROOT / "soft_refusal_projection.csv", pd.DataFrame(rows_E))
    if rows_D:
        save_df(ROOT / "soft_refusal_axis.csv", pd.DataFrame(rows_D))
    save_json(ROOT / "soft_refusal_summary.json", {
        "position": POSITION, "direction": CTRL, "per_model": summary,
        "interpretation": {
            "soft_position_between_poles": (
                "0 = soft refusals project like harmful COMPLIANCES, so R_control "
                "is tracking refusal VOCABULARY; 1 = they project like explicit "
                "REFUSALS, so R_control is tracking the control decision. The soft "
                "items contain no refusal marker and were excluded from the fit, so "
                "this is a held-out test, not a restatement of the fit."),
            "is_distinct_axis": (
                "soft-vs-hard refusal separable along a direction whose |cos| with "
                "R_control is inside the random band -> R_control is not "
                "one-dimensional; feeds C4."),
        },
        "labels_unchanged": True,
    })
    log.info(f"wrote soft_refusal_projection.csv ({len(rows_E)} models)"
             + (f" and soft_refusal_axis.csv ({len(rows_D)})" if rows_D else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
