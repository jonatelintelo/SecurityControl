#!/usr/bin/env python
"""RQ1 — Is LLM safety functionally decomposable?  (E1.1 - E1.5)

Playbook: EXPERIMENTS.md > RQ1.

One script per research question, composed of stages. The playbook holds ~30
experiments across RQ1-RQ7; a script each would be ~30 entry points differing
mainly in which cached artifact they read, and activation capture — by far the
expensive step — would be repeated between them. Here it happens once, in
`extract`, and every later stage reads `activations.pt` on CPU. That blob lives
in a scratch cache keyed by the results root, not in the results root itself —
see `Config.cache_dir`; `activations_cache.json` records where it went.

    labels         E1.1 input   refusal labels; the gate on R_control (GPU)
    extract        E1.1         directions + role probe, all layers (GPU)
    nulls          E1.1         random-direction null distributions,
                                length-matched reruns
    geometry       E1.2         pairwise cosines vs null band and split-half floor
    projections    E1.3         projections + correlations between them
    dimensionality E1.4         stratified directions -> r_eff (spectral half)
    emergence      E1.5         separation vs relative depth, all concepts

    python experiments/rq1.py --list
    python experiments/rq1.py --only nulls geometry      # CPU, from cache
    sbatch --export=ALL,MODELS=qwen2.5-7b slurm/scripts/run_gpu.sh experiments/rq1.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from core import extract, model_meta, pools, refusal  # noqa: E402
from core.capture import ActivationCapture  # noqa: E402
from core.config import Config, load_config, relative_depth  # noqa: E402
from core.io_utils import get_logger, load_torch, save_df, save_json, save_torch  # noqa: E402
from core.stages import Context, Pipeline, Stage, add_pipeline_args, maybe_list  # noqa: E402

NAME = "rq1"
CORPUS = "e1_0_corpus"

# Gate artifacts carry the control variant in their name. The two control
# variables are different directions (measured cos 0.72 against a split-half floor
# of 0.944), so a run under one must never overwrite or be mistaken for the other.
# Resolved at import so the stage's `produces` list — and therefore its
# skip-if-complete check — is variant-aware.
_CV = os.environ.get("CONTROL_VARIANT", "auto").lower()
CAUSAL_SUFFIX = "" if _CV == "auto" else f"__{_CV}"

ROLE_CONTRASTS = [("tool", "user"), ("system", "user"), ("assistant", "user")]
# E1.6 reads the intervention's effect at BOTH corpus positions, because PLAN-INF
# asks for it "at later layers AND later tokens" and `t_inst` -> `t_post_inst` is
# an earlier and a later token of the same forward pass. Order fixes the position
# axis of the captured tensors, so it must match what `build(layer, position)`
# is asked for.
READ_POSITIONS = ("t_inst", "t_post_inst")

# The steered-position sweep, pre-registered in the parameter register:
# "steered positions | Arditi: all | Zhao: all | Ours: SWEPT — all real /
# instruction span / t_inst only / post-instruction only".
#
# Arditi and Zhao both steer every real token, so `all_real` is the literature
# anchor and stays Stage C's operating point; the other three answer PLAN-INF's
# question about steering at an *early token* rather than everywhere, and are
# Stage B's deliverable.
TOKEN_SETS = ("all_real", "instruction_span", "t_inst_only", "post_instruction")


def _steer_mask(r, which: str) -> List[bool]:
    """Per-item steering mask in UNPADDED coordinates.

    `_build_position_mask` maps these onto the padded grid using each item's own
    attention mask and raises if the length disagrees, so masks are built against
    `r.n_tokens` and never against a padded width.

    Always returns a list, never None: `all_real` is handled by the caller
    passing `position_masks=None`, and an empty span here returns an all-False
    mask that the caller must detect and skip. Collapsing "steer everything" and
    "steer nothing" into one return value would make an empty
    `post_instruction` span silently steer the whole prompt — the loudest
    possible wrong answer, arriving as a plausible number.
    """
    n = int(r.n_tokens)
    if which == "instruction_span":
        lo, hi = int(r.content_start), int(r.content_end)
    elif which == "t_inst_only":
        lo, hi = int(r.t_inst), int(r.t_inst) + 1
    elif which == "post_instruction":
        lo, hi = int(r.t_inst) + 1, int(r.t_post_inst) + 1
    else:
        raise ValueError(f"unknown token set {which!r}; known: {TOKEN_SETS}")
    m = [False] * n
    for i in range(max(lo, 0), min(max(hi, 0), n)):
        m[i] = True
    return m
MIN_REFUSAL_VARIANCE = 50
MAX_UNDETERMINED_RATE = 0.30
# EXPERIMENTS.md > Replicates and nulls: 1000 draws, with convergence reported.
# A single draw is uninformative — activations are anisotropic enough that a lucky
# random direction separates a contrast well, so only the null *distribution* is
# meaningful.
N_RANDOM = 1000

# concept -> (position, positive side, negative side, harm subset, stratify_by)
#
# `stratify_by="role"` on the control concepts is the confound fix: their two
# sides differ systematically in role composition because refusal rate varies by
# role, so an unbalanced difference-of-means would be partly a role direction.
# The harm concepts need no balancing — every instruction appears under all four
# roles, so both sides carry identical role composition by construction — and the
# imbalance is measured for them anyway to show that.
#
# `R_control_preguard` refits the primary contrast on the labels as they stood
# BEFORE the Llama-Guard cross-check. The check is applied to one side only, so it
# is not neutral; this is the sensitivity fit that prices that in (O-9).
SPECS = {
    "R_harm":              ("t_inst",      "harmful", "harmless", None,       None),
    "R_harm_user":         ("t_inst",      "harmful", "harmless", "user",     None),
    "R_control":           ("t_post_inst", "refused", "complied", "harmful",  "role"),
    "R_harm_at_post":      ("t_post_inst", "harmful", "harmless", None,       None),
    "R_control_harmless":  ("t_post_inst", "refused", "complied", "harmless", "role"),
    "R_control_unbalanced": ("t_post_inst", "refused", "complied", "harmful", None),
    "R_control_preguard":  ("t_post_inst", "refused_preguard", "complied_preguard", "harmful", "role"),
}


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def _render_items(cfg: Config, tok, slug: str) -> pd.DataFrame:
    fitting, _, _ = pools.load_corpus(cfg.results_root / CORPUS)
    from core.positions import render
    rows = []
    roles = cfg.roles_for(slug)
    for ins in fitting:
        for design in cfg.designs:
            for role in roles:
                R = render(tok, ins.text, role, design)
                rows.append({"uid": ins.uid, "harmful": ins.harmful, "source": ins.source,
                             "category": ins.category, "split": ins.split, "role": role,
                             "design": design, "rendered": R, "text": R.text,
                             "n_tokens": R.n_tokens, "n_words": len(ins.text.split())})
    return pd.DataFrame(rows)


def _side_mask(idx: pd.DataFrame, name: str) -> np.ndarray:
    harmful = idx.harmful.astype(bool).to_numpy()
    if name == "harmful":
        return harmful
    if name == "harmless":
        return ~harmful
    if name.endswith("_preguard"):
        return idx.label_preguard.eq(name[: -len("_preguard")]).to_numpy()
    return idx.label.eq(name).to_numpy()


def _subset_mask(idx: pd.DataFrame, subset) -> np.ndarray:
    harmful = idx.harmful.astype(bool).to_numpy()
    if subset is None:
        return np.ones(len(idx), dtype=bool)
    if subset == "user":
        return idx.role.eq("user").to_numpy()
    # Behavioural subsets, so that a harm contrast can be fitted with REFUSAL held
    # constant — PLAN-EXTRACT asks for exactly this and the pooled fit does not
    # provide it.
    if subset in ("refused", "complied"):
        return idx.label.eq(subset).to_numpy()
    return harmful if subset == "harmful" else ~harmful


def _stratified_subspace(a: torch.Tensor, idx: pd.DataFrame, base_m: np.ndarray,
                         mp: np.ndarray, mn: np.ndarray,
                         min_side: int = 5, min_strata: int = 3) -> Optional[torch.Tensor]:
    """Row-space basis of a concept's *stratified* directions, or None.

    One unit direction is fitted inside each stratum (per role, per source), and
    the stack is given an SVD; the returned `[r, d]` matrix spans them. Fitting
    within a stratum is what keeps composition constant on both sides of the
    contrast, so the spread across strata reflects the concept rather than the
    corpus mixture.

    Shared by E1.4b (which asks how many of these axes the *model* needs to
    reproduce a behavioural effect) and E1.2 pass 2 (which compares the resulting
    subspaces between concepts). They must build the subspace identically or the
    `k` measured by one does not describe the basis used by the other.
    """
    strat = []
    for col in ("role", "source"):
        if col not in idx.columns:
            continue
        for lvl in sorted(idx[col].dropna().unique()):
            m = base_m & idx[col].eq(lvl).to_numpy()
            p, n = torch.tensor(m & mp), torch.tensor(m & mn)
            if int(p.sum()) >= min_side and int(n.sum()) >= min_side:
                v = a[p].float().mean(0) - a[n].float().mean(0)
                if v.norm() > 1e-8:
                    strat.append(v / v.norm())
    if len(strat) < min_strata:
        return None
    M = torch.stack(strat)                                     # [n_strata, d]
    return torch.linalg.svd(M.float(), full_matrices=False).Vh  # [r, d]


def _load_cache(ctx: Context):
    """activations + index + fitted artifacts, from the extract stage."""
    d = ctx.cfg.dir(NAME, ctx.model)
    # The blob lives off the home quota (see Config.cache_dir); everything else
    # this function loads is evidence and stays in the results root.
    blob = load_torch(ctx.cfg.cache_dir(NAME, ctx.model) / "activations.pt")
    idx = pd.DataFrame(blob["index"])
    return blob, idx, load_torch(d / "directions.pt"), pd.read_csv(d / "direction_validation.csv")


# ---------------------------------------------------------------------------
# stage: labels  (E1.1 input, and the R_control gate)
# ---------------------------------------------------------------------------
def _budget_ladder(tok, token_ids, ladder, log, slug) -> Dict[str, object]:
    """Choose the generation budget by the pre-registered rule.

    EXPERIMENTS.md: the smallest budget with `undetermined` < 30% AND >= 95% label
    agreement with the next-larger rung. Agreement matters independently of the
    undetermined rate: a budget can look clean while systematically mislabelling
    refusals-in-progress as compliances, and the only way to see that is to ask
    whether a longer budget would have said the same thing.

    Costs one generation pass — see `refusal.at_budget`.
    """
    rungs: Dict[int, List[str]] = {}
    stats: List[Dict] = []
    for b in ladder:
        resp, trunc = refusal.at_budget(tok, token_ids, b)
        labs = [refusal.label_response(r, t).label for r, t in zip(resp, trunc)]
        rungs[b] = labs
        stats.append({
            "budget": b,
            "undetermined_rate": round(float(np.mean([l == "undetermined" for l in labs])), 4),
            "refused": int(sum(l == "refused" for l in labs)),
            "complied": int(sum(l == "complied" for l in labs)),
            "truncated_rate": round(float(np.mean(trunc)), 4),
        })

    for i, s in enumerate(stats):
        nxt = ladder[i + 1] if i + 1 < len(ladder) else None
        s["agreement_with_next"] = (
            round(float(np.mean([a == b for a, b in zip(rungs[s["budget"]], rungs[nxt])])), 4)
            if nxt is not None else None)

    chosen = None
    for s in stats:
        ok_und = s["undetermined_rate"] < MAX_UNDETERMINED_RATE
        ok_agree = s["agreement_with_next"] is None or s["agreement_with_next"] >= 0.95
        if ok_und and ok_agree:
            chosen = s["budget"]
            break
    if chosen is None:
        chosen = ladder[-1]
        log.warning(f"[{slug}] no budget met the rule; falling back to the largest ({chosen})")

    for s in stats:
        log.info(f"[{slug}]   budget {s['budget']:>4}: undetermined {s['undetermined_rate']:.1%}, "
                 f"refused {s['refused']}, complied {s['complied']}, "
                 f"truncated {s['truncated_rate']:.1%}, "
                 f"agreement-with-next {s['agreement_with_next']}")
    log.info(f"[{slug}] chosen budget = {chosen} (rule: smallest with undetermined < "
             f"{MAX_UNDETERMINED_RATE:.0%} and >= 95% agreement with the next rung)")
    return {"ladder": ladder, "rungs": stats, "chosen": chosen,
            "rule": "smallest budget with undetermined < 0.30 and agreement_with_next >= 0.95"}


def stage_labels(ctx: Context) -> None:
    from core.model_io import load_model, resolve_device
    cfg, log, slug = ctx.cfg, ctx.log, ctx.model
    device = resolve_device()
    model, tok = load_model(ctx.spec().model_id, device, logger=log)

    df = _render_items(cfg, tok, ctx.model)
    log.info(f"[{slug}] rendered {len(df)} items")

    # Generate ONCE at the largest rung; every shorter budget is a prefix of it.
    ladder = sorted(cfg.refusal_budget_ladder)
    log.info(f"[{slug}] generating at max budget {ladder[-1]} (ladder {ladder})")
    responses, truncated, token_ids = refusal.generate(
        model, tok, list(df.text), device, ladder[-1],
        cfg.batch_size, log_every=20, logger=log)

    budget = _budget_ladder(tok, token_ids, ladder, log, slug)
    full_responses, _ = refusal.at_budget(tok, token_ids, ladder[-1])
    responses, truncated = refusal.at_budget(tok, token_ids, budget["chosen"])
    save_json(ctx.out / "budget_ladder.json", budget)

    calib = refusal.calibrate(responses)
    log.info(f"[{slug}] degeneracy: {calib['n_excluded']}/{calib['n_responses']} "
             f"({calib['exclusion_rate']:.2%}) {calib['by_reason']}")
    save_json(ctx.out / "degeneracy_calibration.json", calib)

    # All three declared label rules, so no claim rests on an undeclared choice
    # of marker set. `arditi` is primary; the others are sensitivity.
    disagree = refusal.rule_disagreement(responses, truncated)
    log.info(f"[{slug}] label-rule disagreement: {disagree['disagreement_rate']}")
    save_json(ctx.out / "label_rule_disagreement.json", disagree)

    labs = [refusal.label_response(r, t, refusal.PRIMARY_RULE) for r, t in zip(responses, truncated)]
    df["response"] = responses
    df["response_full"] = full_responses
    df["label"] = [l.label for l in labs]
    df["label_reason"] = [l.reason for l in labs]
    df["truncated"] = [l.truncated for l in labs]
    for rule in refusal.LABEL_RULES:
        df[f"label_{rule}"] = [refusal.label_response(r, t, rule).label
                               for r, t in zip(responses, truncated)]

    # `label_preguard` preserves the pre-cross-check label so `R_control` can be
    # refitted WITHOUT the Guard filter. The filter is applied to one side of the
    # contrast only, so it is not neutral; the sensitivity fit is how that cost is
    # paid rather than assumed away (O-9).
    df["label_preguard"] = df["label"]

    # Guard cross-check, only where response harmfulness is genuinely the
    # question: a harmful prompt labelled `complied` whose response the guard
    # calls safe is an evasive non-answer, not a compliance.
    cell = df[(df.harmful) & (df.label == "complied")]
    guard_stats: Dict[str, object] = {"attempted": False}
    if len(cell):
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            from core.guard import Guard
            g = Guard(device=device, logger=log)
            # Judge the FULL generation, not the label-budget truncation.
            #
            # The two instruments answer questions with different natural
            # horizons: the refusal-prefix rule asks "did the model decline",
            # decided in the opening tokens (Zhao's premise, and why the short
            # budget is legitimate for the LABEL); Llama-Guard asks "is this
            # response harmful", which is a property of the whole response. Asking
            # it about a 48-token stub systematically biases it toward `safe`,
            # because a response that has not yet reached the harmful content
            # looks harmless — and every reclassified item is truncated, so this
            # is not a hypothetical.
            #
            # Both verdicts are recorded: their disagreement measures how much of
            # the cross-check was truncation rather than evasion.
            v_full = g.judge_with_raw(list(cell.text), list(cell.response_full), cfg.batch_size)
            v_trunc = g.judge_with_raw(list(cell.text), list(cell.response), cfg.batch_size)
            g.release()
            safe = [i for i, (uns, _) in zip(cell.index, v_full) if not uns]
            df.loc[safe, "label"] = "undetermined"
            df.loc[safe, "label_reason"] = "guard_says_response_safe"
            n_flip = sum(1 for (uf, _), (ut, _) in zip(v_full, v_trunc) if uf != ut)
            guard_stats = {
                "attempted": True, "judged_on": "full_generation",
                "n_checked": len(cell), "n_reclassified": len(safe),
                "reclassified_rate": round(len(safe) / len(cell), 4),
                "n_unsafe_full": int(sum(u for u, _ in v_full)),
                "n_unsafe_truncated": int(sum(u for u, _ in v_trunc)),
                "verdict_flips_full_vs_truncated": n_flip,
                "flip_rate": round(n_flip / len(cell), 4)}
            log.info(f"[{slug}] guard (on full generation): {len(safe)}/{len(cell)} "
                     f"({len(safe)/len(cell):.1%}) reclassified to undetermined")
            log.info(f"[{slug}] guard unsafe: full={guard_stats['n_unsafe_full']} "
                     f"truncated={guard_stats['n_unsafe_truncated']} "
                     f"(flips {n_flip}, {n_flip/len(cell):.1%}) — the gap is the truncation bias")
        except Exception as e:
            guard_stats = {"attempted": True, "error": f"{type(e).__name__}: {str(e)[:140]}"}
            log.warning(f"[{slug}] guard FAILED: {guard_stats['error']}")

    usable = df[df.label != "undetermined"]
    und = float(df.label.eq("undetermined").mean())
    strata = {}
    for lab, nm in [(True, "within_harmful"), (False, "within_harmless")]:
        s = usable[usable.harmful == lab]
        ref, com = int(s.label.eq("refused").sum()), int(s.label.eq("complied").sum())
        minority = "complied" if com < ref else "refused"
        mino = s[s.label.eq(minority)]
        # The split is BY INSTRUCTION, so the test-side count is what bounds the
        # held-out estimate — a cell of 60 spread over 12 instructions is 12
        # independent items, not 60.
        strata[nm] = {"refused": ref, "complied": com, "minority": min(ref, com),
                      "minority_class": minority,
                      "minority_instructions": int(mino.uid.nunique()),
                      "minority_test_items": int(mino.split.eq("test").sum()),
                      "minority_roles": {k: int(v) for k, v in mino.role.value_counts().items()},
                      "usable": min(ref, com) >= MIN_REFUSAL_VARIANCE}
        log.info(f"[{slug}]   {nm}: refused={ref} complied={com} minority={min(ref, com)} "
                 f"({minority}, {strata[nm]['minority_instructions']} instructions, "
                 f"{strata[nm]['minority_test_items']} test items) "
                 f"{'USABLE' if strata[nm]['usable'] else 'too few'}")
        log.info(f"[{slug}]     minority by role: {strata[nm]['minority_roles']}")

    # Refusal rate by role and by source: the levers the corpus-widening rule
    # relies on, and — for role — the reason `R_control` must be role-balanced.
    by_role = usable.groupby("role").label.apply(lambda s: round(float(s.eq("refused").mean()), 3))
    by_source = usable[usable.harmful].groupby("source").label.apply(
        lambda s: round(float(s.eq("refused").mean()), 3))
    log.info(f"[{slug}] refusal rate by role: {by_role.to_dict()}")
    log.info(f"[{slug}] refusal rate by harmful source: {by_source.to_dict()}")

    checks = {"refusal_variance": {"ok": any(v["usable"] for v in strata.values()), **strata},
              "undetermined_rate": {"ok": und < MAX_UNDETERMINED_RATE, "rate": round(und, 4)}}
    save_df(ctx.out / "refusal_labels.csv", df.drop(columns=["rendered"]))
    save_json(ctx.out / "labels_checks.json", {
        "checks": checks, "guard": guard_stats, "budget": budget,
        "refusal_rate_by_role": by_role.to_dict(),
        "refusal_rate_by_harmful_source": by_source.to_dict(),
        "label_rule_disagreement": disagree["disagreement_rate"]})

    failed = [k for k, v in checks.items() if not v["ok"]]
    if failed and not cfg.fast_dev:
        raise SystemExit(f"[{slug}] RQ1 labels gate FAILED: {failed}. "
                         "Neither harm stratum has enough minority-class items, or too many "
                         "undetermined. Widen the corpus (role framings, then milder harmful "
                         "sources) — NOT jailbreak framings, they leak into RQ4.")
    if failed:
        log.warning(f"[{slug}] gate not met under FAST_DEV ({failed}) — advisory only")


# ---------------------------------------------------------------------------
# stage: extract  (E1.1)
# ---------------------------------------------------------------------------
def _fit(name, acts, mp, mn, sub, df, layers, pc, nc, position, cfg, log,
         balanced=False, stratify_by=None):
    """Fit one concept at every layer, with its validation record.

    `stratify_by` names a column whose distribution must be equalised across the
    two sides before the difference of means is taken — see
    `extract.stratum_balanced_diff_of_means`. It is required for `R_control`,
    whose two sides differ systematically in role composition, and is measured
    (but not applied) for every other concept so that "unconfounded" is a reported
    quantity rather than an assumption.
    """
    is_tr = df.split.eq("train").to_numpy()
    tr_p, tr_n = torch.tensor(mp & is_tr & sub), torch.tensor(mn & is_tr & sub)
    te_p, te_n = torch.tensor(mp & ~is_tr & sub), torch.tensor(mn & ~is_tr & sub)
    if min(map(int, (tr_p.sum(), tr_n.sum(), te_p.sum(), te_n.sum()))) < 2:
        log.warning(f"  {name}: too few examples — skipped")
        return {}, []

    lengths = df.n_tokens.to_numpy()
    sel = (mp | mn) & sub
    lb = extract.length_only_baseline(list(lengths[sel]), list(mp[sel]))

    # Composition imbalance is measured for role AND design on every concept.
    strata_cols = [c for c in ("role", "design") if c in df.columns]
    imbalance = {c: extract.composition_imbalance(
        df[c].to_numpy()[tr_p.numpy()].tolist(), df[c].to_numpy()[tr_n.numpy()].tolist())
        for c in strata_cols}
    for c, v in imbalance.items():
        flag = "  <-- balanced" if c == stratify_by else ""
        if v["total_variation"] >= 0.05 or c == stratify_by:
            log.info(f"  {name}: {c} composition TV={v['total_variation']:.3f}{flag}")

    bal_info = None
    dirs, rows = {}, []
    for li in layers:
        a = acts[li]
        if stratify_by:
            d, bal_info = extract.stratum_balanced_diff_of_means(
                a[tr_p], a[tr_n],
                df[stratify_by].to_numpy()[tr_p.numpy()].tolist(),
                df[stratify_by].to_numpy()[tr_n.numpy()].tolist(),
                name, li, "residual", position, pc, nc)
        else:
            d = extract.diff_of_means(a[tr_p], a[tr_n], name, li, "residual", position, pc, nc)
        dirs[li] = d
        sep = extract.separation(d, a[te_p], a[te_n])
        te_all = te_p | te_n
        cb = extract.cluster_bootstrap_auc(d.project(a[te_all]), te_p[te_all].tolist(),
                                           df.uid.to_numpy()[te_all.numpy()].tolist(), seed=cfg.seed)
        rnd = extract.random_direction_like(d, seed=cfg.seed + li)
        sh = extract.split_half_stability(a[tr_p], a[tr_n], seed=cfg.seed)
        rec = {**d.to_record(), "relative_depth": round(relative_depth(li, len(layers)), 4),
               # Selection quantity. Never reported as a result — it is fitted on
               # these very items — but it is what names a layer, so that the
               # reported test AUC is never chosen on the test split.
               "train_auc": extract.separation(d, a[tr_p], a[tr_n])["auc"],
               "auc": sep["auc"], "auc_ci_low": cb["ci_low"], "auc_ci_high": cb["ci_high"],
               "cohens_d": sep["cohens_d"],
               "random_auc": extract.separation(rnd, a[te_p], a[te_n])["auc"],
               "split_half_cos": sh["mean"], "split_half_sd": sh["sd"],
               "length_only_auc": lb["auc"], "n_test_pos": int(te_p.sum()),
               "n_test_neg": int(te_n.sum()), "n_test_instructions": cb["n_clusters"]}
        if balanced:
            b = extract.balanced_auc(d.project(a[te_p]), d.project(a[te_n]), seed=cfg.seed)
            rec.update(balanced_auc=b["auc"], balanced_ci_low=b["ci_low"],
                       balanced_ci_high=b["ci_high"])
        for c, v in imbalance.items():
            rec[f"imbalance_tv_{c}"] = v["total_variation"]
        rec["stratified_by"] = stratify_by or ""
        rows.append(rec)

    # Layer is selected on the TRAIN split by a rule fixed in advance, then
    # reported on test — EXPERIMENTS.md > Layer selection. Picking the best test
    # AUC would be selection on the evaluation set.
    best = max(rows, key=lambda r: r["train_auc"])
    log.info(f"  {name:<20} AUC {best['auc']:.3f} [{best['auc_ci_low']:.2f},{best['auc_ci_high']:.2f}] "
             f"L{best['layer']} d={best['relative_depth']:.2f} | length-only {lb['auc']:.3f} "
             f"| split-half {best['split_half_cos']:.3f} | n_instr={best['n_test_instructions']}")
    if bal_info:
        log.info(f"  {name:<20} balanced on {stratify_by}: shared={bal_info['shared_strata']} "
                 f"dropped={bal_info['dropped_strata']} "
                 f"weight kept pos={bal_info['weight_retained_positive']:.2f} "
                 f"neg={bal_info['weight_retained_negative']:.2f}")
    return dirs, rows


def stage_extract(ctx: Context) -> None:
    from core.model_io import load_model, resolve_device
    cfg, log, slug = ctx.cfg, ctx.log, ctx.model
    device = resolve_device()
    model, tok = load_model(ctx.spec().model_id, device, logger=log)
    n_layers = model_meta.num_layers(model.config)
    layers = list(range(n_layers))

    df = _render_items(cfg, tok, ctx.model)
    labels = pd.read_csv(cfg.dir(NAME, slug) / "refusal_labels.csv")[
        ["uid", "role", "design", "label", "label_preguard", "truncated"]]
    df = df.merge(labels, on=["uid", "role", "design"], how="left")
    if int(df.label.isna().sum()):
        raise ValueError("items missing a refusal label; corpora out of sync")

    cap = ActivationCapture(model, site="residual")
    log.info(f"[{slug}] capturing residual at both positions, {n_layers} layers...")
    acts = cap.at_positions(tok, list(df.rendered), ("t_inst", "t_post_inst"), layers,
                            cfg.batch_size, device, log_every=25, logger=log)

    directions, val = {}, []
    for name, (position, pos, neg, subset, strat) in SPECS.items():
        d, r = _fit(name, acts[position], _side_mask(df, pos), _side_mask(df, neg),
                    _subset_mask(df, subset), df, layers, pos, neg, position, cfg, log,
                    balanced=name.startswith("R_control"), stratify_by=strat)
        if d:
            directions[name], val = d, val + r
    save_df(ctx.out / "direction_validation.csv", pd.DataFrame(val))

    # role probe, token level
    log.info(f"[{slug}] role probe: token-level capture, {n_layers} layers...")
    tok_acts, owner = cap.at_content_tokens(tok, list(df.rendered), layers,
                                            cfg.max_content_tokens, cfg.batch_size,
                                            device, seed=cfg.seed, log_every=25, logger=log)
    own = owner.tolist()
    t_role = [df.role.iloc[i] for i in own]
    t_uid = [df.uid.iloc[i] for i in own]
    t_len = [float(df.n_tokens.iloc[i]) for i in own]
    t_train = torch.tensor([df.split.iloc[i] == "train" for i in own])

    probes, prows = {}, []
    n_cls = len(set(t_role))
    for li in layers:
        x = tok_acts[li]
        p = extract.train_role_probe(x[t_train], [r for r, m in zip(t_role, t_train.tolist()) if m],
                                     x[~t_train], [r for r, m in zip(t_role, (~t_train).tolist()) if m],
                                     layer=li, site="residual", seed=cfg.seed)
        probes[li] = p
        te = (~t_train).tolist()
        pred = [p.classes[i] for i in (x[~t_train].float() @ p.coef.T + p.intercept).argmax(1).tolist()]
        truth = [r for r, m in zip(t_role, te) if m]
        ab = extract.cluster_bootstrap_accuracy(
            [q == t for q, t in zip(pred, truth)],
            [t_uid[i] for i, m in enumerate(te) if m], seed=cfg.seed)
        tr_pred = [p.classes[i] for i in
                   (x[t_train].float() @ p.coef.T + p.intercept).argmax(1).tolist()]
        tr_truth = [r for r, m in zip(t_role, t_train.tolist()) if m]
        row = {"layer": li, "relative_depth": round(relative_depth(li, n_layers), 4),
               # selection quantity only — never reported as a result
               "train_accuracy": float(np.mean([q == t for q, t in zip(tr_pred, tr_truth)])),
               "accuracy": p.accuracy, "acc_ci_low": ab["ci_low"], "acc_ci_high": ab["ci_high"],
               "chance": 1.0 / n_cls, "n_test_instructions": ab["n_clusters"]}
        for a_r, b_r in ROLE_CONTRASTS:
            ka = torch.tensor([r == a_r for r in t_role])
            kb = torch.tensor([r == b_r for r in t_role])
            pair = ka | kb
            dm = extract.diff_of_means(x[ka & t_train], x[kb & t_train],
                                       f"role_{a_r}_vs_{b_r}", li, "residual",
                                       "content_tokens", a_r, b_r)
            uids_pair = [t_uid[i] for i in torch.nonzero(pair & ~t_train).flatten().tolist()]
            cbr = extract.cluster_bootstrap_auc(
                dm.project(x[pair & ~t_train]), ka[pair & ~t_train].tolist(),
                uids_pair, seed=cfg.seed)
            row[f"auc_{a_r}_{b_r}"] = cbr["auc"]
            row[f"auc_ci_low_{a_r}_{b_r}"] = cbr["ci_low"]
            row[f"auc_ci_high_{a_r}_{b_r}"] = cbr["ci_high"]
            row[f"cos_probe_vs_dom_{a_r}_{b_r}"] = float(p.contrast(a_r, b_r).vector @ dm.vector)

            # length-matched rerun: authentic markup makes length informative
            idxs = torch.nonzero(pair).flatten().tolist()
            mm = extract.length_matched_mask([t_len[i] for i in idxs],
                                             [t_role[i] == a_r for i in idxs], seed=cfg.seed)
            sel = torch.zeros(len(t_role), dtype=torch.bool)
            sel[torch.tensor(idxs)[mm]] = True
            mtr, mte = sel & t_train, sel & ~t_train
            if min(int((mtr & ka).sum()), int((mtr & kb).sum()),
                   int((mte & ka).sum()), int((mte & kb).sum())) >= 2:
                dmm = extract.diff_of_means(x[mtr & ka], x[mtr & kb],
                                            f"role_{a_r}_vs_{b_r}_matched", li, "residual",
                                            "content_tokens", a_r, b_r)
                cbm = extract.cluster_bootstrap_auc(
                    dmm.project(x[mte]), ka[mte].tolist(),
                    [t_uid[i] for i in torch.nonzero(mte).flatten().tolist()], seed=cfg.seed)
                sel_i = torch.nonzero(sel).flatten().tolist()
                row[f"auc_matched_{a_r}_{b_r}"] = cbm["auc"]
                row[f"auc_matched_ci_low_{a_r}_{b_r}"] = cbm["ci_low"]
                row[f"auc_matched_ci_high_{a_r}_{b_r}"] = cbm["ci_high"]
                row[f"length_matched_auc_{a_r}_{b_r}"] = extract.length_only_baseline(
                    [t_len[i] for i in sel_i], [t_role[i] == a_r for i in sel_i])["auc"]
        prows.append(row)

    save_df(ctx.out / "role_probe.csv", pd.DataFrame(prows))
    save_torch(ctx.out / "directions.pt", directions)
    save_torch(ctx.out / "role_probes.pt", probes)
    _cache = ctx.cfg.cache_dir(NAME, slug)
    save_torch(_cache / "activations.pt",
               {"t_inst": acts["t_inst"], "t_post_inst": acts["t_post_inst"],
                "index": df.drop(columns=["rendered"]).to_dict("list")})
    # A pointer in the results root, so a later reader (or a human) can find the
    # cache without re-deriving the hash, and so the manifest records where the
    # activations for THIS root went.
    save_json(ctx.out / "activations_cache.json", {
        "model": slug, "results_root": str(cfg.results_root.resolve()),
        "path": str(_cache / "activations.pt"),
        "note": "regenerable cache, deliberately outside the results root; "
                "keyed by results root so `results` and `results_verify` never share it"})
    log.info(f"[{slug}] activations cached at {_cache / 'activations.pt'}")
    save_json(ctx.out / "extract_summary.json", {
        "model": slug, "n_layers": n_layers, "d_model": model_meta.hidden_size(model.config),
        "concepts": sorted(directions), "n_items": len(df)})


# ---------------------------------------------------------------------------
# stage: nulls  (E1.1 validation)
# ---------------------------------------------------------------------------
def stage_nulls(ctx: Context) -> None:
    """Random-direction NULL DISTRIBUTIONS and length-matched reruns.

    A single random direction is not a usable control. Real activations are
    strongly anisotropic, so one draw can separate the classes by luck and is as
    likely to flatter a direction as to challenge it; only the null DISTRIBUTION
    is interpretable. N_RANDOM draws, with the p95 reported against draw count so
    convergence is visible.

    The length-matched rerun answers the separate question of whether a direction
    tracks sequence length. Harmful prompts run longer than harmless ones in the
    standard corpora and the authentic role markup differs in length, so wherever
    a direction fails to beat its length-only baseline, the matched refit is the
    number that may be reported.
    """
    cfg, log = ctx.cfg, ctx.log
    blob, idx, dirs, val = _load_cache(ctx)
    rows = []
    for concept, (position, pos, neg, subset, _strat) in SPECS.items():
        if concept not in dirs:
            continue
        sub = val[val.concept == concept]
        li = int(sub.loc[sub.train_auc.idxmax(), "layer"])
        observed = float(sub.loc[sub.train_auc.idxmax(), "auc"])
        test = idx.split.eq("test").to_numpy() & _subset_mask(idx, subset)
        mp = torch.tensor(test & _side_mask(idx, pos))
        mn = torch.tensor(test & _side_mask(idx, neg))
        if int(mp.sum()) < 2 or int(mn.sum()) < 2:
            continue
        a = blob[position][li]

        null = extract.random_direction_null(a[mp], a[mn], N_RANDOM, seed=cfg.seed)
        pval = extract.null_pvalue(observed, a[mp], a[mn], N_RANDOM, seed=cfg.seed)

        # length-matched rerun, refitting on the matched TRAIN subset
        lengths = idx.n_tokens.to_numpy()
        both = _subset_mask(idx, subset) & (_side_mask(idx, pos) | _side_mask(idx, neg))
        ii = np.nonzero(both)[0]
        mm = extract.length_matched_mask(list(lengths[ii]),
                                         list(_side_mask(idx, pos)[ii]), seed=cfg.seed)
        keep = np.zeros(len(idx), dtype=bool)
        keep[ii[mm.numpy()]] = True
        tr = keep & idx.split.eq("train").to_numpy()
        te = keep & idx.split.eq("test").to_numpy()
        mrow = {}
        p_tr = torch.tensor(tr & _side_mask(idx, pos)); n_tr = torch.tensor(tr & _side_mask(idx, neg))
        p_te = torch.tensor(te & _side_mask(idx, pos)); n_te = torch.tensor(te & _side_mask(idx, neg))
        if min(map(int, (p_tr.sum(), n_tr.sum(), p_te.sum(), n_te.sum()))) >= 2:
            dm = extract.diff_of_means(a[p_tr], a[n_tr], f"{concept}_matched", li,
                                       "residual", position, pos, neg)
            te_all = p_te | n_te
            cb = extract.cluster_bootstrap_auc(
                dm.project(a[te_all]), p_te[te_all].tolist(),
                idx.uid.to_numpy()[te_all.numpy()].tolist(), seed=cfg.seed)
            ki = np.nonzero(keep)[0]
            mrow = {"matched_auc": cb["auc"], "matched_ci_low": cb["ci_low"],
                    "matched_ci_high": cb["ci_high"],
                    "matched_length_only": extract.length_only_baseline(
                        list(lengths[ki]), list(_side_mask(idx, pos)[ki]))["auc"],
                    "matched_n_items": int(keep.sum()),
                    "matched_n_instructions": cb["n_clusters"]}

        rows.append({"model": ctx.model, "concept": concept, "layer": li,
                     "relative_depth": float(sub.loc[sub.train_auc.idxmax(), "relative_depth"]),
                     "auc": observed, **null, "p_value": pval,
                     "beats_null_p95": observed > null["null_p95"],
                     "length_only_auc": float(sub.loc[sub.train_auc.idxmax(), "length_only_auc"]),
                     **mrow})
        m = f" | matched {mrow['matched_auc']:.3f} (len {mrow['matched_length_only']:.3f})" if mrow else ""
        log.info(f"  {concept:<20} AUC {observed:.3f} | null p95 {null['null_p95']:.3f} "
                 f"max {null['null_max']:.3f} | p={pval:.4f} "
                 f"{'PASS' if observed > null['null_p95'] else 'FAILS NULL'}{m}")

    save_df(ctx.out / "null_distributions.csv", pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# stage: geometry (E1.2)
# ---------------------------------------------------------------------------
def stage_geometry(ctx: Context) -> None:
    """Pairwise cosines, read against the random null band AND the split-half floor.

    A cosine is only meaningful if it exceeds what random unit vectors give
    (~1/sqrt(d)) *and* is large relative to how much a direction varies against
    itself. Comparisons use the common position `t_post_inst`; cross-position
    cosines are not claimed.
    """
    cfg, log = ctx.cfg, ctx.log
    _, _, dirs, val = _load_cache(ctx)
    d_model = next(iter(next(iter(dirs.values())).values())).vector.numel()
    band = extract.random_cosine_band(d_model, seed=cfg.seed)
    log.info(f"  null band: mean {band['mean']:+.4f} sd {band['sd']:.4f} "
             f"p97.5 {band['p97.5']:+.4f} (analytic 1/sqrt(d) = {band['analytic_sd']:.4f})")

    # Role was missing from the geometry, which made RQ1's headline question --
    # "at layer L, is harmfulness anti-aligned with the system role?" --
    # unanswerable. The role PROBE lives on content tokens, so its contrasts are
    # not position-comparable with R_harm (t_inst) or R_control (t_post-inst).
    # Fix: derive role contrast directions by difference-of-means at the SAME
    # positions, from the cached activations, so the comparison is licensed.
    blob, idx, _, _ = _load_cache(ctx)
    train = idx.split.eq("train").to_numpy()
    n_layers_model = len(blob["t_post_inst"])   # the model's depth, for relative_depth
    dirs = dict(dirs)
    for position in ("t_inst", "t_post_inst"):
        for a_r, b_r in ROLE_CONTRASTS:
            ma = train & idx.role.eq(a_r).to_numpy()
            mb = train & idx.role.eq(b_r).to_numpy()
            if ma.sum() < 5 or mb.sum() < 5:
                continue
            name = f"R_role_{a_r}_vs_{b_r}@{position}"
            per_layer = {}
            for li in sorted(blob[position]):
                A = blob[position][li]
                per_layer[li] = extract.diff_of_means(
                    A[torch.tensor(ma)], A[torch.tensor(mb)], name, li,
                    "residual", position, a_r, b_r)
            dirs[name] = per_layer

    shf = {c: dict(zip(val[val.concept == c].layer, val[val.concept == c].split_half_cos))
           for c in val.concept.unique()}
    names = sorted(dirs)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            common = sorted(set(dirs[a]) & set(dirs[b]))
            for li in common:
                cos = float(dirs[a][li].vector @ dirs[b][li].vector)
                floor = min(shf.get(a, {}).get(li, float("nan")),
                            shf.get(b, {}).get(li, float("nan")))
                rows.append({"concept_a": a, "concept_b": b, "layer": li,
                             # n_layers, not len(common): relative depth is a
                             # property of the MODEL, and a pair that shares only
                             # part of the stack would otherwise be reported at a
                             # depth computed against the wrong denominator.
                             "relative_depth": round(relative_depth(li, n_layers_model), 4),
                             "cosine": cos, "abs_cosine": abs(cos),
                             "null_p97_5": band["p97.5"], "split_half_floor": floor,
                             "beyond_null": abs(cos) > band["p97.5"],
                             "position_a": dirs[a][li].position, "position_b": dirs[b][li].position,
                             "same_position": dirs[a][li].position == dirs[b][li].position})
    df = pd.DataFrame(rows)
    save_df(ctx.out / "geometry_cosines.csv", df)
    save_json(ctx.out / "geometry_null_band.json", band)
    if len(df):
        same = df[df.same_position]
        is_role = same.concept_a.str.startswith("R_role") ^ same.concept_b.str.startswith("R_role")
        cross = same[is_role]
        log.info("  strongest ROLE vs safety-variable cosines (RQ1's headline question):")
        for r in cross.reindex(cross.abs_cosine.sort_values(ascending=False).index).head(6).itertuples():
            log.info(f"    {r.concept_a} vs {r.concept_b} @L{r.layer}: {r.cosine:+.3f} "
                     f"(null p97.5 {r.null_p97_5:.3f})")
        within = same[~is_role & ~same.concept_a.str.startswith("R_role")]
        log.info("  strongest safety-variable pairs:")
        for r in within.reindex(within.abs_cosine.sort_values(ascending=False).index).head(4).itertuples():
            log.info(f"    {r.concept_a} vs {r.concept_b} @L{r.layer}: {r.cosine:+.3f} "
                     f"(split-half floor {r.split_half_floor:.3f})")


# ---------------------------------------------------------------------------
# stage: projections (E1.3)
# ---------------------------------------------------------------------------
def stage_projections(ctx: Context) -> None:
    """Projections onto each direction, and correlations between them.

    The null for a projection correlation is the correlation between projections
    onto *random* directions, which is not zero on anisotropic activations.
    """
    cfg, log = ctx.cfg, ctx.log
    blob, idx, dirs, val = _load_cache(ctx)
    test = idx.split.eq("test").to_numpy()
    best = {c: int(val[val.concept == c].loc[val[val.concept == c].train_auc.idxmax(), "layer"])
            for c in dirs}

    proj = {}
    for c, li in best.items():
        d = dirs[c][li]
        proj[c] = d.project(blob[d.position][li][torch.tensor(test)]).numpy()
    pdf = pd.DataFrame(proj)
    for col in ("harmful", "role", "source", "label", "split"):
        pdf[col] = idx[col].to_numpy()[test]
    save_df(ctx.out / "projections.csv", pdf)

    g = torch.Generator().manual_seed(cfg.seed)
    d_model = next(iter(dirs.values()))[best[next(iter(best))]].vector.numel()
    ref = blob["t_post_inst"][best[next(iter(best))]][torch.tensor(test)].float()
    nulls = []
    for _ in range(200):
        v1, v2 = torch.randn(d_model, generator=g), torch.randn(d_model, generator=g)
        a, b = ref @ (v1 / v1.norm()), ref @ (v2 / v2.norm())
        nulls.append(abs(float(np.corrcoef(a.numpy(), b.numpy())[0, 1])))
    null_p95 = float(np.quantile(nulls, 0.95))

    rows, names = [], sorted(proj)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            r = float(np.corrcoef(proj[a], proj[b])[0, 1])
            rows.append({"concept_a": a, "concept_b": b, "pearson_r": r,
                         "abs_r": abs(r), "null_p95": null_p95,
                         "beyond_null": abs(r) > null_p95})
    save_df(ctx.out / "projection_correlations.csv", pd.DataFrame(rows))
    log.info(f"  projection-correlation null p95 = {null_p95:.3f}")
    for r in sorted(rows, key=lambda x: -x["abs_r"])[:5]:
        log.info(f"    r({r['concept_a']}, {r['concept_b']}) = {r['pearson_r']:+.3f} "
                 f"{'beyond null' if r['beyond_null'] else 'within null'}")


# ---------------------------------------------------------------------------
# stage: dimensionality (E1.4, spectral half)
# ---------------------------------------------------------------------------
def stage_dimensionality(ctx: Context) -> None:
    """Stratified directions, then the spectrum of THOSE.

    `r_eff` on raw activations measures the width of the representation space, and
    the between-class scatter is rank-1 with two classes — neither answers "single
    axis or subspace?". Estimating the direction separately within each stratum
    (role, source, bootstrap resample) and taking the spectrum of the stacked unit
    directions does: r_eff ~ 1 means every stratum recovers the same axis.
    """
    cfg, log = ctx.cfg, ctx.log
    blob, idx, dirs, val = _load_cache(ctx)
    train = idx.split.eq("train").to_numpy()
    rows = []
    for concept, (position, pos, neg, subset, _strat) in SPECS.items():
        if concept not in dirs:
            continue
        sub = val[val.concept == concept]
        li = int(sub.loc[sub.train_auc.idxmax(), "layer"])
        a = blob[position][li]
        base = train & _subset_mask(idx, subset)
        mp, mn = _side_mask(idx, pos), _side_mask(idx, neg)

        strat = []
        for kind, col in (("role", "role"), ("source", "source")):
            for lvl in sorted(idx[col].unique()):
                m = base & idx[col].eq(lvl).to_numpy()
                p, n = torch.tensor(m & mp), torch.tensor(m & mn)
                if int(p.sum()) >= 5 and int(n.sum()) >= 5:
                    v = a[p].float().mean(0) - a[n].float().mean(0)
                    if v.norm() > 1e-8:
                        strat.append(v / v.norm())
        g = torch.Generator().manual_seed(cfg.seed)
        ip, ineg = np.nonzero(base & mp)[0], np.nonzero(base & mn)[0]
        for _ in range(20):
            if len(ip) < 5 or len(ineg) < 5:
                break
            si = ip[torch.randint(len(ip), (len(ip),), generator=g).numpy()]
            sj = ineg[torch.randint(len(ineg), (len(ineg),), generator=g).numpy()]
            v = a[si].float().mean(0) - a[sj].float().mean(0)
            if v.norm() > 1e-8:
                strat.append(v / v.norm())
        if len(strat) < 3:
            continue
        M = torch.stack(strat)
        r_eff = extract.effective_rank(M)
        d0 = dirs[concept][li]
        resid = extract.separation_after_projecting_out(
            d0, a[torch.tensor(base & mp)], a[torch.tensor(base & mn)])
        rows.append({"model": ctx.model, "concept": concept, "layer": li,
                     "n_strata": len(strat), "r_eff_stratified": r_eff,
                     "auc_after_projecting_out_top1": resid["auc"],
                     "mean_pairwise_cos": float((M @ M.T).mean())})
        log.info(f"  {concept:<20} r_eff(stratified) {r_eff:.2f} over {len(strat)} strata "
                 f"| AUC after removing top-1: {resid['auc']:.3f}")
    save_df(ctx.out / "dimensionality.csv", pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# stage: harm_controls  (E1.1 — R_harm with refusal held constant)
# ---------------------------------------------------------------------------
def stage_harm_controls(ctx: Context) -> None:
    """Estimate `R_harm` with REFUSAL HELD CONSTANT, and ask what that changes.

    PLAN-EXTRACT specifies this and we had not done it:

        "Using matched harmful and benign prompts, *while controlling for refusal
         behavior where possible*, we estimate R_harm. Particular attention will be
         paid to examples in which a model recognizes harmfulness but nevertheless
         complies, allowing harmfulness representation to be separated from
         refusal behavior."

    Our `R_harm` is fitted pooled over refusal, so it may carry a refusal
    component — and if it does, the reported overlap between harm and control is
    partly an artifact of how harm was estimated rather than a fact about the
    model. This is the mirror image of the role-composition confound already
    corrected for `R_control`.

    Two refusal-controlled variants, one per row of the 2x2:

        R_harm_in_refused   harmful vs harmless, among REFUSED items
        R_harm_in_complied  harmful vs harmless, among COMPLIED items

    The second is literally the cell the plan names ("recognizes harmfulness but
    nevertheless complies"); the first is the one that is well populated on both
    models. Where a cell is too thin the variant is skipped rather than fitted on
    noise.

    The decisive comparisons, all against the split-half floor:
      * cos(refusal-controlled harm, pooled harm) — is the pooled fit contaminated?
      * cos(refusal-controlled harm @post, R_control) vs the pooled value — does
        holding refusal constant reduce the harm/control overlap?

    Both variants are fitted with the **role-balanced** estimator, unlike pooled
    `R_harm`. That asymmetry is deliberate and load-bearing: pooled `R_harm` is
    exempt from stratification because every instruction is rendered under all
    four roles, so its two sides carry identical role composition. Conditioning on
    the refusal label destroys that guarantee — refusal rate varies by role, which
    is the same fact that forces `R_control` to be balanced. Fitting these
    variants plainly would give them a role component, and because the headline
    compares them against a role-balanced `R_control`, that component would push
    the cosine DOWN and make the two variables look more separable than they are.
    """
    cfg, log = ctx.cfg, ctx.log
    blob, idx, dirs, val = _load_cache(ctx)
    train = idx.split.eq("train").to_numpy()
    test = ~train
    harmful = idx.harmful.astype(bool).to_numpy()

    VARIANTS = [("R_harm_in_refused", "refused"), ("R_harm_in_complied", "complied")]
    POSITIONS = ["t_inst", "t_post_inst"]
    MIN_SIDE = 25

    fitted, rows = {}, []
    for name, subset in VARIANTS:
        sub = _subset_mask(idx, subset)
        n_h, n_s = int((sub & train & harmful).sum()), int((sub & train & ~harmful).sum())
        if min(n_h, n_s) < MIN_SIDE:
            log.warning(f"  {name}: minority side {min(n_h, n_s)} < {MIN_SIDE} — skipped "
                        f"(harmful {n_h}, harmless {n_s} in train)")
            continue

        # These variants MUST be role-balanced, and pooled `R_harm` must not be.
        #
        # Pooled `R_harm` is exempt because every instruction is rendered under all
        # four roles, so its two sides carry identical role composition by
        # construction. Conditioning on the refusal label destroys precisely that
        # property: refusal rate varies by role, so P(role | harmful, refused)
        # differs from P(role | harmless, refused). Fitting these variants with the
        # plain estimator would therefore bake a role component into them — and
        # since the headline compares them against a role-balanced `R_control`, an
        # unbalanced fit would DEPRESS that cosine and overstate the separability.
        imb = {c: extract.composition_imbalance(
                   idx[c].to_numpy()[(sub & train & harmful)].tolist(),
                   idx[c].to_numpy()[(sub & train & ~harmful)].tolist())
               for c in ("role", "design") if c in idx.columns}
        for c, v in imb.items():
            log.info(f"  {name}: {c} composition TV={v['total_variation']:.3f}"
                     + ("  <-- balanced" if c == "role" else ""))

        # Length-only baseline, on the SAME conditioned population. Without it an
        # AUC here is uninterpretable: `R_harm_in_refused` contrasts ~870 harmful
        # items against ~50 over-refused harmless ones, and harmful instructions
        # are systematically longer (median 14 words vs 9). A direction that does
        # not beat this is measuring length, not harm — and at this imbalance a
        # perfect AUC is exactly what a length cue would produce.
        fit_sel = sub & train
        lb = extract.length_only_baseline(
            idx.n_tokens.to_numpy()[fit_sel].tolist(),
            harmful[fit_sel].tolist())
        log.info(f"  {name}: length-only AUC {lb['auc']:.3f} "
                 f"({lb['n_pos']} harmful vs {lb['n_neg']} harmless)")

        for pos in POSITIONS:
            key = f"{name}{'_at_post' if pos == 't_post_inst' else ''}"
            best = None
            for li in range(len(blob[pos])):
                a = blob[pos][li]
                tr_p = torch.tensor(sub & train & harmful)
                tr_n = torch.tensor(sub & train & ~harmful)
                te_p = torch.tensor(sub & test & harmful)
                te_n = torch.tensor(sub & test & ~harmful)
                if min(map(int, (tr_p.sum(), tr_n.sum(), te_p.sum(), te_n.sum()))) < 2:
                    continue
                d, _bal = extract.stratum_balanced_diff_of_means(
                    a[tr_p], a[tr_n],
                    idx["role"].to_numpy()[tr_p.numpy()].tolist(),
                    idx["role"].to_numpy()[tr_n.numpy()].tolist(),
                    key, li, "residual", pos, "harmful", "harmless")
                tr_auc = extract.separation(d, a[tr_p], a[tr_n])["auc"]
                if best is None or tr_auc > best[0]:
                    te_all = te_p | te_n
                    cb = extract.cluster_bootstrap_auc(
                        d.project(a[te_all]), te_p[te_all].tolist(),
                        idx.uid.to_numpy()[te_all.numpy()].tolist(), seed=cfg.seed)
                    sh = extract.split_half_stability(a[tr_p], a[tr_n], seed=cfg.seed)
                    best = (tr_auc, d, cb, sh, li)
            if best is None:
                continue
            _, d, cb, sh, li = best
            fitted[key] = d
            rows.append({"model": ctx.model, "concept": key, "position": pos, "layer": li,
                         "relative_depth": round(relative_depth(li, len(blob[pos])), 4),
                         "auc": cb["auc"], "auc_ci_low": cb["ci_low"], "auc_ci_high": cb["ci_high"],
                         "n_test_instructions": cb["n_clusters"],
                         "split_half_cos": sh["mean"],
                         "n_train_harmful": n_h, "n_train_harmless": n_s,
                         "role_tv_before_balancing": imb.get("role", {}).get("total_variation"),
                         "design_tv": imb.get("design", {}).get("total_variation"),
                         "role_balanced": True,
                         "length_only_auc": lb["auc"],
                         "beats_length_baseline": bool(cb["auc"] > lb["auc"])})
            log.info(f"  {key:<26} L{li:<3} AUC {cb['auc']:.3f} "
                     f"[{cb['ci_low']:.2f},{cb['ci_high']:.2f}]  split-half {sh['mean']:.3f}  "
                     f"(train n: {n_h} harmful / {n_s} harmless)")

    save_df(ctx.out / "harm_controls.csv", pd.DataFrame(rows))

    # ---- what does holding refusal constant change?
    comp = []
    def _cos(a, b):
        return float(a.vector @ b.vector)

    for key, d in fitted.items():
        pos = d.position
        pooled_name = "R_harm_at_post" if pos == "t_post_inst" else "R_harm"
        if pooled_name not in dirs:
            continue
        floor = float(val[val.concept == pooled_name].split_half_cos.max())
        p = dirs[pooled_name][d.layer]
        c_pool = abs(_cos(d, p))
        row = {"model": ctx.model, "variant": key, "position": pos, "layer": d.layer,
               "cos_with_pooled_harm": c_pool, "split_half_floor": floor,
               "pooled_harm_uncontaminated": bool(c_pool >= floor)}
        # and the headline: harm/control overlap, pooled vs refusal-controlled
        for ctrl in ("R_control", "R_control_harmless"):
            if ctrl in dirs and pos == "t_post_inst":
                cc = dirs[ctrl][d.layer]
                row[f"cos_{ctrl}_refctrl"] = abs(_cos(d, cc))
                row[f"cos_{ctrl}_pooled"] = abs(_cos(p, cc))
        comp.append(row)
        log.info(f"  {key:<26} cos with pooled {pooled_name}: {c_pool:.3f} "
                 f"(split-half floor {floor:.3f}) -> "
                 f"{'pooled fit is NOT refusal-contaminated' if c_pool >= floor else 'DIFFERENT DIRECTION — pooled fit carries refusal'}")
        for ctrl in ("R_control", "R_control_harmless"):
            if f"cos_{ctrl}_refctrl" in row:
                log.info(f"    overlap with {ctrl}: pooled {row[f'cos_{ctrl}_pooled']:.3f} "
                         f"-> refusal-controlled {row[f'cos_{ctrl}_refctrl']:.3f}")
    save_df(ctx.out / "harm_controls_geometry.csv", pd.DataFrame(comp))
    save_torch(ctx.out / "harm_control_directions.pt", fitted)


# ---------------------------------------------------------------------------
# stage: dimensionality_behavioural (E1.4b)
# ---------------------------------------------------------------------------
def _behavioural_null(ctx: Context, alpha: float) -> float:
    """The magnitude-matched random direction's |Δrefusal| at this α, from E1.6.

    Reused rather than recomputed: it is the same quantity, measured on the same
    probe set with the same readout, and recomputing it would cost another set of
    generation runs to arrive at the same number. Returns 0.0 if the gate has not
    run, which makes the guard inert rather than silently wrong.
    """
    import json
    best = 0.0
    for p in sorted(ctx.cfg.dir(NAME, ctx.model).glob("causal_gate*.json")):
        try:
            g = json.loads(p.read_text())
        except Exception:
            continue
        for v in g.get("per_bound", {}).values():
            band = v.get("G3_behavioural_dissociation", {}).get(
                "behavioural_null_p95_by_alpha", {})
            for k, val in band.items():
                # keys are stringified tuples like "(1.0,)"
                digits = "".join(c for c in str(k) if c.isdigit() or c == ".")
                try:
                    if abs(float(digits) - alpha) < 1e-9:
                        best = max(best, float(val))
                except ValueError:
                    continue
    return best


def stage_dimensionality_behavioural(ctx: Context) -> None:
    """The `k` the paper reports: the smallest subspace that reproduces the
    intervention effect of the full direction.

    PLAN-EXTRACT refuses to assume refusal is one-dimensional and asks for "the
    smallest dimensionality required to explain the observed behavioural
    interventions". That is an *intervention* criterion, so it cannot be answered
    by a spectrum: E1.4's spectral half says how many axes the estimator wanders
    over across strata, which is a statement about estimation variability, not
    about how many the model uses.

    Construction. The stratified unit directions (per role, per source) are stacked
    and given an SVD; `U_k` spans the top-`k` principal subspace of that stack. The
    rank-`k` intervention steers along `P_k v_raw`, the projection of the raw
    difference-in-means onto that subspace, so `k = full` recovers the ordinary
    intervention exactly and the sequence is nested and monotone in what it can
    express. `k` is then the smallest value reaching 90% of the full effect.

    The effect is measured behaviourally — change in refusal rate under the
    intervention, labelled with the published rule — because that is what
    "explains the behavioural interventions" means. The representational shift on
    the concept's own readout is reported alongside.
    """
    from core.interventions import generate_steered
    from core.model_io import load_model, resolve_device
    cfg, log, slug = ctx.cfg, ctx.log, ctx.model

    blob, idx, dirs, val = _load_cache(ctx)
    train = idx.split.eq("train").to_numpy()
    device = resolve_device()
    model, tok = load_model(ctx.spec().model_id, device, logger=log)
    rendered = _render_items(cfg, tok, ctx.model)
    if not (list(rendered.uid) == list(idx.uid) and list(rendered.role) == list(idx.role)):
        raise ValueError("re-rendered corpus does not align with the cached activation index")

    harmful = idx.harmful.astype(bool).to_numpy()
    test = (idx.split.eq("test").to_numpy() & idx.role.eq("user").to_numpy()
            & idx.design.eq("fixed_slot").to_numpy())
    p_i, n_i = np.nonzero(test & harmful)[0], np.nonzero(test & ~harmful)[0]
    n_side = min(len(p_i), len(n_i), 8 if cfg.fast_dev else 50)
    sel = np.concatenate([p_i[:n_side], n_i[:n_side]])
    texts = [rendered.text.iloc[i] for i in sel]
    log.info(f"[{slug}] E1.4b probe set: {len(texts)} items")

    KS = [1, 2, 3, 4, 6, 8]
    ALPHA = 1.0                                  # the published operating point
    rows = []

    for concept, (position, pos, neg, subset, _strat) in SPECS.items():
        if concept not in dirs or concept.endswith(("_unbalanced", "_preguard", "_user")):
            continue
        sub = val[val.concept == concept]
        li = int(sub.loc[sub.train_auc.idxmax(), "layer"])
        a = blob[position][li]
        base_m = train & _subset_mask(idx, subset)
        mp, mn = _side_mask(idx, pos), _side_mask(idx, neg)

        U = _stratified_subspace(a, idx, base_m, mp, mn)
        if U is None:
            log.warning(f"  {concept}: too few strata for a subspace — skipped")
            continue
        full = dirs[concept][li]
        v_raw = full.raw().float()                                # raw diff-of-means

        base_gen = generate_steered(model, tok, texts, None, li, 0.0, device,
                                    cfg.batch_size, cfg.refusal_max_new_tokens)
        base_ref = float(np.mean([refusal.has_refusal_marker(g) for g in base_gen]))

        eff = {}
        for k in [k for k in KS if k <= U.shape[0]] + ["full"]:
            if k == "full":
                vk = v_raw
            else:
                Uk = U[:k]                                        # [k, d]
                vk = (Uk.T @ (Uk @ v_raw))                        # P_k v_raw
            if float(vk.norm()) < 1e-8:
                continue
            d_k = extract.Direction(
                vector=vk / vk.norm(), concept=f"{concept}_k{k}", layer=li,
                site="residual", position=position, positive_class=pos,
                negative_class=neg, n_positive=full.n_positive,
                n_negative=full.n_negative, raw_norm=float(vk.norm()))
            gen = generate_steered(model, tok, texts, d_k, li, ALPHA, device,
                                   cfg.batch_size, cfg.refusal_max_new_tokens)
            ref = float(np.mean([refusal.has_refusal_marker(g) for g in gen]))
            eff[k] = ref - base_ref
            rows.append({"model": slug, "concept": concept, "layer": li, "k": k,
                         # U's rows ARE the stratified axes: `_stratified_subspace`
                         # stacks one unit direction per stratum and returns the
                         # row-space basis, so this is the count the inline `strat`
                         # list used to give before that refactor.
                         "n_strata": int(U.shape[0]), "alpha": ALPHA,
                         "baseline_refusal": base_ref, "refusal": ref,
                         "d_refusal": ref - base_ref,
                         "retained_norm": float(vk.norm()) / float(v_raw.norm())})

        # A ratio to the full effect is only interpretable if the full effect is
        # itself distinguishable from noise. Below the behavioural null band, the
        # denominator is noise and the ratios wander above 1 — so `k` is reported
        # as UNDEFINED rather than as 1. The band is the magnitude-matched random
        # direction's own |Δrefusal| at this α, taken from the gate's null.
        beh_null = _behavioural_null(ctx, ALPHA)
        if "full" not in eff or abs(eff["full"]) <= beh_null:
            log.warning(f"  {concept:<22} full Δrefusal {eff.get('full', float('nan')):+.3f} "
                        f"<= behavioural null {beh_null:.3f} at α={ALPHA} — k UNDEFINED "
                        f"(the denominator is noise)")
            for r in rows:
                if r["concept"] == concept:
                    r["k_star"] = None
                    r["below_behavioural_null"] = True
            continue
        ratios = {k: eff[k] / eff["full"] for k in eff if k != "full"}
        k_star = next((k for k in sorted(ratios) if ratios[k] >= 0.90), None)
        for r in rows:
            if r["concept"] == concept and r["k"] != "full":
                r["ratio_to_full"] = ratios.get(r["k"])
                r["k_star"] = k_star
                r["below_behavioural_null"] = False
        log.info(f"  {concept:<22} full Δrefusal {eff['full']:+.3f} (null {beh_null:.3f}) | "
                 + " ".join(f"k={k}:{ratios[k]:.2f}" for k in sorted(ratios))
                 + f" | k* = {k_star}")

    df = pd.DataFrame(rows)
    save_df(ctx.out / "dimensionality_behavioural.csv", df)
    if len(df):
        ks = df.dropna(subset=["k_star"]).groupby("concept").k_star.first().to_dict()
        save_json(ctx.out / "behavioural_k.json",
                  {"criterion": "smallest k whose rank-k subspace reproduces >=90% of the "
                                "full direction's change in refusal rate at alpha=1",
                   "k_star": {c: (int(v) if pd.notna(v) else None) for c, v in ks.items()}})
        log.info(f"[{slug}] behavioural k*: {ks}")


# ---------------------------------------------------------------------------
# stage: emergence (E1.5)
# ---------------------------------------------------------------------------
def stage_emergence(ctx: Context) -> None:
    """Separation vs relative depth, plus persistence of a layer-l direction.

    Relative depth, not layer index: Qwen2.5-7B has 28 layers and Qwen3.5-9B 32,
    so absolute indices are not comparable across models.
    """
    log = ctx.log
    d = ctx.cfg.dir(NAME, ctx.model)
    val = pd.read_csv(d / "direction_validation.csv")
    probe = pd.read_csv(d / "role_probe.csv")
    curves = val[["concept", "layer", "relative_depth", "train_auc", "auc",
                  "auc_ci_low", "auc_ci_high", "length_only_auc", "random_auc"]].copy()
    curves["model"] = ctx.model
    r = probe[["layer", "relative_depth", "train_accuracy", "accuracy",
               "acc_ci_low", "acc_ci_high"]].copy()
    r["concept"], r["model"] = "R_role_probe", ctx.model
    r = r.rename(columns={"train_accuracy": "train_auc", "accuracy": "auc",
                          "acc_ci_low": "auc_ci_low", "acc_ci_high": "auc_ci_high"})
    out = pd.concat([curves, r], ignore_index=True)

    # The probe's chance level is 1/n_roles and it RECORDS it. Reading it here
    # rather than assuming 0.25 matters as soon as a model's chat template cannot
    # express all four role classes: `tool` exists only in the Qwen and Llama
    # templates, so a three-role model has chance 1/3, and an onset measured
    # against 0.25 would be computed from a baseline that is too low — inflating
    # every role onset on exactly the cross-family models added to answer the
    # generalisation question.
    role_chance = float(probe.chance.iloc[0]) if "chance" in probe.columns else 0.25

    # Pre-registered emergence rule (EXPERIMENTS.md > E1.5): the depth at which
    # separation FIRST reaches a fraction of its within-model peak — not the
    # argmax. The peak says where a concept is most decodable; onset says where it
    # becomes available, and only the latter speaks to emergence. Reported at
    # 80/90/95% so no conclusion rests on one arbitrary fraction.
    #
    # The peak itself is selected on TRAIN and reported on test.
    rows = []
    log.info("  emergence by concept — onset depth (peak selected on train):")
    for c in out.concept.unique():
        s = out[out.concept == c].sort_values("layer")
        b = s.loc[s.train_auc.idxmax()]
        base = role_chance if c == "R_role_probe" else 0.5   # chance for that readout

        def onset(frac):
            hit = s[s.auc >= base + frac * (b.auc - base)]
            return float(hit.relative_depth.min()) if len(hit) else float("nan")

        # Bootstrap CI on the onset DEPTH itself. Without it there is no way to
        # tell a sharply-located onset from one whose AUC-vs-depth curve is flat
        # near its peak — and an independent reproduction showed exactly that
        # failure mode: one concept's onset moved by 0.42 between runs while every
        # other moved 0.000. The interval is what distinguishes them in advance.
        #
        # Resampled from each layer's own bootstrap interval, treating the layer
        # curve as the object of uncertainty: draw a perturbed curve, recompute
        # peak and onset, repeat.
        rng = np.random.default_rng(ctx.cfg.seed)
        se = ((s.auc_ci_high.values - s.auc_ci_low.values) / (2 * 1.96)).clip(min=1e-9)
        depths, aucs = s.relative_depth.values, s.auc.values
        boot = []
        for _ in range(500):
            draw = aucs + rng.normal(0.0, se)
            pk = draw.max()
            hit = np.nonzero(draw >= base + 0.90 * (pk - base))[0]
            if len(hit):
                boot.append(float(depths[hit.min()]))
        onset90_lo, onset90_hi = ((float(np.percentile(boot, 2.5)),
                                   float(np.percentile(boot, 97.5)))
                                  if boot else (float("nan"), float("nan")))

        rec = {"model": ctx.model, "concept": c, "peak_auc": float(b.auc),
               "peak_depth": float(b.relative_depth), "peak_layer": int(b.layer),
               "onset_80": onset(0.80), "onset_90": onset(0.90), "onset_95": onset(0.95),
               "onset_90_ci_low": onset90_lo, "onset_90_ci_high": onset90_hi,
               "onset_90_ci_width": onset90_hi - onset90_lo,
               "layer0_auc": (float(s[s.layer == 0].auc.iloc[0])
                              if (s.layer == 0).any() else float("nan"))}
        rows.append(rec)
        log.info(f"    {c:<22} peak {rec['peak_auc']:.3f} @ d={rec['peak_depth']:.2f} "
                 f"| onset90 d={rec['onset_90']:.2f} "
                 f"[{onset90_lo:.2f},{onset90_hi:.2f}] (w={rec['onset_90_ci_width']:.2f}) "
                 f"| 80/95%: {rec['onset_80']:.2f}/{rec['onset_95']:.2f}")

    save_df(ctx.out / "emergence_curves.csv", out)
    save_df(ctx.out / "emergence_summary.csv", pd.DataFrame(rows))



# ---------------------------------------------------------------------------
# stage: style  (E1.7 Level 1 — metadata versus style, and prompt-category variation)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# stage: geometry_subspace  (E1.2 pass 2)
# ---------------------------------------------------------------------------
def stage_geometry_subspace(ctx: Context) -> None:
    """E1.2 pass 2 — subspace geometry at the `k` that E1.4b measured.

    PLAN-GEOM asks for principal angles, projection measures and canonical
    correlations, and calls subspace measures "the primary analysis", with cosine
    as the one-dimensional special case. EXPERIMENTS.md breaks the circularity —
    subspace measures need `k`, `k` comes from E1.4 — by running E1.2 in two
    passes, and rules:

        If E1.4 returns k = 1 for every concept, pass 2 is degenerate and pass 1
        is the analysis — state that outcome explicitly rather than reporting
        principal angles between one-dimensional subspaces.

    This stage implements that rule, **including the degenerate branch**, which
    is why it runs unconditionally rather than only when some k > 1: "pass 2 was
    degenerate" is a finding the paper has to state, and it cannot be stated by a
    stage that silently declines to run.

    Three named quantities, one spectrum. Between subspaces, the canonical
    correlations *are* the cosines of the principal angles, and the projection
    metric is their normalised sum of squares — see `extract.principal_angles`.
    Reporting them as three independent measurements would triple-count one fact,
    so the spectrum is computed once and the summaries derived from it. The
    genuinely independent measurement is the **data-weighted** canonical
    correlation, which asks whether the activations that actually occur live in
    the part of the two subspaces that agrees.

    Nulls. A rank-1 cosine band is the wrong reference here: two random
    `k`-dimensional subspaces overlap more than two random lines, by an amount
    that grows with `k`, so the pass-1 band would make any `k > 1` overlap look
    significant by construction. Each pair gets a null matched to its own
    `(k_a, k_b)`.
    """
    import json
    cfg, log, slug = ctx.cfg, ctx.log, ctx.model
    blob, idx, dirs, val = _load_cache(ctx)
    train = idx.split.eq("train").to_numpy()

    kpath = ctx.out / "behavioural_k.json"
    if not kpath.exists():
        raise SystemExit(
            f"[{slug}] E1.2 pass 2 requires behavioural_k.json from E1.4b; run "
            f"`--only dim_behavioural` first. Pass 2 is defined at the k that "
            f"E1.4b measures, so guessing k here would defeat its purpose.")
    k_star = {c: v for c, v in json.loads(kpath.read_text())["k_star"].items()
              if v is not None}
    if not k_star:
        raise SystemExit(f"[{slug}] behavioural_k.json contains no usable k")

    # ---- build each concept's rank-k subspace, by the SAME construction E1.4b
    # used to measure k. A different basis here would mean the reported k does
    # not describe the subspace being compared.
    bases, meta = {}, []
    for concept, k in sorted(k_star.items()):
        if concept not in dirs:
            continue
        position, pos, neg, subset, _strat = SPECS[concept]
        sub = val[val.concept == concept]
        li = int(sub.loc[sub.train_auc.idxmax(), "layer"])
        a = blob[position][li]
        U = _stratified_subspace(a, idx, train & _subset_mask(idx, subset),
                                 _side_mask(idx, pos), _side_mask(idx, neg))
        if U is None:
            log.warning(f"  {concept}: too few strata for a subspace — skipped")
            continue
        kk = min(int(k), U.shape[0])
        bases[concept] = (U[:kk], position, li)
        meta.append({"model": slug, "concept": concept, "k_star": int(k),
                     "k_used": kk, "position": position, "layer": li,
                     "n_strata_axes": int(U.shape[0])})
        log.info(f"  {concept:<22} k*={k} basis {kk}x{U.shape[1]} @L{li} ({position})")

    degenerate = bool(bases) and all(v[0].shape[0] == 1 for v in bases.values())
    d_model = next(iter(bases.values()))[0].shape[1] if bases else 0

    # ---- pairwise, WITHIN a position: the residual basis differs by layer and by
    # position, so a cross-position angle is not interpretable.
    rows = []
    names = sorted(bases)
    for i, A in enumerate(names):
        for B in names[i + 1:]:
            Ua, pa_, la = bases[A]
            Ub, pb_, lb = bases[B]
            if pa_ != pb_:
                continue
            pa = extract.principal_angles(Ua, Ub)
            null = extract.random_subspace_null(d_model, pa["k_a"], pa["k_b"],
                                                n_draws=1000, seed=cfg.seed)
            # data-weighted CCA on held-out activations at A's layer
            test = ~train
            Xa = (blob[pa_][la][torch.tensor(test)].float() @ Ua.T)
            Xb = (blob[pb_][la][torch.tensor(test)].float() @ Ub.T)
            cca = extract.data_canonical_correlations(Xa, Xb)
            rows.append({
                "model": slug, "concept_a": A, "concept_b": B, "position": pa_,
                "layer_a": la, "layer_b": lb, "same_layer": la == lb,
                "k_a": pa["k_a"], "k_b": pa["k_b"],
                "projection_metric": pa["projection_metric"],
                "null_p97_5": null["p97.5"], "null_mean": null["mean"],
                "beyond_null": pa["projection_metric"] > null["p97.5"],
                "smallest_angle_deg": pa["smallest_principal_angle_deg"],
                "largest_angle_deg": pa["largest_principal_angle_deg"],
                "cos_principal_angles": ";".join(f"{v:.4f}" for v in pa["cos_principal_angles"]),
                "data_cca": ";".join(f"{v:.4f}" for v in cca),
                "data_cca_max": max(cca) if cca else float("nan"),
                "degenerate_rank_one": pa["degenerate_rank_one"],
            })
            log.info(f"  {A} vs {B}: proj {pa['projection_metric']:.3f} "
                     f"(null p97.5 {null['p97.5']:.3f}) "
                     f"angles {[round(v,1) for v in pa['principal_angles_deg']]} deg"
                     + (f"  data-CCA max {max(cca):.3f}" if cca else ""))

    save_df(ctx.out / "geometry_subspace.csv", pd.DataFrame(rows))
    save_json(ctx.out / "geometry_subspace_summary.json", {
        "model": slug,
        "k_star": k_star,
        "bases": meta,
        "pass2_degenerate": degenerate,
        "interpretation": (
            "Every concept has k = 1, so principal angles reduce exactly to the "
            "pass-1 cosine and pass 2 adds nothing: PASS 1 IS THE ANALYSIS."
            if degenerate else
            "At least one concept has k > 1, so subspace measures are not "
            "reducible to the pass-1 cosine and are the primary geometry."),
        "note": ("Between subspaces the canonical correlations ARE the cosines of "
                 "the principal angles and the projection metric is their "
                 "normalised sum of squares; the three named quantities are one "
                 "spectrum. `data_cca` is the independent, data-weighted measure."),
    })
    log.info(f"[{slug}] E1.2 pass 2: {'DEGENERATE (all k=1)' if degenerate else 'subspace measures apply'}")


def stage_style(ctx: Context) -> None:
    """Does the role direction track the role TAG, or the linguistic style?

    PLAN-EXTRACT asks whether "explicit role metadata and linguistic style produce
    compatible or conflicting representations", which connects to the role paper's
    central finding that **style dominates tags**. Level 2 needs the same content
    rewritten in each style and is a separate corpus build; Level 1 is available
    now because `source` was recorded per instruction at E1.0 and the sources
    differ natively in style — imperative harmful requests, task instructions,
    questions.

    Method: refit the role contrast (and `R_harm`) **within each source**, at the
    train-selected layer, then compare the per-source directions to each other.
    The reference is the **split-half floor** for the same contrast — two estimates
    of one direction from disjoint halves. A per-source pair at the floor means the
    direction is the same regardless of style; well below it means style is
    changing the direction, not just its strength.

    Also PLAN-GEOM's "variation across prompt categories": the same comparison for
    `R_harm` across the recorded `category` factor.
    """
    cfg, log = ctx.cfg, ctx.log
    blob, idx, dirs, val = _load_cache(ctx)
    train = idx.split.eq("train").to_numpy()
    rows = []
    g = torch.Generator().manual_seed(cfg.seed)

    def _splithalf_floor(a, pmask, nmask, reps: int = 30) -> float:
        """Noise floor for THIS contrast on THIS subset: the same direction,
        estimated twice on disjoint halves. Any style comparison must be read
        against it, since two estimates of one direction do not reach 1.0."""
        pi, ni = np.nonzero(pmask)[0], np.nonzero(nmask)[0]
        if len(pi) < 8 or len(ni) < 8:
            return float("nan")
        out = []
        for _ in range(reps):
            pp = pi[torch.randperm(len(pi), generator=g).numpy()]
            nn = ni[torch.randperm(len(ni), generator=g).numpy()]
            ph, nh = len(pp) // 2, len(nn) // 2
            va = a[pp[:ph]].float().mean(0) - a[nn[:nh]].float().mean(0)
            vb = a[pp[ph:]].float().mean(0) - a[nn[nh:]].float().mean(0)
            if va.norm() > 1e-8 and vb.norm() > 1e-8:
                out.append(float((va / va.norm()) @ (vb / vb.norm())))
        return float(np.median(out)) if out else float("nan")

    def _compare(name: str, layer: int, a, groups: Dict[str, tuple], factor: str,
                 min_side: int = 15) -> None:
        vecs, floors = {}, []
        for lvl, (pm, nm) in groups.items():
            if int(pm.sum()) < min_side or int(nm.sum()) < min_side:
                continue
            v = a[torch.tensor(pm)].float().mean(0) - a[torch.tensor(nm)].float().mean(0)
            if v.norm() > 1e-8:
                vecs[lvl] = v / v.norm()
                f = _splithalf_floor(a, pm, nm)
                if np.isfinite(f):
                    floors.append(f)
        if len(vecs) < 2:
            log.warning(f"  {name:<26} across {factor}: fewer than 2 estimable levels — skipped")
            return
        floor = float(np.median(floors)) if floors else float("nan")
        keys, cs = sorted(vecs), []
        for i, x in enumerate(keys):
            for y in keys[i + 1:]:
                c = float(vecs[x] @ vecs[y])
                cs.append(c)
                rows.append({"model": ctx.model, "concept": name, "factor": factor,
                             "level_a": x, "level_b": y, "layer": layer, "cosine": c,
                             "split_half_floor": floor,
                             "at_floor": bool(np.isfinite(floor) and c >= floor)})
        verdict = ("style-INVARIANT" if np.isfinite(floor) and min(cs) >= floor
                   else "style-DEPENDENT" if np.isfinite(floor) else "no floor")
        log.info(f"  {name:<26} across {factor:<8} ({len(vecs)} levels): cos "
                 f"min {min(cs):.3f} med {float(np.median(cs)):.3f} max {max(cs):.3f} "
                 f"| within-level split-half floor {floor:.3f} -> {verdict}")

    log.info(f"[{ctx.model}] E1.7 Level 1 — directions refitted within each native style")

    # --- Role: tool-vs-user, refitted within each source.
    probe_csv = pd.read_csv(cfg.dir(NAME, ctx.model) / "role_probe.csv")
    li_role = int(probe_csv.loc[probe_csv.train_accuracy.idxmax(), "layer"])
    a_role = blob["t_post_inst"][li_role]
    groups = {}
    for lvl in sorted(idx["source"].dropna().unique()):
        m = train & idx["source"].eq(lvl).to_numpy()
        groups[str(lvl)] = (m & idx.role.eq("tool").to_numpy(),
                            m & idx.role.eq("user").to_numpy())
    _compare("R_role_tool_vs_user", li_role, a_role, groups, "source")

    # --- Harm: CANNOT be refit within a source — the sources are confounded with
    # the harm label (AdvBench is entirely harmful, Alpaca entirely harmless), so
    # no single source contains both sides. The style comparison is therefore over
    # SOURCE PAIRS: one harmful source against one harmless source, which varies
    # the style of both sides while holding the contrast fixed.
    sub = val[val.concept == "R_harm"]
    li_h = int(sub.loc[sub.train_auc.idxmax(), "layer"])
    a_h = blob["t_inst"][li_h]
    harm_src = sorted(idx[idx.harmful.astype(bool)].source.unique())
    safe_src = sorted(idx[~idx.harmful.astype(bool)].source.unique())
    pair_groups = {}
    for hs in harm_src:
        for ss in safe_src:
            pair_groups[f"{hs}|{ss}"] = (
                train & idx.source.eq(hs).to_numpy(),
                train & idx.source.eq(ss).to_numpy())
    _compare("R_harm", li_h, a_h, pair_groups, "source_pair")

    # --- E1.1 checklist: the Zhao replication, as a named artifact.
    #
    # O-11 downgraded this on the grounds that the experiment already runs here —
    # `advbench|alpaca` is a canonical Zhao-style setup and is one of the pairs
    # refitted above. What was missing is that `style_vs_metadata.csv` records
    # only the pairwise COSINES between refits, not each refit's separation, so
    # the one number a like-for-like comparison needs was never emitted.
    #
    # Swept over every layer because Zhao report layer-wise; the write-up picks
    # the comparison point. `published_auc` is deliberately left empty rather than
    # filled from memory — it must be read off their paper.
    ZHAO_PAIR = ("advbench", "alpaca")
    zrows = []
    if ZHAO_PAIR[0] in harm_src and ZHAO_PAIR[1] in safe_src:
        hp = idx.source.eq(ZHAO_PAIR[0]).to_numpy()
        sp = idx.source.eq(ZHAO_PAIR[1]).to_numpy()
        for li in sorted(blob["t_inst"]):
            a = blob["t_inst"][li]
            tr_p, tr_n = torch.tensor(hp & train), torch.tensor(sp & train)
            te_p, te_n = torch.tensor(hp & ~train), torch.tensor(sp & ~train)
            if min(map(int, (tr_p.sum(), tr_n.sum(), te_p.sum(), te_n.sum()))) < 5:
                continue
            d = extract.diff_of_means(a[tr_p], a[tr_n], "R_harm_zhao", li,
                                      "residual", "t_inst", "harmful", "harmless")
            te_all = te_p | te_n
            cb = extract.cluster_bootstrap_auc(
                d.project(a[te_all]), te_p[te_all].tolist(),
                idx.uid.to_numpy()[te_all.numpy()].tolist(), seed=cfg.seed)
            lb = extract.length_only_baseline(
                idx.n_tokens.to_numpy()[(hp | sp)].tolist(), hp[(hp | sp)].tolist())
            sh = extract.split_half_stability(a[tr_p], a[tr_n], seed=cfg.seed)
            zrows.append({
                "model": ctx.model, "contrast": f"{ZHAO_PAIR[0]} vs {ZHAO_PAIR[1]}",
                "position": "t_inst", "layer": li,
                "relative_depth": round(relative_depth(li, len(blob["t_inst"])), 4),
                "auc": cb["auc"], "auc_ci_low": cb["ci_low"], "auc_ci_high": cb["ci_high"],
                "n_test_instructions": cb["n_clusters"],
                "length_only_auc": lb["auc"], "split_half_cos": sh["mean"],
                "n_train_harmful": int(tr_p.sum()), "n_train_harmless": int(tr_n.sum()),
                "published_auc": "",          # fill from Zhao et al. 2507.11878
                "published_source": "Zhao et al., arXiv 2507.11878",
            })
        save_df(ctx.out / "zhao_replication.csv", pd.DataFrame(zrows))
        if zrows:
            best = max(zrows, key=lambda r: r["auc"])
            log.info(f"  Zhao replication ({ZHAO_PAIR[0]} vs {ZHAO_PAIR[1]}): "
                     f"peak AUC {best['auc']:.3f} at L{best['layer']} "
                     f"(depth {best['relative_depth']:.2f}), "
                     f"length-only {best['length_only_auc']:.3f}")
    else:
        log.warning(f"  Zhao replication skipped — {ZHAO_PAIR} not both present "
                    f"(harmful sources {harm_src}, harmless {safe_src})")

    # --- Harm across prompt CATEGORY (PLAN-GEOM), same pairing logic.
    cat_groups = {}
    for c in sorted(idx[idx.harmful.astype(bool)].category.dropna().unique()):
        m = train & idx.category.eq(c).to_numpy() & idx.harmful.astype(bool).to_numpy()
        if m.sum() >= 15:
            cat_groups[str(c)] = (m, train & ~idx.harmful.astype(bool).to_numpy())
    _compare("R_harm", li_h, a_h, cat_groups, "category")

    df = pd.DataFrame(rows)
    save_df(ctx.out / "style_vs_metadata.csv", df)
    save_json(ctx.out / "style_summary.json", {
        "level": 1,
        "note": "source is the native-style proxy recorded at E1.0; Level 2 (same "
                "content rewritten per style, crossed with the tag) needs its own corpus",
        "n_comparisons": int(len(df))})


# ---------------------------------------------------------------------------
# stage: fidelity  (E1.1 criteria that the first consolidation dropped)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# stage: style_level2  (E1.7 Level 2 — tag versus register, crossed)
# ---------------------------------------------------------------------------
def stage_style_level2(ctx: Context) -> None:
    """Does `R_role` track the role TAG or the linguistic REGISTER?

    PLAN-EXTRACT asks whether *"explicit role metadata and linguistic style
    produce compatible or conflicting representations."* Level 1 can only measure
    how the two co-vary in the corpus as it happens to be, because there the tag
    and the register of the content are not independent. Level 2 crosses them on
    the frozen E1.7 corpus: the same request in three registers, under every tag.

    Two directions per arm, fitted at the same position and layer so they are
    comparable at all:

      TAG       role `a` vs role `b`, register held constant (pooled over
                registers, which is balanced by construction here since every
                base appears in every register)
      REGISTER  register `x` vs register `y`, tag held constant

    The decisive quantity is `cos(TAG, REGISTER)` read against the split-half
    floor of each. Near zero: `R_role` is metadata, indifferent to how the text
    sounds. Near the floor: the two are the same direction and the role variable
    is a style detector.

    Two arms, and the difference between them is the point. `generated` is the
    real test — register redistributed through the text. `template` is a control
    in which register lives in a fixed framing phrase, so it is trivially
    separable and upper-bounds how detectable register can be. Orthogonality in
    the template arm is the stronger statement of the two.
    """
    from core.capture import ActivationCapture
    from core.model_io import load_model, resolve_device
    from core.positions import render

    cfg, log, slug = ctx.cfg, ctx.log, ctx.model
    sdir = cfg.results_root / "e1_7_style"
    if not (sdir / "style_corpus.jsonl").exists():
        raise SystemExit(
            f"[{slug}] E1.7 Level 2 needs the frozen style corpus; run "
            f"`experiments/e1_7_style_corpus.py` first. It is a separate build "
            f"because it requires generation and E1.0 is deliberately model-free.")
    items = pools.load_style_corpus(sdir)
    log.info(f"[{slug}] style corpus: {len(items)} items "
             f"({len({i.arm for i in items})} arms, {len({i.register for i in items})} registers)")

    device = resolve_device()
    model, tok = load_model(ctx.spec().model_id, device, logger=log)
    n_layers = model_meta.num_layers(model.config)

    recs, rendered = [], []
    for it in items:
        for role in cfg.roles_for(slug):
            R = render(tok, it.text, role, "fixed_slot")
            rendered.append(R)
            recs.append({"uid": it.uid, "base_uid": it.base_uid, "arm": it.arm,
                         "register": it.register, "role": role, "split": it.split})
    idx = pd.DataFrame(recs)
    log.info(f"[{slug}] rendering {len(rendered)} items "
             f"({len(items)} style items x {len(cfg.roles_for(slug))} role tags)")

    cap = ActivationCapture(model, site="residual")
    acts = cap.at_positions(tok, rendered, ("t_inst", "t_post_inst"),
                            list(range(n_layers)), cfg.batch_size, device,
                            log_every=25, logger=log)

    train = idx.split.eq("train").to_numpy()
    out = []
    for arm in sorted(idx.arm.unique()):
        in_arm = train & idx.arm.eq(arm).to_numpy()
        for position in ("t_inst", "t_post_inst"):
            # --- TAG directions, register pooled (balanced by construction)
            tag_dirs = {}
            for a_r, b_r in ROLE_CONTRASTS:
                ma = in_arm & idx.role.eq(a_r).to_numpy()
                mb = in_arm & idx.role.eq(b_r).to_numpy()
                if ma.sum() < 10 or mb.sum() < 10:
                    continue
                tag_dirs[f"{a_r}_vs_{b_r}"] = (ma, mb)
            # --- REGISTER directions, tag pooled (also balanced by construction)
            regs = sorted(idx.register.unique())
            reg_dirs = {}
            for i, x in enumerate(regs):
                for y in regs[i + 1:]:
                    mx = in_arm & idx.register.eq(x).to_numpy()
                    my = in_arm & idx.register.eq(y).to_numpy()
                    if mx.sum() < 10 or my.sum() < 10:
                        continue
                    # The register contrast must be PAIRED within base, or it is
                    # partly a contrast between different requests — the confound
                    # the whole design exists to remove. The corpus build already
                    # guarantees it by keeping only bases complete in every
                    # register, which makes the two sides carry identical base
                    # sets and so makes the pooled difference algebraically equal
                    # to the mean within-base difference. Asserted rather than
                    # assumed, because it is the property everything rests on and
                    # it lives in a different file.
                    bx = set(idx.base_uid.to_numpy()[mx])
                    by = set(idx.base_uid.to_numpy()[my])
                    if bx != by:
                        log.warning(
                            f"  [{arm}/{position}] {x} vs {y}: base sets differ "
                            f"({len(bx - by)} only-{x}, {len(by - bx)} only-{y}) — "
                            f"contrast is NOT paired; skipped")
                        continue
                    reg_dirs[f"{x}_vs_{y}"] = (mx, my)

            for li in range(n_layers):
                A = acts[position][li]

                def fit(mp, mn, name):
                    d = extract.diff_of_means(A[torch.tensor(mp)], A[torch.tensor(mn)],
                                              name, li, "residual", position, "p", "n")
                    sh = extract.split_half_stability(A[torch.tensor(mp)],
                                                      A[torch.tensor(mn)], seed=cfg.seed)
                    return d, sh["mean"]

                fitted_tag = {k: fit(*v, f"tag_{k}") for k, v in tag_dirs.items()}
                fitted_reg = {k: fit(*v, f"reg_{k}") for k, v in reg_dirs.items()}
                for tk, (td, tf) in fitted_tag.items():
                    for rk, (rd, rf) in fitted_reg.items():
                        out.append({
                            "model": slug, "arm": arm, "position": position, "layer": li,
                            "relative_depth": round(relative_depth(li, n_layers), 4),
                            "tag_contrast": tk, "register_contrast": rk,
                            "cos_tag_register": abs(float(td.vector @ rd.vector)),
                            "tag_split_half": tf, "register_split_half": rf,
                            "floor": min(tf, rf),
                            # Below the floor means the two directions differ by
                            # more than either differs from itself — the only
                            # reading under which "distinct" is defensible.
                            "distinct": bool(abs(float(td.vector @ rd.vector)) < min(tf, rf)),
                        })

    df = pd.DataFrame(out)
    save_df(ctx.out / "style_level2.csv", df)

    summary = {"model": slug, "n_rows": int(len(df)),
               "corpus": json.loads((sdir / "style_corpus_meta.json").read_text())}
    for arm, g in df.groupby("arm"):
        best = g.loc[g.cos_tag_register.idxmax()]
        summary[f"{arm}"] = {
            "median_cos_tag_register": round(float(g.cos_tag_register.median()), 4),
            "max_cos_tag_register": round(float(g.cos_tag_register.max()), 4),
            "median_floor": round(float(g.floor.median()), 4),
            "frac_distinct": round(float(g.distinct.mean()), 4),
            "worst_case": {"layer": int(best.layer), "position": best.position,
                           "tag": best.tag_contrast, "register": best.register_contrast,
                           "cos": round(float(best.cos_tag_register), 4),
                           "floor": round(float(best.floor), 4)},
        }
        log.info(f"  [{arm}] cos(tag, register) median "
                 f"{g.cos_tag_register.median():.3f} max {g.cos_tag_register.max():.3f} "
                 f"vs floor {g.floor.median():.3f} — distinct in "
                 f"{g.distinct.mean():.1%} of cells")
    save_json(ctx.out / "style_level2_summary.json", summary)


def stage_fidelity(ctx: Context) -> None:
    """Two E1.1 checks: the role paper's own read site, and cross-corpus transfer.

    `pre_mlp` fidelity asks whether we reproduce their probe where *they* hook
    (`post_attention_layernorm`). Cross-corpus transfer asks the sharper question:
    does a probe trained on our crossed corpus (instructions inside role tags)
    classify roles on their constant-content webtext? If it cannot, it is reading
    "instruction-ness" rather than role and must be rebuilt.
    """
    from core.model_io import load_model, resolve_device
    from core.positions import render
    cfg, log, slug = ctx.cfg, ctx.log, ctx.model
    device = resolve_device()
    model, tok = load_model(ctx.spec().model_id, device, logger=log)
    n_layers = model_meta.num_layers(model.config)

    df = _render_items(cfg, tok, ctx.model)
    probes = load_torch(cfg.dir(NAME, slug) / "role_probes.pt")
    probe_csv = pd.read_csv(cfg.dir(NAME, slug) / "role_probe.csv")
    best_layer = int(probe_csv.loc[probe_csv.train_accuracy.idxmax(), "layer"])
    out_rows = {}

    # -- pre_mlp fidelity, at the residual probe's best layer
    #
    # Only where the architecture HAS that site. Nemotron-H is a Mamba/attention
    # hybrid (`MEMEM*EMEM...`); its blocks expose `norm` and `mixer` only, so
    # `post_attention_layernorm` does not exist and the role paper's read site
    # cannot be reproduced on it. That is a property of the model, not a failure
    # of the run — but it used to raise, killing the job at stage 13 of 14 and
    # taking `causal` (and therefore GATE 1) with it.
    #
    # Recorded as `applicable: false` with the reason, never silently skipped:
    # the write-up must be able to say which models the reproduction covers.
    from core.capture import has_site
    if not has_site(model_meta.find_layers(model)[0], "pre_mlp"):
        out_rows["pre_mlp_fidelity"] = {
            "applicable": False,
            "reason": f"{type(model_meta.find_layers(model)[0]).__name__} has no "
                      f"post_attention_layernorm — this architecture has no "
                      f"post-attention site to read",
            "architecture": type(model).__name__,
            "note": "the role paper's read-site reproduction is not defined for this "
                    "architecture; the residual-stream result is unaffected"}
        log.warning(f"[{slug}] pre_mlp fidelity NOT APPLICABLE: "
                    f"{out_rows['pre_mlp_fidelity']['reason']}")
        log.info(f"[{slug}] continuing to the cross-corpus transfer check, which "
                 f"reads the residual stream and is unaffected")
        _skip_pre_mlp = True
    else:
        _skip_pre_mlp = False

    if not _skip_pre_mlp:
        cap2 = ActivationCapture(model, site="pre_mlp")
        ta, ow = cap2.at_content_tokens(tok, list(df.rendered), [best_layer],
                                        cfg.max_content_tokens, cfg.batch_size, device, seed=cfg.seed)
        o = ow.tolist()
        r2 = [df.role.iloc[i] for i in o]
        m2 = torch.tensor([df.split.iloc[i] == "train" for i in o])
        p2 = extract.train_role_probe(ta[best_layer][m2], [r for r, k in zip(r2, m2.tolist()) if k],
                                      ta[best_layer][~m2], [r for r, k in zip(r2, (~m2).tolist()) if k],
                                      layer=best_layer, site="pre_mlp", seed=cfg.seed)
        out_rows["pre_mlp_fidelity"] = {
            "layer": best_layer, "pre_mlp_accuracy": p2.accuracy,
            "residual_accuracy": float(probe_csv.accuracy.max())}
        log.info(f"  pre_mlp @L{best_layer}: {p2.accuracy:.3f} "
                 f"(residual {float(probe_csv.accuracy.max()):.3f})")

    # -- cross-corpus transfer, both directions
    try:
        # The FROZEN E1.0b corpus, not a fresh draw. Re-sampling C4 here would make
        # the transfer check unreproducible and model-dependent, and the whole
        # point of freezing the corpus is that both models see identical text.
        transfer = pools.load_transfer_corpus(cfg.results_root / CORPUS)
        ref = [{"role": role, "uid": t.uid, "split": t.split,
                "rendered": render(tok, t.text, role, "fixed_slot")}
               for t in transfer for role in cfg.roles_for(slug)]
        rdf = pd.DataFrame(ref)
        log.info(f"  transfer corpus: {len(transfer)} C4 passages x {len(cfg.roles_for(slug))} roles "
                 f"= {len(rdf)} items (frozen at E1.0b)")
        cap = ActivationCapture(model, site="residual")
        ra, ro = cap.at_content_tokens(tok, list(rdf.rendered), [best_layer],
                                       cfg.max_content_tokens, cfg.batch_size, device, seed=cfg.seed)
        owner = ro.tolist()
        roles_ref = [rdf.role.iloc[i] for i in owner]
        ref_train = torch.tensor([rdf.split.iloc[i] == "train" for i in owner])

        # Direction 1: OUR probe (fitted on the crossed instruction corpus) applied
        # to constant-content C4 prose, on that corpus's HELD-OUT split. If it
        # fails, the probe was reading "instruction-ness" rather than role.
        ours_on_theirs = extract.probe_transfer_accuracy(
            probes[best_layer], ra[best_layer][~ref_train],
            [r for r, m in zip(roles_ref, (~ref_train).tolist()) if m])

        # Direction 2: a probe fitted on the transfer corpus's TRAIN split,
        # applied to our corpus. Fitting and evaluating on the same rows would
        # report a training accuracy and prove nothing.
        their_probe = extract.train_role_probe(
            ra[best_layer][ref_train],
            [r for r, m in zip(roles_ref, ref_train.tolist()) if m],
            ra[best_layer][~ref_train],
            [r for r, m in zip(roles_ref, (~ref_train).tolist()) if m],
            layer=best_layer, site="residual", seed=cfg.seed)
        cap3 = ActivationCapture(model, site="residual")
        oa, oo = cap3.at_content_tokens(tok, list(df.rendered), [best_layer],
                                        cfg.max_content_tokens, cfg.batch_size, device, seed=cfg.seed)
        ours_test = torch.tensor([df.split.iloc[i] == "test" for i in oo.tolist()])
        roles_ours = [df.role.iloc[i] for i, m in zip(oo.tolist(), ours_test.tolist()) if m]
        theirs_on_ours = extract.probe_transfer_accuracy(
            their_probe, oa[best_layer][ours_test], roles_ours)
        chance = 1.0 / len(set(roles_ref))
        out_rows["transfer"] = {"layer": best_layer, "chance": chance,
                                "ours_on_constant_content": ours_on_theirs,
                                "theirs_on_our_corpus": theirs_on_ours,
                                "their_probe_own_heldout": their_probe.accuracy,
                                "n_documents": len(transfer),
                                "n_ref_train_tokens": int(ref_train.sum()),
                                "n_ref_test_tokens": int((~ref_train).sum())}
        log.info(f"  transfer @L{best_layer}: ours->constant-content {ours_on_theirs:.3f}, "
                 f"theirs->our corpus {theirs_on_ours:.3f} (chance {chance:.3f})")
    except Exception as e:
        out_rows["transfer"] = {"error": f"{type(e).__name__}: {str(e)[:140]}"}
        log.warning(f"  transfer check failed: {out_rows['transfer']['error']}")

    save_json(ctx.out / "fidelity.json", out_rows)


# ---------------------------------------------------------------------------
# stage: causal  (E1.6 — GO/NO-GO GATE 1)
# ---------------------------------------------------------------------------
def _role_direction(blob, idx, position, layer, a_role, b_role, name):
    train = idx.split.eq("train").to_numpy()
    A = blob[position][layer]
    ma = torch.tensor(train & idx.role.eq(a_role).to_numpy())
    mb = torch.tensor(train & idx.role.eq(b_role).to_numpy())
    return extract.diff_of_means(A[ma], A[mb], name, layer, "residual", position, a_role, b_role)


def _class_gap(direction) -> float:
    """`B`'s own between-class separation along `r_B`, in projection units.

    This is the denominator that makes effects on different concepts comparable:
    it converts "moved 3.2 units along R_harm" into "moved 0.4 of R_harm's own
    class gap". Without it the gate compares quantities on arbitrary scales.

    It is the direction's own `raw_norm` — the norm of the difference of class
    means from which it was fitted — NOT a gap recomputed on the steering probe
    set. An earlier version did the latter and used the harmful/harmless split for
    every target, including `R_role`; since the probe set is all one role, the
    role direction had no class variation there, its denominator was noise, and
    its deltas were inflated by more than an order of magnitude (class gap 6.7
    against ~120 for the other two). That corrupted both the diagonal and the
    random null band.

    Using `raw_norm` also gives the measure its intended calibration: steering
    adds `alpha * raw_norm * unit`, so projection onto that unit direction rises
    by `alpha * raw_norm`, and dividing by `raw_norm` makes `delta_AA == alpha`
    exactly at the steer layer. The diagonal is thereby interpretable, and its
    departure from `alpha` downstream is the propagation being measured.
    """
    g = float(getattr(direction, "raw_norm", float("nan")))
    return g if np.isfinite(g) and g > 1e-8 else float("nan")


def _delta(steered: torch.Tensor, base: torch.Tensor, direction, gap: float) -> Dict[str, float]:
    """Paired standardised projection shift, with a bootstrap CI.

    PAIRED — same items with and without the intervention — so any baseline offset
    between conditions cancels exactly rather than being corrected for afterwards.
    That offset is the failure mode the "offset-free readout" rule was written
    against; pairing removes it at the source.

    Note this is deliberately NOT AUC. Additive steering shifts every item equally
    along the steered direction, and AUC is rank-based, hence invariant to a
    uniform shift: it cannot see the effect steering produces. AUC returns as a
    SECONDARY readout, answering the different question of whether the
    intervention degraded the concept's separability.
    """
    if not np.isfinite(gap) or gap < 1e-8:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    per_item = ((direction.project(steered).float() - direction.project(base).float()) / gap)
    v = per_item.numpy()
    g = np.random.default_rng(0)
    # Vectorised resample rather than a Python loop over draws — identical
    # estimator. This runs once per matrix cell, and E1.6 now reads EVERY layer
    # downstream of the steer layer (order 20k cells per model, up from ~2k), so
    # the loop form would add tens of minutes of pure CPU to every causal run.
    boot = v[g.integers(0, len(v), size=(2000, len(v)))].mean(axis=1)
    return {"delta": float(v.mean()),
            "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5))}


def stage_causal(ctx: Context) -> None:
    """E1.6 — are R_role, R_harm and R_control causally distinguishable? GATE 1.

    Zhao et al.'s asymmetric steering test, extended to a 3x3 matrix: steer along
    A, read B downstream, and record behaviour.

    **Why the gate is not diagonal dominance.** At the steer layer, steering along
    the raw difference-in-means by alpha raises the projection onto that same
    direction by exactly alpha class-gaps — by construction. The diagonal is
    therefore close to tautological and stays favoured at downstream layers, so it
    cannot carry the argument. It is retained as a sanity condition: failing it
    means something is broken, passing it establishes nothing.

    The evidential weight sits on the off-diagonal ASYMMETRY (two variables that
    were one variable measured twice would couple symmetrically) and on
    BEHAVIOURAL DISSOCIATION (an intervention that moves behaviour while leaving
    another variable inside its null band).

    **The capability range, and a correction.** An earlier version of this stage
    defined a cell as in-range iff its KL did not exceed the magnitude-matched
    RANDOM direction's KL at the same (layer, alpha). That was wrong, and wrong in
    a way that silently disabled the gate: a direction that carries real content
    *should* move the model more than an equally large meaningless perturbation —
    that gap is exactly what `steering_sanity` measures and requires to be large.
    Using it as a capability bound therefore demanded that the intervention have
    no more effect than noise, admitted only ~21% of cells, and never admitted two
    sources at the same (layer, alpha) simultaneously, so the asymmetry condition
    had no pairs to test and reported a vacuous FAIL.

    The two questions are now separated:

    * **content** (a validity precondition, in `steering_sanity`): the real
      direction's KL must EXCEED the magnitude-matched random direction's. Also
      recorded per cell as `content_ratio`.
    * **capability** (the in-range condition): disruption on HARMLESS prompts,
      which is Arditi's `kl_score` convention. There is no published cutoff, so
      the bound is not fixed — it is a **swept dimension**, and the gate verdict
      is reported as a function of it. A verdict that holds only at one bound is
      not a verdict.
    """
    from core.interventions import (generate_steered, kl_from_baseline,
                                    next_token_stats, steer_and_capture,
                                    steering_sanity)
    from core.model_io import load_model, resolve_device
    cfg, log, slug = ctx.cfg, ctx.log, ctx.model

    blob, idx, dirs, val = _load_cache(ctx)
    n_layers = len(blob["t_post_inst"])
    probe_csv = pd.read_csv(cfg.dir(NAME, slug) / "role_probe.csv")

    # ---- the three variables, at any layer. Role uses the CONTRAST estimator:
    # a probe boundary has no activation-space magnitude, so alpha would have no
    # unit. tool-vs-user is the canonical steering direction (it is the
    # injection-relevant contrast and the pre-registered PLAN-INF cell).
    # WHICH control variable? This must be explicit, because the two candidates are
    # NOT interchangeable — measured cos(under, over) = 0.72 against a split-half
    # floor of 0.944, so they are different directions:
    #
    #   under : refused vs complied among HARMFUL prompts   (under-refusal)
    #   over  : refused vs complied among HARMLESS prompts  (over-refusal)
    #
    # Silently falling back from one to the other makes a cross-model comparison
    # meaningless: it would compare `harm -> under-refusal` on one model against
    # `harm -> over-refusal` on another and call the difference a failure to
    # replicate. `auto` preserves the old fallback for convenience; a cross-model
    # claim must pin the variant explicitly and both runs must use the same one.
    variant = os.environ.get("CONTROL_VARIANT", "auto").lower()
    _CTRL = {"under": "R_control", "over": "R_control_harmless"}
    if variant == "auto":
        ctrl_key = "R_control" if "R_control" in dirs else "R_control_harmless"
    else:
        if variant not in _CTRL:
            raise ValueError(f"CONTROL_VARIANT must be one of {sorted(_CTRL)} or 'auto'")
        ctrl_key = _CTRL[variant]
        if ctrl_key not in dirs:
            # Whether `under` is estimable is a PROPERTY OF THE MODEL's refusal
            # behaviour, discovered at E1.1 Stage 1, not something a submission
            # script can know in advance. The matched cross-model arm is always
            # `over`; the `under` arm is run opportunistically per model, and
            # `CONTROL_VARIANT_OPTIONAL=1` lets that arm report "not estimable
            # on this model" and exit CLEANLY rather than failing the job and
            # breaking the `afterok` chain for everything downstream.
            msg = (f"[{slug}] CONTROL_VARIANT={variant} requires {ctrl_key}, which was not "
                   f"extracted for this model (available: {sorted(dirs)}). On a model whose "
                   f"harmful-and-complied cell is empty, only 'over' is estimable.")
            if os.environ.get("CONTROL_VARIANT_OPTIONAL", "").strip().lower() in {"1", "true", "yes"}:
                log.warning(msg)
                log.warning(f"[{slug}] CONTROL_VARIANT_OPTIONAL set -> skipping the "
                            f"'{variant}' arm for this model, exiting 0. The matched "
                            f"cross-model arm is unaffected.")
                (cfg.dir(NAME, slug) / f"causal_skipped__{variant}.json").write_text(
                    json.dumps({"model": slug, "variant": variant,
                                "required_direction": ctrl_key,
                                "available_directions": sorted(dirs),
                                "reason": "harmful-and-complied cell too small to fit "
                                          "the refused-vs-complied contrast"}, indent=2))
                return
            raise SystemExit(msg)
    ctrl_variant = "under" if ctrl_key == "R_control" else "over"
    log.info(f"[{slug}] control variable: {ctrl_key} (variant='{ctrl_variant}', "
             f"requested '{variant}')")

    def build(layer, position: str = "t_post_inst") -> Dict[str, object]:
        """Read-out directions at ONE position.

        Position matters and cannot be mixed. The residual basis differs between
        `t_inst` and `t_post_inst`, so projecting a `t_inst` activation onto a
        `t_post_inst`-fitted direction is exactly the cross-position comparison
        the analysis protocol forbids — it would produce a number with no
        interpretation rather than an error.

        `R_control` therefore has **no `t_inst` entry**: refused-versus-complied
        is a property of the state after the instruction has been read, and no
        `t_inst` fit of it exists. That asymmetry is a fact about the variable,
        not a missing feature, so the token-resolved readout reports `R_harm` and
        `R_role` at `t_inst` and all three at `t_post_inst`.
        """
        c = {}
        if position == "t_post_inst":
            if "R_harm_at_post" in dirs:
                c["R_harm"] = dirs["R_harm_at_post"][layer]
            c["R_control"] = dirs[ctrl_key][layer]
        else:
            if "R_harm" in dirs:
                c["R_harm"] = dirs["R_harm"][layer]
        c["R_role"] = _role_direction(blob, idx, position, layer, "tool", "user",
                                      f"R_role_tool_vs_user@{position}")
        return c

    # ---- probe set: balanced, held-out, one role and one design so that the
    # steering effect is not confounded by corpus composition.
    test = (idx.split.eq("test").to_numpy() & idx.role.eq("user").to_numpy()
            & idx.design.eq("fixed_slot").to_numpy())
    train = (idx.split.eq("train").to_numpy() & idx.role.eq("user").to_numpy()
             & idx.design.eq("fixed_slot").to_numpy())
    harmful = idx.harmful.astype(bool).to_numpy()

    def balanced(mask, cap):
        p, n = np.nonzero(mask & harmful)[0], np.nonzero(mask & ~harmful)[0]
        k = min(len(p), len(n), cap)
        return np.concatenate([p[:k], n[:k]]), k

    sel_te, k_te = balanced(test, 8 if cfg.fast_dev else 50)
    sel_tr, k_tr = balanced(train, 8 if cfg.fast_dev else 40)
    log.info(f"[{slug}] probe sets: train {2*k_tr}, test {2*k_te} (balanced harmful/harmless)")

    device = resolve_device()
    model, tok = load_model(ctx.spec().model_id, device, logger=log)
    rendered = _render_items(cfg, tok, ctx.model)
    if not (list(rendered.uid) == list(idx.uid) and list(rendered.role) == list(idx.role)):
        raise ValueError("re-rendered corpus does not align with the cached activation index")
    texts_tr = [rendered.text.iloc[i] for i in sel_tr]
    texts_te = [rendered.text.iloc[i] for i in sel_te]
    # `Rendered` objects for the train probe set: Stage B's steered-position masks
    # are per item, built from each item's own token spans.
    rendered_tr = [rendered.rendered.iloc[i] for i in sel_tr]
    # The `Rendered` objects carry each item's own token positions; the readout
    # needs them because `t_inst` sits at a different index in every row.
    rendered_te = [rendered.rendered.iloc[i] for i in sel_te]
    harm_te = harmful[sel_te]

    # ---- precondition on the primitive itself
    mid = max(1, int(0.45 * n_layers))
    sanity = steering_sanity(model, tok, texts_tr, build(mid)["R_harm"], mid,
                             device, cfg.batch_size, seed=cfg.seed)
    save_json(ctx.out / f"steering_sanity{CAUSAL_SUFFIX}.json", sanity)
    if not sanity["deterministic"]:
        raise SystemExit("steering is non-deterministic; the measurement is unreliable")
    log.info(f"  sanity @L{mid}: determinism KL {sanity['determinism_kl']:.1e} "
             f"({'ok' if sanity['deterministic'] else 'NON-DETERMINISTIC'}); "
             f"raw_norm {sanity['direction_raw_norm']:.2f}")
    for a, v in sanity["alpha_curve"].items():
        log.info(f"    alpha={a}: real KL {v['real_kl']:.4f} vs magnitude-matched "
                 f"random {v['random_kl_mean']:.4f} (content gap {v['content_gap']:+.4f})")

    ALPHAS = [0.25, 0.5, 1.0, 2.0, 4.0] if not cfg.fast_dev else [1.0]
    SIGNS = [1.0, -1.0]
    N_RANDOM_DIRS = 8 if not cfg.fast_dev else 2

    # =====================================================================
    # Stage A — layer profile, on the TRAIN split.
    # Every layer, alpha = +/-1, so no layer is privileged by selection and the
    # layer that is later named was named on train.
    # =====================================================================
    log.info(f"[{slug}] stage A: layer profile on train, all {n_layers} layers, alpha=+/-1")
    prof = []
    cand_layers = [l for l in range(1, n_layers - 1)]
    if cfg.fast_dev:
        cand_layers = cand_layers[:: max(1, len(cand_layers) // 4)]
    base_stats_tr = next_token_stats(model, tok, texts_tr, None, 1, 0.0, device, cfg.batch_size)
    for L in cand_layers:
        cs = build(L)
        rl = [l for l in (L + 1, min(n_layers - 1, L + 2)) if l > L][:1]
        if not rl:
            continue
        for name, d in cs.items():
            for sgn in SIGNS:
                st = next_token_stats(model, tok, texts_tr, d, L, sgn, device, cfg.batch_size)
                kl = float(kl_from_baseline(base_stats_tr["logprobs"], st["logprobs"]).mean())
                prof.append({"model": slug, "stage": "A", "steer_layer": L,
                             "relative_depth": round(relative_depth(L, n_layers), 4),
                             "source": name, "alpha": sgn, "mean_kl": kl,
                             "margin_shift": float((st["refusal_margin"]
                                                    - base_stats_tr["refusal_margin"]).mean())})
    prof_df = pd.DataFrame(prof)
    save_df(ctx.out / f"causal_layer_profile{CAUSAL_SUFFIX}.csv", prof_df)

    # Steer layers named on TRAIN: the depth at which each source moves the
    # behavioural readout most, plus fixed sensitivity depths so the verdict is
    # never read off a single selected layer.
    # EXPERIMENTS.md: "top-3 layers per source from A, plus fixed sensitivity
    # layers at relative depth {0.25, 0.5, 0.75}". Top-3, not the argmax: a single
    # best layer per source makes Stage C's layer set hostage to one noisy
    # train-side maximum.
    TOP_PER_SOURCE = 3
    chosen = sorted({
        int(r.steer_layer)
        for _, g in prof_df.assign(absshift=prof_df.margin_shift.abs()).groupby("source")
        for _, r in g.nlargest(TOP_PER_SOURCE, "absshift").iterrows()})
    fixed = sorted({max(1, int(f * (n_layers - 1))) for f in (0.25, 0.5, 0.75)})
    steer_layers = sorted(set(chosen) | set(fixed))
    steer_layers = [l for l in steer_layers if l < n_layers - 1]
    log.info(f"  stage A top-{TOP_PER_SOURCE}/source {chosen}; "
             f"sensitivity depths {fixed}; steering at {steer_layers}")

    # =====================================================================
    # Stage B — refine, on the TRAIN split.
    # The alpha and token-position profiles. This is where the pre-registered
    # steered-position sweep lives: Stage A fixes alpha=+/-1 and steers every
    # real token, Stage C is the verdict and must not be used to choose anything.
    # =====================================================================
    log.info(f"[{slug}] stage B: alpha x token-position profile on train, "
             f"{len(steer_layers)} layers x {len(ALPHAS)*len(SIGNS)} alpha x "
             f"{len(TOKEN_SETS)} token sets")
    brows = []
    for L in steer_layers:
        cs = build(L)
        for tset in TOKEN_SETS:
            if tset == "all_real":
                pmasks = None
            else:
                pmasks = [_steer_mask(r, tset) for r in rendered_tr]
                empty = sum(1 for m in pmasks if not any(m))
                if empty:
                    # Skipped, not silently steered everywhere. An empty span is a
                    # fact about the chat template, so it is logged once per cell
                    # rather than averaged into a profile that looks complete.
                    log.info(f"    L{L} {tset}: {empty}/{len(pmasks)} items have an "
                             f"empty span — token set skipped at this layer")
                    continue
            for name, d in cs.items():
                for a in ALPHAS:
                    for sgn in SIGNS:
                        st = next_token_stats(model, tok, texts_tr, d, L, sgn * a,
                                              device, cfg.batch_size,
                                              position_masks=pmasks)
                        kl = float(kl_from_baseline(base_stats_tr["logprobs"],
                                                    st["logprobs"]).mean())
                        brows.append({
                            "model": slug, "stage": "B", "steer_layer": L,
                            "relative_depth": round(relative_depth(L, n_layers), 4),
                            "source": name, "token_set": tset,
                            "alpha": sgn * a, "abs_alpha": a, "mean_kl": kl,
                            "margin_shift": float((st["refusal_margin"]
                                                   - base_stats_tr["refusal_margin"]).mean())})
        log.info(f"    L{L}: {len([r for r in brows if r['steer_layer'] == L])} cells")
    b_df = pd.DataFrame(brows)
    save_df(ctx.out / f"causal_stage_b{CAUSAL_SUFFIX}.csv", b_df)
    if len(b_df):
        for tset, g in b_df.groupby("token_set"):
            log.info(f"  stage B {tset:<18} mean |margin shift| "
                     f"{g.margin_shift.abs().mean():.4f}  mean KL {g.mean_kl.mean():.4f}")

    # =====================================================================
    # Stage C — the verdict, on the TEST split.
    # =====================================================================
    rows = []
    for L in steer_layers:
        # EVERY layer downstream of the steer layer, as EXPERIMENTS.md requires:
        # "Read layers are free within a forward pass, so every layer downstream
        # of the steer layer is read." An earlier version read three of them
        # (L+1, midpoint, last), which reduced a dense depth profile to three
        # samples of it.
        read_layers = list(range(L + 1, n_layers))
        if not read_layers:
            continue
        # ...but the GATE's test family stays those three, pre-registered.
        #
        # Reading a layer is free; *testing* it is not. Two reasons, and the
        # second is the stronger one:
        #
        # 1. Adjacent read layers of one steered forward pass are near-duplicates,
        #    so folding ~30 of them into the BH family multiplies the family
        #    without adding evidence, and every real effect pays for it.
        # 2. `range(L+1, n_layers)` has a length that DEPENDS ON L — 27 layers when
        #    steering at 5, 7 when steering at 25. An all-layer family would
        #    therefore penalise early steer layers ~4x harder than late ones for
        #    a purely positional reason. Defined relative to L, these three make
        #    every steer layer contribute equally.
        #
        # Fixed in advance, not chosen after seeing the profile: selecting gate
        # layers by which ones looked strongest is the exact researcher degree of
        # freedom the pre-registration exists to remove.
        gate_layers = {l for l in (L + 1, int((L + n_layers) / 2), n_layers - 1) if l > L}
        steer_cs = build(L)
        # Read at BOTH corpus positions: PLAN-INF asks for the effect "at later
        # layers and later tokens", and `t_inst` -> `t_post_inst` is precisely an
        # earlier and a later token of the same forward pass. Each position uses
        # directions fitted at that position (see `build`).
        read_cs = {(l, p): build(l, p) for l in read_layers for p in READ_POSITIONS}

        base_caps = steer_and_capture(model, tok, texts_te, None, L, read_layers,
                                      0.0, device, cfg.batch_size,
                                      read_rendered=rendered_te, read_which=READ_POSITIONS)
        base_stats = next_token_stats(model, tok, texts_te, None, L, 0.0, device, cfg.batch_size)

        # Behavioural baseline by GENERATION, labelled with the published
        # refusal-prefix rule. This is the readout G3 rests on; the logit margin
        # is retained only as a cheap cross-check and is never reported alone.
        base_gen = generate_steered(model, tok, texts_te, None, L, 0.0, device,
                                    cfg.batch_size, cfg.refusal_max_new_tokens)
        base_ref = np.array([refusal.has_refusal_marker(g, refusal.PRIMARY_RULE)
                             for g in base_gen], dtype=float)
        log.info(f"  L{L}: baseline refusal rate {base_ref.mean():.3f} "
                 f"(harmful {base_ref[harm_te].mean():.3f}, "
                 f"harmless {base_ref[~harm_te].mean():.3f})")

        # Each concept's own class-mean separation, from its own fitted contrast.
        gaps = {key: {nm: _class_gap(d) for nm, d in cs.items()}
                for key, cs in read_cs.items()}

        # Magnitude-matched random reference at this layer. NOT a capability
        # bound (see the docstring) — it is the denominator of `content_ratio`,
        # which says how much of a cell's effect is content rather than push size.
        hmask = torch.tensor(~harm_te)          # harmless items: Arditi's kl_score set
        rnd_kl = {}
        for a in ALPHAS:
            vals = []
            for r in range(N_RANDOM_DIRS):
                rd = extract.random_direction_like(steer_cs["R_harm"], seed=cfg.seed + 1000 * r + L)
                st = next_token_stats(model, tok, texts_te, rd, L, a, device, cfg.batch_size)
                vals.append(float(kl_from_baseline(base_stats["logprobs"],
                                                   st["logprobs"])[hmask].mean()))
            rnd_kl[a] = float(np.mean(vals))
        log.info(f"  L{L}: magnitude-matched random KL (harmless) by alpha " +
                 ", ".join(f"{a}:{v:.4f}" for a, v in rnd_kl.items()))

        sources = dict(steer_cs)
        sources["random"] = extract.random_direction_like(steer_cs["R_harm"], seed=cfg.seed + L)

        for src, sd in sources.items():
            for a in ALPHAS:
                for sgn in SIGNS:
                    alpha = sgn * a
                    caps = steer_and_capture(model, tok, texts_te, sd, L, read_layers,
                                             alpha, device, cfg.batch_size,
                                             read_rendered=rendered_te,
                                             read_which=READ_POSITIONS)
                    st = next_token_stats(model, tok, texts_te, sd, L, alpha, device, cfg.batch_size)
                    kl_per = kl_from_baseline(base_stats["logprobs"], st["logprobs"])
                    kl = float(kl_per.mean())
                    kl_harmless = float(kl_per[hmask].mean())   # the capability quantity
                    content_ratio = kl_harmless / max(rnd_kl[a], 1e-9)

                    gen = generate_steered(model, tok, texts_te, sd, L, alpha, device,
                                           cfg.batch_size, cfg.refusal_max_new_tokens)
                    ref = np.array([refusal.has_refusal_marker(g, refusal.PRIMARY_RULE)
                                    for g in gen], dtype=float)
                    d_ref = float(ref.mean() - base_ref.mean())
                    d_ref_harmful = float(ref[harm_te].mean() - base_ref[harm_te].mean())
                    d_ref_harmless = float(ref[~harm_te].mean() - base_ref[~harm_te].mean())
                    for l in read_layers:
                        for pi, rpos in enumerate(READ_POSITIONS):
                            cs = read_cs[(l, rpos)]
                            A, B = caps[l][:, pi, :], base_caps[l][:, pi, :]
                            for tgt, td in cs.items():
                                dd = _delta(A, B, td, gaps[(l, rpos)][tgt])
                                # secondary readout: did separability degrade?
                                try:
                                    auc_s = extract.auc(td.project(A)[harm_te],
                                                        td.project(A)[~harm_te])
                                    auc_b = extract.auc(td.project(B)[harm_te],
                                                        td.project(B)[~harm_te])
                                except Exception:
                                    auc_s = auc_b = float("nan")
                                rows.append({
                                "model": slug, "stage": "C", "steer_layer": L,
                                "read_layer": l, "read_position": rpos,
                                "gate_read_layer": l in gate_layers,
                                "relative_read_depth": round(relative_depth(l, n_layers), 4),
                                "source": src, "target": tgt,
                                "alpha": alpha, "abs_alpha": a,
                                "delta": dd["delta"], "ci_low": dd["ci_low"],
                                "ci_high": dd["ci_high"],
                                "delta_auc": auc_s - auc_b,
                                "mean_kl": kl, "kl_harmless": kl_harmless,
                                "random_kl_harmless": rnd_kl[a],
                                "content_ratio": content_ratio,
                                "refusal_rate": float(ref.mean()),
                                "d_refusal": d_ref,
                                "d_refusal_harmful": d_ref_harmful,
                                "d_refusal_harmless": d_ref_harmless,
                                "margin_shift": float((st["refusal_margin"]
                                                       - base_stats["refusal_margin"]).mean()),
                                "class_gap": gaps[(l, rpos)][tgt],
                                })
        log.info(f"  L{L}: {len([r for r in rows if r['steer_layer'] == L])} cells")

    mat = pd.DataFrame(rows)
    save_df(ctx.out / f"causal_matrix{CAUSAL_SUFFIX}.csv", mat)

    # =====================================================================
    # The gate
    # =====================================================================
    verdict = _adjudicate(mat, log, slug)
    save_json(ctx.out / f"causal_gate{CAUSAL_SUFFIX}.json", verdict)


def _adjudicate(mat: pd.DataFrame, log, slug: str,
                opts: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """Gate 1, adjudicated across a swept capability bound.

    There is no published cutoff for "the intervention preserved capability", so
    fixing one would decide the verdict by fiat. The bound on harmless-prompt KL
    is therefore swept, and the verdict is reported at each level. A verdict is
    only reported as such if it is stable across the bounds where any cell
    survives; otherwise the sweep itself is the finding.
    """
    # GATE 1 is adjudicated at `t_post_inst` ONLY.
    #
    # The matrix now carries a second read position (`t_inst`), added so the
    # token-resolved readout PLAN-INF asks for exists. Adjudicating over both
    # would silently change what the gate means: it would double the FDR family,
    # mix two incomparable residual bases into one asymmetry test, and count the
    # same intervention twice. The token profile is *additional evidence about
    # the readout*, not additional gate cells.
    if "read_position" in mat.columns:
        mat = mat[mat.read_position == "t_post_inst"]
    # The READ-LAYER family is NOT filtered here — it is an adjudication option
    # (`gate_layers`) applied inside `_adjudicate_at`, so that the sweep can vary
    # it. See `ADJUDICATION_DEFAULTS`.

    bounds = [0.1, 0.25, 0.5, 1.0, 2.0]
    per_bound = {}
    for b in bounds:
        v = _adjudicate_at(mat, b, opts)
        per_bound[str(b)] = v
        log.info(f"  KL_harmless <= {b:<4}: n_live={v['n_live']:<4} "
                 f"G1 {v['G1_diagonal_dominance']['rate']} "
                 f"G2 {v['G2_asymmetry']['n_asymmetric']}/{v['G2_asymmetry']['n_pairs_tested']} "
                 f"G3 {v['G3_behavioural_dissociation']['n']} -> {v['verdict']}")

    testable = [b for b in bounds if per_bound[str(b)]["G2_asymmetry"]["n_pairs_tested"] > 0]
    verdicts = {per_bound[str(b)]["verdict"] for b in testable}
    if not testable:
        overall = "undecidable"
    elif verdicts == {"PASS"}:
        overall = "PASS"
    elif verdicts == {"FAIL"}:
        overall = "FAIL"
    else:
        overall = "bound-sensitive"

    out = {"model": slug, "n_cells": int(len(mat)),
           "adjudication_options": {**ADJUDICATION_DEFAULTS, **(opts or {})},
           "capability_bounds_swept": bounds,
           "bounds_with_testable_pairs": testable,
           "per_bound": per_bound,
           "verdict": overall,
           "criterion": ("G2 (off-diagonal asymmetry) AND G3 (behavioural dissociation) "
                         "on cells within the capability bound; G1 is a sanity check "
                         "and is not evidence. Reported across a swept bound.")}
    log.info(f"[{slug}] GATE 1 — {overall} "
             f"(testable at bounds {testable or 'none'})")
    return out


def _bh_fdr(pvals: Sequence[float], q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up. Returns a boolean 'rejected' mask.

    EXPERIMENTS.md pre-registers FDR control across the ordered pairs x layers
    tested, and the family is large: ~100 asymmetry tests per model. Counting
    qualifying cells without it would report an expected handful of false
    positives as structure.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    thresh = q * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresh
    k = np.nonzero(passed)[0].max() + 1 if passed.any() else 0
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out


def _two_sample_p(d1: float, lo1: float, hi1: float,
                  d2: float, lo2: float, hi2: float) -> float:
    """Two-sided p for 'the two directed effects differ', from bootstrap CIs.

    The bootstrap CI half-width gives an approximate standard error
    (95% percentile interval ~ +/-1.96 SE), so a Wald test on the difference is
    the natural companion to the interval that is already computed. This is an
    approximation and is used only to rank tests for FDR control; the reported
    evidence remains the interval itself.
    """
    se1 = max((hi1 - lo1) / (2 * 1.96), 1e-12)
    se2 = max((hi2 - lo2) / (2 * 1.96), 1e-12)
    z = abs(d1 - d2) / np.sqrt(se1 ** 2 + se2 ** 2)
    from math import erfc, sqrt
    return float(erfc(z / sqrt(2)))


# Every discretionary choice in the adjudication, in one place. The verdict has
# been shown to move when these change (O-12), so they are named, defaulted to the
# pre-registered settings, and swept by `tools/gate_sensitivity.py`. A verdict that
# holds only at one setting is not a verdict.
ADJUDICATION_DEFAULTS: Dict[str, object] = {
    "null_q": 95.0,          # percentile of |delta| over random directions
    "null_group": "alpha",   # "alpha" | "alpha_layer" | "pooled"
    "g2_rule": "any",        # "any" | "both" — how many directions must beat the null
    "fdr_q": 0.05,           # Benjamini-Hochberg level
    "beh_null_q": 95.0,      # percentile for the behavioural null band
    # Which read layers form the BH-FDR family. E1.6 READS every layer downstream
    # of the steer layer; that dense profile is description and needs no
    # multiplicity correction. Which of it is TESTED is a separate choice, and
    # rather than defend one, it is swept — the same treatment the capability
    # bound already gets (O-12), and free, because `tools/gate_sensitivity.py`
    # re-adjudicates saved matrices on CPU.
    #
    #   "all"       every downstream layer. Uses all the evidence; a larger BH
    #               family only makes the verdict HARDER to obtain, which is the
    #               right failure mode for a go/no-go gate. Caveat: the family
    #               size grows with distance from the output, so early steer
    #               layers face a larger family than late ones — a
    #               conservativeness gradient, not a false-positive risk.
    #   "relative3" L+1, midpoint, last. Equal family size at every steer depth,
    #               and adjacent layers are near-duplicates so little independent
    #               evidence is lost.
    "gate_layers": "all",    # "all" | "relative3"
}


def _adjudicate_at(mat: pd.DataFrame, kl_bound: float,
                   opts: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    """G1 sanity, G2 asymmetry, G3 behavioural dissociation at one capability bound.

    `opts` overrides any of `ADJUDICATION_DEFAULTS`; omitted keys take the
    pre-registered value.
    """
    o = {**ADJUDICATION_DEFAULTS, **(opts or {})}
    # Restrict the TEST family to the chosen read layers before anything is
    # counted — including the random rows, so the null band is built from the
    # same layers the real cells are tested at.
    if str(o["gate_layers"]) == "relative3" and "gate_read_layer" in mat.columns:
        mat = mat[mat.gate_read_layer.astype(bool)]
    real = mat[(mat.source != "random") & (mat.target.notna())]
    live = real[real.kl_harmless <= kl_bound]
    rnd = mat[mat.source == "random"]

    # Null band for "moved": the magnitude-matched random direction's own |delta|.
    #
    # It MUST be matched on alpha. Pooling across alpha compares a real effect at
    # alpha=1 against a null whose upper tail is set by random directions at
    # alpha=4, where every direction disrupts the model — a like-for-unlike
    # comparison that silently suppresses real effects. (Measured: pooling put the
    # behavioural band at 0.39 while no in-range real cell exceeded 0.27, so the
    # dissociation condition could never fire; at matched alpha the same real
    # directions move refusal 4-8x more than random.)
    def _band(df: pd.DataFrame, col: str, by: Sequence[str], q: float) -> Dict:
        """Null band per group, with keys the callers can actually look up.

        **`groupby` on a single-element LIST yields TUPLE keys** — `(0.25,)`, not
        `0.25`. `beh_nb` looked up the scalar, missed every key, and fell through
        to its `float("inf")` default, so `moved_behaviour = shift > inf` was
        ALWAYS FALSE and **G3 could never fire**. The gate is `G2 AND G3`, so
        every FAIL it has ever recorded was FAIL by construction rather than by
        evidence. The same mismatch hit `nb` under `null_group="pooled"`, which
        also groups by one column.

        Single-column keys are therefore unwrapped to scalars here, so a caller
        indexing by the natural value is right and cannot silently miss.
        """
        if not by:
            v = df[col].abs().dropna()
            return {(): float(np.percentile(v, q)) if len(v) else float("nan")}
        out = {}
        for k, g in df.groupby(list(by)):
            if len(by) == 1 and isinstance(k, tuple):
                k = k[0]
            out[k] = (float(np.percentile(g[col].abs().dropna(), q))
                      if len(g[col].dropna()) else float("nan"))
        return out

    _GROUPS = {"alpha": ["target", "abs_alpha"],
               "alpha_layer": ["target", "abs_alpha", "steer_layer"],
               "pooled": ["target"]}
    grp = _GROUPS[str(o["null_group"])]
    null_by = _band(rnd, "delta", grp, float(o["null_q"]))
    beh_col = "d_refusal" if "d_refusal" in mat.columns else "margin_shift"
    beh_grp = [] if o["null_group"] == "pooled" else ["abs_alpha"]
    beh_by = _band(rnd, beh_col, beh_grp, float(o["beh_null_q"]))

    def beh_nb(abs_alpha):
        return beh_by.get(() if not beh_grp else abs_alpha, float("inf"))

    def nb(target, abs_alpha, steer_layer=None):
        if o["null_group"] == "pooled":
            return null_by.get(target, float("inf"))
        if o["null_group"] == "alpha_layer":
            return null_by.get((target, abs_alpha, steer_layer), float("inf"))
        return null_by.get((target, abs_alpha), float("inf"))

    # A band that is not finite makes its criterion unsatisfiable, and an
    # unsatisfiable criterion reads exactly like a negative result. Checked here
    # so it surfaces as a loud failure instead of a quiet FAIL.
    _alphas_present = sorted(live.abs_alpha.dropna().unique()) if len(live) else []
    _dead = [a for a in _alphas_present if not np.isfinite(beh_nb(a))]
    if _dead:
        raise ValueError(
            f"behavioural null band is not finite for abs_alpha={_dead} — G3 could "
            f"never fire and the gate would report FAIL by construction. "
            f"beh_by keys={list(beh_by)[:6]} (check _band key types)")

    out: Dict[str, object] = {
        "kl_bound": kl_bound, "n_live": int(len(live)),
        "adjudication_options": dict(o),
        "null_band_delta": {str(k): round(v, 4) for k, v in null_by.items()},
        "null_band_behaviour": {str(k): round(v, 4) for k, v in beh_by.items()}}
    if not len(live):
        out.update(verdict="undecidable",
                   G1_diagonal_dominance={"rate": None, "n": 0},
                   G2_asymmetry={"n_pairs_tested": 0, "n_asymmetric": 0,
                                 "holds": False, "examples": []},
                   G3_behavioural_dissociation={"n": 0, "holds": False, "examples": []})
        return out

    # --- G1: |delta_AA| > |delta_AB| downstream (sanity; partly tautological)
    g1_rows = []
    for (L, rl, a), g in live.groupby(["steer_layer", "read_layer", "alpha"]):
        for src in g.source.unique():
            sub = g[g.source == src]
            own = sub[sub.target == src].delta.abs()
            oth = sub[sub.target != src].delta.abs()
            if len(own) and len(oth):
                g1_rows.append({"steer_layer": L, "read_layer": rl, "alpha": a, "source": src,
                                "own": float(own.iloc[0]), "max_other": float(oth.max()),
                                "holds": bool(own.iloc[0] > oth.max())})
    g1 = pd.DataFrame(g1_rows)
    out["G1_diagonal_dominance"] = {
        "note": "sanity only — partly true by construction; failing it means something is broken",
        "rate": round(float(g1.holds.mean()), 4) if len(g1) else None,
        "n": int(len(g1))}

    # --- G2: off-diagonal asymmetry, effect(A->B) != effect(B->A), CIs disjoint
    asym = []
    for (L, rl, a), g in live.groupby(["steer_layer", "read_layer", "abs_alpha"]):
        gg = g[g.alpha > 0]
        for A in gg.source.unique():
            for B in gg.source.unique():
                if A >= B:
                    continue
                ab = gg[(gg.source == A) & (gg.target == B)]
                ba = gg[(gg.source == B) & (gg.target == A)]
                if not len(ab) or not len(ba):
                    continue
                r1, r2 = ab.iloc[0], ba.iloc[0]
                disjoint = (r1.ci_low > r2.ci_high) or (r2.ci_low > r1.ci_high)
                # An asymmetry between two effects that are both indistinguishable
                # from a random direction is a difference between two nulls, not
                # evidence of directed influence. With n=100 items the CIs are
                # tight enough that such pairs separate readily, so magnitude is
                # required as well as separation: at least one of the two directed
                # effects must exceed its target's random-direction null band.
                b1 = abs(r1.delta) > nb(B, a, L)
                b2 = abs(r2.delta) > nb(A, a, L)
                big = (b1 or b2) if o["g2_rule"] == "any" else (b1 and b2)
                pv = _two_sample_p(r1.delta, r1.ci_low, r1.ci_high,
                                   r2.delta, r2.ci_low, r2.ci_high)
                asym.append({"steer_layer": L, "read_layer": rl, "abs_alpha": a,
                             "A": A, "B": B, "pair": f"{A}->{B} vs {B}->{A}",
                             "d_ab": r1.delta, "d_ba": r2.delta, "p": pv,
                             "cis_disjoint": bool(disjoint),
                             "beyond_null": bool(big)})
    asym_df = pd.DataFrame(asym)
    if len(asym_df):
        # FDR across the whole family of asymmetry tests at this bound
        # (ordered pairs x steer layer x read layer x alpha), as pre-registered.
        asym_df["fdr_reject"] = _bh_fdr(asym_df.p.values, q=float(o["fdr_q"]))
        asym_df["qualifies"] = (asym_df.cis_disjoint & asym_df.beyond_null
                                & asym_df.fdr_reject)
    else:
        asym_df["fdr_reject"] = []
        asym_df["qualifies"] = []
    n_asym = int(asym_df.qualifies.sum()) if len(asym_df) else 0
    out["G2_asymmetry"] = {
        "criterion": "CIs disjoint AND at least one direction beyond its alpha-matched "
                     "null band AND surviving BH-FDR (q=0.05) over the family of "
                     "asymmetry tests at this bound",
        "family_size": int(len(asym_df)),
        "n_pairs_tested": int(len(asym_df)),
        "n_cis_disjoint": int(asym_df.cis_disjoint.sum()) if len(asym_df) else 0,
        "n_beyond_null": int(asym_df.beyond_null.sum()) if len(asym_df) else 0,
        "n_fdr_reject": int(asym_df.fdr_reject.sum()) if len(asym_df) else 0,
        "n_asymmetric": n_asym,
        "holds": bool(n_asym > 0),
        "direction_counts": (asym_df[asym_df.qualifies]
                             .assign(stronger=lambda d: np.where(
                                 d.d_ab.abs() > d.d_ba.abs(),
                                 d.A + "->" + d.B, d.B + "->" + d.A))
                             .stronger.value_counts().to_dict() if n_asym else {}),
        "examples": (asym_df[asym_df.qualifies].head(6).to_dict("records") if n_asym else [])}

    # --- G3: behavioural dissociation — moved behaviour, left another variable
    #         inside its null band
    # The behavioural readout is the change in REFUSAL RATE under the
    # intervention, measured by generating and applying the published
    # refusal-prefix rule. The single-token logit margin is a proxy over two
    # hand-chosen token ids and is kept only as a cross-check.
    #
    # "Moved behaviour" needs a threshold or the condition is vacuous: any
    # non-zero shift would qualify. The threshold is the random direction's own
    # behavioural effect at matched magnitude — the same null logic used for the
    # representational readout.
    # A band of EXACTLY ZERO carries no information, and admitting it makes G3
    # vacuous at that magnitude: `shift > 0` is true for any non-zero shift, so a
    # one-item change in refusal rate qualifies. That is the mirror image of the
    # original G3 bug, where a missing key defaulted the band to `inf` and the
    # criterion could never fire; here it always fires.
    #
    # It is not hypothetical. On qwen3.5-35b-a3b the behavioural band at
    # |alpha|=0.25 came out 0.0 — random steering at that magnitude never moved
    # the refusal rate — and 206 of 362 qualifying cells came from that alpha
    # alone, every one of them a 0.01 shift against a 0.0 threshold. (The verdict
    # survived on 156 cells at |alpha| 0.5 and 1.0 where the band was informative,
    # so the PASS stood; the COUNT was inflated.)
    #
    # Degenerate alphas are therefore excluded from G3 and the exclusion is
    # recorded, rather than being silently counted or silently dropped.
    _degenerate_alphas = sorted({a for a in (float(x) for x in beh_by)
                                 if not (beh_nb(a) > 0)}) if beh_grp else []
    _n_excluded = 0
    diss = []
    for (L, rl, al), g in live.groupby(["steer_layer", "read_layer", "alpha"]):
        for src in g.source.unique():
            sub = g[g.source == src]
            if not len(sub):
                continue
            aa = float(sub.abs_alpha.iloc[0])
            if not (beh_nb(aa) > 0):
                _n_excluded += 1
                continue
            shift = abs(float(sub[beh_col].iloc[0]))
            moved_behaviour = shift > beh_nb(aa)
            unmoved = [t for t, d in zip(sub.target, sub.delta.abs())
                       if np.isfinite(d) and np.isfinite(nb(t, aa, L))
                       and d <= nb(t, aa, L) and t != src]
            if moved_behaviour and unmoved:
                diss.append({"steer_layer": L, "read_layer": rl, "alpha": al, "source": src,
                             "behaviour_shift": float(sub[beh_col].iloc[0]),
                             "left_inside_null": unmoved})
    out["G3_behavioural_dissociation"] = {
        "criterion": "behavioural shift beyond the random-direction band, while at "
                     "least one other variable stays inside its own null band",
        "behavioural_readout": beh_col,
        "behavioural_null_p95_by_alpha": {str(k): round(v, 4) for k, v in beh_by.items()},
        "degenerate_alphas_excluded": _degenerate_alphas,
        "n_cells_excluded_degenerate_band": _n_excluded,
        "n": len(diss), "holds": bool(diss),
        # a spread of alphas, not the first six, so the reader can see whether the
        # verdict leans on one magnitude
        "examples": ([diss[i] for i in
                      sorted({0, len(diss)//4, len(diss)//2, 3*len(diss)//4, len(diss)-1}
                             & set(range(len(diss))))] if diss else []),
        "qualifying_cells_by_abs_alpha": {
            str(a): sum(1 for x in diss if abs(float(x["alpha"])) == a)
            for a in sorted({abs(float(x["alpha"])) for x in diss})}}

    # A verdict requires the asymmetry test to have had pairs to test. Reporting
    # FAIL when n_pairs_tested == 0 would present a test that never ran as a
    # negative result — the exact error the first version of this stage made.
    if out["G2_asymmetry"]["n_pairs_tested"] == 0:
        out["verdict"] = "undecidable"
    else:
        out["verdict"] = ("PASS" if (out["G2_asymmetry"]["holds"]
                                     and out["G3_behavioural_dissociation"]["holds"])
                          else "FAIL")
    return out


PIPELINE = Pipeline(NAME, [
    Stage("labels", stage_labels, produces=["refusal_labels.csv", "labels_checks.json"],
          needs_gpu=True, doc="E1.1 input — refusal labels; gates R_control identifiability"),
    Stage("extract", stage_extract,
          # `activations_cache.json`, not `activations.pt`: the blob itself now
          # lives outside the results root (Config.cache_dir), and `produces`
          # drives the skip-if-complete check, which looks in the results root.
          # Naming the blob here would make `extract` appear incomplete forever
          # and re-run on every invocation.
          produces=["directions.pt", "role_probes.pt", "activations_cache.json",
                    "direction_validation.csv", "role_probe.csv"],
          requires=["labels"], needs_gpu=True,
          doc="E1.1 — directions + role probe at every layer"),
    Stage("nulls", stage_nulls, produces=["null_distributions.csv"], requires=["extract"],
          doc="E1.1 validation — random-direction nulls + length-matched reruns"),
    Stage("geometry", stage_geometry,
          produces=["geometry_cosines.csv", "geometry_null_band.json"], requires=["extract"],
          doc="E1.2 — pairwise cosines vs null band and split-half floor"),
    Stage("projections", stage_projections,
          produces=["projections.csv", "projection_correlations.csv"], requires=["extract"],
          doc="E1.3 — projections and correlations between them"),
    Stage("dimensionality", stage_dimensionality, produces=["dimensionality.csv"],
          requires=["extract"], doc="E1.4 — stratified directions -> r_eff"),
    Stage("harm_controls", stage_harm_controls,
          produces=["harm_controls.csv", "harm_controls_geometry.csv"],
          requires=["extract"],
          doc="E1.1 — R_harm with refusal held constant (PLAN-EXTRACT)"),
    Stage("dim_behavioural", stage_dimensionality_behavioural,
          produces=["dimensionality_behavioural.csv", "behavioural_k.json"],
          requires=["extract"], needs_gpu=True,
          doc="E1.4b — smallest k reproducing the full steering effect"),
    Stage("geometry_subspace", stage_geometry_subspace,
          produces=["geometry_subspace.csv", "geometry_subspace_summary.json"],
          requires=["extract", "dim_behavioural"],
          doc="E1.2 pass 2 — principal angles / projection / CCA at E1.4b's k"),
    Stage("emergence", stage_emergence, produces=["emergence_curves.csv", "emergence_summary.csv"], requires=["extract"],
          doc="E1.5 — separation vs relative depth"),
    Stage("style", stage_style,
          produces=["style_vs_metadata.csv", "style_summary.json",
                    "zhao_replication.csv"],
          requires=["extract"],
          doc="E1.7 Level 1 — metadata vs style; variation across prompt categories"),
    # No `requires`: this stage consumes the FROZEN style corpus and the model,
    # not any artifact of `extract`. Declaring a dependency it does not use would
    # be misleading and would block running it on its own.
    Stage("style_level2", stage_style_level2,
          produces=["style_level2.csv", "style_level2_summary.json"],
          needs_gpu=True,
          doc="E1.7 Level 2 — tag vs register, crossed on the frozen style corpus"),
    Stage("fidelity", stage_fidelity, produces=["fidelity.json"], requires=["extract"],
          needs_gpu=True,
          doc="E1.1 — pre_mlp fidelity + cross-corpus transfer (dropped in the first consolidation)"),
    Stage("causal", stage_causal,
          produces=[f"causal_matrix{CAUSAL_SUFFIX}.csv", f"causal_gate{CAUSAL_SUFFIX}.json",
                    f"causal_stage_b{CAUSAL_SUFFIX}.csv"],
          requires=["extract"], needs_gpu=True,
          doc="E1.6 — causal distinguishability; GO/NO-GO GATE 1"),
])


def main() -> None:
    ap = add_pipeline_args(argparse.ArgumentParser(description=__doc__))
    args = ap.parse_args()
    if maybe_list(PIPELINE, args):
        return
    cfg = load_config()
    log = get_logger(NAME, cfg.dir(NAME), suffix=f"__{'+'.join(cfg.models)}")
    from core.io_utils import write_run_manifest
    write_run_manifest(cfg, NAME)
    PIPELINE.run(cfg, log, only=args.only, start=args.start,
                 force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
