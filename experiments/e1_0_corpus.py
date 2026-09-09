#!/usr/bin/env python
"""E1.0 — build, verify and freeze the corpus (+ E1.0b transfer corpus).

Spec: EXPERIMENTS.md > RQ1 > E1.0.

No model weights, no GPU: everything here is checkable offline, and every
downstream experiment inherits it, so it is verified first and independently.

  * instruction pools, source-balanced round-robin, `source` and `category`
    recorded as factors;
  * a train/test split **by instruction**, stratified by (label, source), so no
    instruction's role renderings straddle it;
  * an attack pool disjoint **by construction**, not checked afterwards;
  * the **E1.0b transfer corpus** — C4 prose under the same role tags, for E1.1's
    cross-corpus transfer criterion;
  * per-model rendering and position resolution for every (role, design), swept
    **corpus-wide**, emitting the per-item length index the length-only baselines
    need.

The corpus produced here is a **candidate**. It is frozen only after E1.1 Stage 1
labels it on both models and the harmful-and-complied cell is shown adequate on
each — cell size is a property of a model's refusal behaviour, so freezing before
labelling would mean widening for one model silently changes the other's corpus.

    python experiments/e1_0_corpus.py
    FAST_DEV=1 python experiments/e1_0_corpus.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Before any import that pulls in torch — shared login nodes cap `ulimit -u` and
# torch's default OMP pool overruns it. A helper cannot do this from inside a
# module that imports torch, because the import initialises torch first.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Dict, List, Sequence, Tuple  # noqa: E402

import pandas as pd  # noqa: E402

from core import pools  # noqa: E402
from core.config import Config, load_config  # noqa: E402
from core.io_utils import get_logger, save_df, save_json  # noqa: E402
from core.positions import describe, render  # noqa: E402

NAME = "e1_0_corpus"

# The cross-role tokenisation invariant is *measured*, never asserted, and
# affected items are always excluded from token-matched analyses regardless of
# rate. This bound is not an acceptance threshold — it is the point at which the
# corpus design itself is wrong and needs rethinking rather than filtering.
MISMATCH_DESIGN_LIMIT = 0.05


def _sweep(cfg: Config, items: Sequence, kind: str, log) -> Tuple[pd.DataFrame, Dict, pd.DataFrame]:
    """Render every item under every role and design, on every model.

    Two things are established here that cannot be established on a sample. First,
    that position resolution is exact for every item — a single bad `t_inst` reads
    the wrong token and produces a plausible number. Second, the **per-item token
    length**, which is the input to the length-only baselines: the authentic markup
    differs by role (the `<tool_response>` wrapper, Qwen3.5's empty think block),
    so length is a live confound for any direction fitted across role classes.
    """
    from transformers import AutoTokenizer

    index_rows: List[Dict] = []
    checks: Dict[str, Dict] = {}
    mismatch_rows: List[Dict] = []

    for slug in cfg.models:
        spec = cfg.spec(slug)
        tok = AutoTokenizer.from_pretrained(spec.model_id)
        if not tok.is_fast:
            raise RuntimeError(f"{spec.model_id} has no fast tokenizer; offset mapping is required")

        n_render, errors, bad_inst, bad_span, mism = 0, [], 0, 0, 0
        for ins in items:
            for design in cfg.designs:
                content_lens = {}
                for role in cfg.roles:
                    try:
                        R = render(tok, ins.text, role, design)
                    except Exception as e:
                        errors.append(f"{ins.uid}/{role}/{design}: {type(e).__name__}: {str(e)[:60]}")
                        continue
                    n_render += 1
                    ids = tok(R.text, add_special_tokens=False)["input_ids"]

                    # `t_inst` must be the last token of the instruction itself.
                    tail = ins.text.rstrip()[-40:]
                    if not tok.decode(ids[: R.t_inst + 1]).rstrip().endswith(tail):
                        bad_inst += 1
                    # The content span must decode back to exactly the instruction.
                    if tok.decode(ids[R.content_start:R.content_end]).strip() != ins.text.strip():
                        bad_span += 1

                    content_lens[role] = R.content_end - R.content_start
                    index_rows.append({
                        "model": slug, "kind": kind, "uid": ins.uid, "source": ins.source,
                        "harmful": ins.harmful, "split": ins.split, "role": role,
                        "design": design, "n_tokens": R.n_tokens,
                        "content_tokens": R.content_end - R.content_start,
                        "t_inst": R.t_inst, "t_post_inst": R.t_post_inst,
                        "slot_is_fixed": R.slot_is_fixed,
                    })

                if len(set(content_lens.values())) > 1:
                    mism += 1
                    mismatch_rows.append({"model": slug, "kind": kind, "uid": ins.uid,
                                          "source": ins.source, "design": design,
                                          **content_lens, "text": ins.text[:120]})

        n_pairs = len(items) * len(cfg.designs)
        rate = mism / n_pairs if n_pairs else 0.0
        checks[f"render_{kind}_{slug}"] = {
            "ok": not errors and bad_inst == 0 and bad_span == 0 and rate <= MISMATCH_DESIGN_LIMIT,
            "renders": n_render, "errors": len(errors),
            "bad_t_inst": bad_inst, "bad_content_span": bad_span,
            "cross_role_length_mismatch": mism, "mismatch_rate": round(rate, 4),
            "design_limit": MISMATCH_DESIGN_LIMIT,
            "note": "affected uids are excluded from token-matched analyses regardless of rate",
            "example_errors": errors[:3]}
        log.info(f"  {slug}/{kind}: {n_render} renders, {len(errors)} errors, "
                 f"bad_t_inst={bad_inst}, bad_span={bad_span}, "
                 f"cross-role length mismatch={mism}/{n_pairs} ({rate:.2%})")

    return pd.DataFrame(index_rows), checks, pd.DataFrame(mismatch_rows)


def _length_confound_report(index: pd.DataFrame, log) -> pd.DataFrame:
    """What the length-only baselines will be up against.

    Reported, not gated. Any length asymmetry here is legitimate — it comes from
    authentic markup and from the sources themselves — but it means every direction
    fitted across these classes must beat a classifier fitted on length alone.
    """
    rows: List[Dict] = []
    for (model, design), g in index[index.kind == "fitting"].groupby(["model", "design"]):
        by_role = g.groupby("role").n_tokens.median()
        h = g[g.harmful].n_tokens.median()
        n = g[~g.harmful].n_tokens.median()
        rows.append({
            "model": model, "design": design,
            "role_median_spread": int(by_role.max() - by_role.min()),
            **{f"median_{r}": int(v) for r, v in by_role.items()},
            "median_harmful": int(h), "median_harmless": int(n),
            "harm_length_ratio": round(float(h / n), 3) if n else None,
        })
    df = pd.DataFrame(rows)
    for _, r in df.iterrows():
        log.info(f"  {r.model}/{r.design}: role median spread {r.role_median_spread} tok, "
                 f"harmful/harmless median ratio {r.harm_length_ratio} "
                 f"-> length-only baseline required for role AND harm")
    return df


def run(cfg: Config) -> None:
    out = cfg.dir(NAME)                      # model-independent: no slug
    log = get_logger(NAME, out)
    log.info("=" * 78)
    log.info("E1.0 — build, verify and freeze the corpus (candidate until E1.1 Stage 1)")
    log.info("=" * 78)

    # ------------------------------------------------------------- build
    log.info(f"Building: {cfg.n_harmful} harmful + {cfg.n_harmless} harmless, "
             f"{cfg.n_attack} attack intents, seed={cfg.seed}")
    fitting, attacks = pools.build_corpus(
        cfg.n_harmful, cfg.n_harmless, cfg.n_attack, cfg.train_fraction, cfg.seed)

    harmful = [i for i in fitting if i.harmful]
    harmless = [i for i in fitting if not i.harmful]
    log.info(f"  harmful  {len(harmful):4d}  {pools.describe_pool(harmful)}")
    log.info(f"  harmless {len(harmless):4d}  {pools.describe_pool(harmless)}")
    log.info(f"  attacks  {len(attacks):4d}  {pools.describe_pool(attacks)}")

    log.info(f"Building E1.0b transfer corpus: {cfg.n_transfer} C4 passages")
    transfer = pools.transfer_corpus(
        n=cfg.n_transfer, seed=cfg.seed, train_fraction=cfg.train_fraction)
    log.info(f"  transfer {len(transfer):4d}  {pools.describe_pool(transfer)}")

    if pools.SOURCE_FAILURES:
        log.warning(f"  dataset sources that failed to load: {pools.SOURCE_FAILURES}")

    n_items = len(fitting) * len(cfg.roles) * len(cfg.designs)
    log.info(f"Crossing: {len(fitting)} instructions x {len(cfg.roles)} roles "
             f"x {len(cfg.designs)} designs = {n_items} rendered items per model")

    # ------------------------------------------------------------ verify
    checks = pools.verify_corpus(fitting, attacks)
    checks.update(pools.verify_transfer_corpus(transfer, fitting))

    log.info("-" * 78)
    failed = []
    for name, res in checks.items():
        ok = bool(res.get("ok"))
        detail = {k: v for k, v in res.items() if k != "ok"}
        log.info(f"  [{'ok  ' if ok else 'FAIL'}] {name}: {detail}")
        if not ok:
            failed.append(name)

    # --------------------------------------------------- per-model render
    log.info("-" * 78)
    log.info("Corpus-wide rendering sweep (tokenizer only, no weights):")
    idx_fit, ck_fit, mm_fit = _sweep(cfg, fitting, "fitting", log)
    idx_tr, ck_tr, mm_tr = _sweep(cfg, transfer, "transfer", log)

    index = pd.concat([idx_fit, idx_tr], ignore_index=True)
    mismatches = pd.concat([mm_fit, mm_tr], ignore_index=True) if len(mm_fit) or len(mm_tr) \
        else pd.DataFrame(columns=["model", "kind", "uid", "source", "design"])
    checks.update(ck_fit)
    checks.update(ck_tr)
    for name in list(ck_fit) + list(ck_tr):
        if not checks[name].get("ok"):
            failed.append(name)

    log.info("-" * 78)
    log.info("Length confound (reported, not gated):")
    length_df = _length_confound_report(index, log)

    # Per-role frame summary, for the record.
    from transformers import AutoTokenizer
    frames = []
    for slug in cfg.models:
        tok = AutoTokenizer.from_pretrained(cfg.spec(slug).model_id)
        frames += [{"model": slug, **r} for r in describe(tok)]
    render_df = pd.DataFrame(frames)

    # `slot_is_fixed` must be False exactly for `system` under fixed_slot, because
    # Qwen3.5 forces a system message to position 0. Anywhere else it means the
    # fixed-slot design silently fell back and the position control is not what the
    # design claims.
    ff = index[(index.design == "fixed_slot") & (~index.slot_is_fixed)]
    unexpected = sorted(set(ff.role.unique()) - {"system"})
    checks["fixed_slot_fallbacks_are_system_only"] = {
        "ok": not unexpected, "unexpected_roles": unexpected,
        "n_fallback_rows": int(len(ff))}
    if unexpected:
        failed.append("fixed_slot_fallbacks_are_system_only")

    # ------------------------------------------------------------ freeze
    meta = {
        "status": "candidate — frozen only after E1.1 Stage 1 labels both models",
        "seed": cfg.seed,
        "n_harmful": len(harmful), "n_harmless": len(harmless),
        "n_attack": len(attacks), "n_transfer": len(transfer),
        "train_fraction": cfg.train_fraction,
        "roles": list(cfg.roles), "designs": list(cfg.designs),
        "rendered_items_per_model": n_items,
        "source_composition": {
            "harmful": pools.describe_pool(harmful),
            "harmless": pools.describe_pool(harmless),
            "attack": pools.describe_pool(attacks),
            "transfer": pools.describe_pool(transfer),
        },
        "source_failures": pools.SOURCE_FAILURES,
        "source_provenance": pools.SOURCE_PROVENANCE,
        "checks": checks,
        "models_checked": list(cfg.models),
        "fast_dev": cfg.fast_dev,
    }
    pools.save_corpus(out, fitting, attacks, meta, transfer=transfer)
    save_df(out / "rendering_report.csv", render_df)
    save_df(out / "rendered_index.csv", index)
    save_df(out / "tokenisation_mismatches.csv", mismatches)
    save_df(out / "length_confound.csv", length_df)
    save_json(out / "verification.json", checks)

    log.info("-" * 78)
    if failed:
        log.error(f"E1.0 FAILED {len(failed)} check(s): {failed}")
        log.error("Artifacts are written anyway so the failure is inspectable, "
                  "but downstream stages must not consume this corpus.")
        raise SystemExit(1)
    log.info(f"E1.0 complete — {len(checks)} checks pass, corpus written to {out}")
    log.info("Status: CANDIDATE. E1.1 Stage 1 labels it on both models; the freeze "
             "follows only if the harmful-and-complied cell is adequate on each.")


if __name__ == "__main__":
    from core.io_utils import write_run_manifest
    cfg = load_config()
    write_run_manifest(cfg, NAME)
    run(cfg)
