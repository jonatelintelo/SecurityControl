#!/usr/bin/env python
"""E1.1 — Recover and validate R_role, R_harm, R_control.

Playbook: EXPERIMENTS.md > RQ1 > E1.1.  Source: PEP-1, PART1.

Estimators are the source papers'; the substrate is ours. All three variables are
estimated on one crossed corpus (`instructions x {harmful, harmless} x roles`) so
they share a space and E1.2's geometry is interpretable — running Zhao on
instruction data and the role probe on webtext would leave every cross-concept
cosine confounded by that distribution gap.

Directions produced, per layer:

    R_harm          diff-of-means, t_inst,      harmful vs harmless   (role balanced)
    R_control       diff-of-means, t_post-inst, refused vs complied   (role balanced)
    R_control_pos   diff-of-means, t_post-inst, harmful vs harmless   (control condition)
    R_harm_user     diff-of-means, t_inst,      harmful vs harmless, user role only
    R_role          multiclass logistic probe over content tokens

`R_control_pos` is a control, not a rival estimate: since `R_harm` is the
harmful/harmless contrast at `t_inst`, the same contrast at `t_post-inst` may be
nothing but `R_harm` transported downstream. `cos(R_harm, R_control_pos)`
quantifies that, separating positional carry-over from the behavioural signal.

    python experiments/e1_1_directions.py
    FAST_DEV=1 MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct python experiments/e1_1_directions.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# MUST come before any import that pulls in torch — including our own core
# modules, which import it at module level. Shared login nodes cap `ulimit -u`
# (1900 here) and torch's default OMP pool (= core count) overruns it, dying in
# libgomp thread creation. Importing a helper from `core` to do this would be
# self-defeating: the import itself initialises torch first.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from core import extract, model_meta, pools  # noqa: E402
from core.capture import ActivationCapture  # noqa: E402
from core.config import Config, load_config  # noqa: E402
from core.io_utils import get_logger, save_df, save_json, save_torch  # noqa: E402
from core.model_io import load_model  # noqa: E402
from core.positions import content_token_spans, resolve_positions  # noqa: E402
from core.refusal import label_refusals  # noqa: E402

NAME = "e1_1"


def _mask(items, fn) -> torch.Tensor:
    return torch.tensor([bool(fn(it)) for it in items])


def run(cfg: Config) -> None:
    out = cfg.dir(NAME)
    log = get_logger(NAME, out)
    log.info("=" * 78)
    log.info("E1.1 — recover and validate the three latent variables")
    log.info("=" * 78)

    # ---------------------------------------------------------------- corpus
    items = pools.crossed_corpus(cfg.n_harmful, cfg.n_harmless, cfg.roles, seed=cfg.seed)
    train, test = pools.split_by_instruction(items, cfg.train_fraction, seed=cfg.seed)
    log.info(f"Crossed corpus: {len(items)} items "
             f"({cfg.n_harmful}+{cfg.n_harmless} instructions x {len(cfg.roles)} roles)")
    log.info(f"Split by instruction: {len(train)} train / {len(test)} test")
    if pools.SOURCE_FAILURES:
        log.warning(f"Optional dataset sources unavailable: {pools.SOURCE_FAILURES}")

    # ------------------------------------------------------------ model
    model, tok = load_model(cfg.model_id, cfg.device, logger=log)
    n_layers = model_meta.num_layers(model.config)
    d_model = model_meta.hidden_size(model.config)
    layers = cfg.layers(n_layers)

    positions = [resolve_positions(tok, it.instruction, role=it.role) for it in items]
    log.info(f"Positions resolved. Median prompt length: "
             f"{int(torch.tensor([p.n_tokens for p in positions]).median())} tokens")

    # ------------------------------------- behavioural labels for R_control
    # Zhao's refusal contrast is refuse-vs-accept by the model's ACTUAL response,
    # not by prompt harmfulness. This is the step our old forced "I"/"Sure"
    # scheme replaced with a lexical artefact.
    log.info(f"Labelling refuse/comply ({cfg.refusal_max_new_tokens} new tokens)...")
    lab = label_refusals(model, tok, [p.text for p in positions], cfg.device,
                         cfg.refusal_max_new_tokens, cfg.batch_size)
    refused = torch.tensor(lab["refused"])
    harmful = _mask(items, lambda it: it.harmful)
    log.info(f"Refusal rate: overall {refused.float().mean():.3f} | "
             f"harmful {refused[harmful].float().mean():.3f} | "
             f"harmless {refused[~harmful].float().mean():.3f}")
    save_df(out / "refusal_labels.csv", pd.DataFrame({
        "instruction": [it.instruction for it in items],
        "role": [it.role for it in items],
        "harmful": harmful.tolist(),
        "refused": lab["refused"],
        "response": lab["responses"],
    }))

    # ------------------------------------------------------------- capture
    cap = ActivationCapture(model, site="residual")
    log.info(f"Capturing residual stream at t_inst and t_post_inst, {n_layers} layers...")
    acts_inst = cap.at_positions(tok, positions, "t_inst", layers, cfg.batch_size, cfg.device)
    acts_post = cap.at_positions(tok, positions, "t_post_inst", layers, cfg.batch_size, cfg.device)

    train_uids = {t.uid for t in train}
    is_train = _mask(items, lambda it: it.uid in train_uids)
    is_user = _mask(items, lambda it: it.role == "user")

    # --------------------------------------------------------- directions
    specs = [
        ("R_harm", acts_inst, harmful, None, "t_inst", "harmful", "harmless"),
        ("R_control", acts_post, refused, None, "t_post_inst", "refused", "complied"),
        ("R_control_pos", acts_post, harmful, None, "t_post_inst", "harmful", "harmless"),
        ("R_harm_user", acts_inst, harmful, is_user, "t_inst", "harmful", "harmless"),
    ]

    directions, rows = {}, []
    for concept, acts, label, subset, pos_name, pos_cls, neg_cls in specs:
        keep = torch.ones_like(is_train) if subset is None else subset
        tr, te = is_train & keep, (~is_train) & keep
        if (label & tr).sum() < 2 or (~label & tr).sum() < 2:
            log.warning(f"{concept}: too few examples per class in train "
                        f"(pos={int((label & tr).sum())}, neg={int((~label & tr).sum())}) — skipped")
            continue

        directions[concept] = {}
        for li in layers:
            a = acts[li]
            d = extract.diff_of_means(a[label & tr], a[~label & tr], concept, li,
                                      "residual", pos_name, pos_cls, neg_cls)
            directions[concept][li] = d

            sep = extract.separation(d, a[label & te], a[~label & te])
            lo, hi = extract.bootstrap_auc_ci(d.project(a[label & te]),
                                              d.project(a[~label & te]), seed=cfg.seed)
            rnd = extract.random_direction_like(d, seed=cfg.seed + li)
            rsep = extract.separation(rnd, a[label & te], a[~label & te])
            rows.append({
                **d.to_record(), "auc": sep["auc"], "auc_ci_low": lo, "auc_ci_high": hi,
                "cohens_d": sep["cohens_d"], "random_auc": rsep["auc"],
                "n_test_pos": int((label & te).sum()), "n_test_neg": int((~label & te).sum()),
            })
            # Dimensionality is deliberately NOT computed here. `r_eff` of the raw
            # activations measures the width of the representation space, not of
            # the concept, and with two classes the between-class scatter is rank-1
            # by construction — both answer a different question from E1.4's
            # "single axis or subspace?". E1.4 will stratify (by role, source,
            # bootstrap), stack the resulting directions, and take the spectrum of
            # *that*, which is what actually measures concept dimensionality.
        log.info(f"{concept}: directions at {len(directions[concept])} layers")

    df = pd.DataFrame(rows)
    save_df(out / "direction_validation.csv", df)

    for concept in df["concept"].unique():
        sub = df[df["concept"] == concept]
        top = sub.loc[sub["auc"].idxmax()]
        l0 = sub[sub["layer"] == 0]["auc"]
        log.info(f"  {concept:<16} best AUC {top['auc']:.3f} "
                 f"[{top['auc_ci_low']:.2f},{top['auc_ci_high']:.2f}] @L{int(top['layer'])} | "
                 f"random {top['random_auc']:.3f} | layer0 {float(l0.iloc[0]) if len(l0) else float('nan'):.3f}")

    # ------------------------------------------------------- role probe
    probe_layers = cfg.probe_layers(n_layers)
    log.info(f"Role probe: token-level capture over {len(probe_layers)} layers, "
             f"<={cfg.max_content_tokens} content tokens/seq...")
    spans = [content_token_spans(tok, it.instruction, role=it.role) for it in items]
    tok_acts, owner = cap.at_content_tokens(tok, spans, probe_layers,
                                            cfg.max_content_tokens, cfg.batch_size,
                                            cfg.device, seed=cfg.seed)
    tok_role = [items[i].role for i in owner.tolist()]
    tok_train = torch.tensor([bool(is_train[i]) for i in owner.tolist()])
    log.info(f"Captured {len(owner)} content tokens "
             f"({int(tok_train.sum())} train / {int((~tok_train).sum())} test)")

    probes, probe_rows = {}, []
    for li in probe_layers:
        x = tok_acts[li]
        p = extract.train_role_probe(
            x[tok_train], [r for r, m in zip(tok_role, tok_train.tolist()) if m],
            x[~tok_train], [r for r, m in zip(tok_role, (~tok_train).tolist()) if m],
            layer=li, site="residual", seed=cfg.seed)
        probes[li] = p
        probe_rows.append({"layer": li, "accuracy": p.accuracy, "n_classes": len(p.classes),
                           "chance": 1.0 / len(p.classes), "n_train": p.n_train, "n_test": p.n_test})
        log.info(f"  L{li:<3} role-probe accuracy {p.accuracy:.3f} (chance {1/len(p.classes):.3f})")

    save_df(out / "role_probe_accuracy.csv", pd.DataFrame(probe_rows))

    # ------------------------------------------- cross-corpus transfer check
    # If a probe trained on instructions-in-role-tags cannot classify roles on the
    # role paper's constant-content webtext, it is reading "instruction-ness".
    log.info("Cross-corpus transfer: role paper's C4 constant-content design...")
    n_docs = 8 if cfg.fast_dev else 60
    ref_items = pools.role_paper_corpus(n_docs, cfg.roles, seed=cfg.seed)
    ref_items = [pools.CrossedItem(it.instruction[:500], it.harmful, it.role, it.source, it.uid)
                 for it in ref_items]
    ref_spans = [content_token_spans(tok, it.instruction, role=it.role) for it in ref_items]
    ref_acts, ref_owner = cap.at_content_tokens(tok, ref_spans, probe_layers,
                                                cfg.max_content_tokens, cfg.batch_size,
                                                cfg.device, seed=cfg.seed)
    ref_role = [ref_items[i].role for i in ref_owner.tolist()]
    transfer = [{"layer": li,
                 "crossed_to_webtext": extract.probe_transfer_accuracy(probes[li], ref_acts[li], ref_role),
                 "in_corpus": probes[li].accuracy,
                 "chance": 1.0 / len(probes[li].classes)}
                for li in probe_layers]
    save_df(out / "role_probe_transfer.csv", pd.DataFrame(transfer))
    best_t = max(transfer, key=lambda r: r["crossed_to_webtext"])
    log.info(f"  best transfer {best_t['crossed_to_webtext']:.3f} @L{best_t['layer']} "
             f"(in-corpus {best_t['in_corpus']:.3f}, chance {best_t['chance']:.3f})")

    # ------------------------------------------------------------- artifacts
    save_torch(out / "directions.pt", directions)
    save_torch(out / "role_probes.pt", probes)
    save_json(out / "null_band.json", extract.random_cosine_band(d_model, seed=cfg.seed))
    save_json(out / "corpus.json", {
        "n_items": len(items), "n_train": len(train), "n_test": len(test),
        "roles": list(cfg.roles), "source_failures": pools.SOURCE_FAILURES,
        "model": model_meta.describe(cfg.model_id, model.config),
        "refusal_rate_overall": float(refused.float().mean()),
        "refusal_rate_harmful": float(refused[harmful].float().mean()),
        "refusal_rate_harmless": float(refused[~harmful].float().mean()),
    })
    log.info(f"E1.1 complete -> {out}")


if __name__ == "__main__":
    from core.io_utils import write_run_manifest
    cfg = load_config()
    write_run_manifest(cfg, NAME)
    run(cfg)
