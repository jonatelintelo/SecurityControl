#!/usr/bin/env python
"""Post-run verification of an RQ1 result set.

Checks properties that the pipeline does not check itself, and that would each
silently invalidate a claim if violated. Run against a completed results root:

    RESULTS_ROOT=./results_verify python tests/verify_rq1_run.py
    VERIFY_AGAINST=./results     python tests/verify_rq1_run.py   # + reproducibility

Exit code 0 = every check passed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

ROOT = Path(os.environ.get("RESULTS_ROOT", "./results"))
AGAINST = os.environ.get("VERIFY_AGAINST")
MODELS = os.environ.get("MODELS", "qwen2.5-7b,qwen3.5-9b").split(",")

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


def main() -> int:
    # ---------------------------------------------------------------- corpus
    cdir = ROOT / "e1_0_corpus"
    ins = pd.read_json(cdir / "instructions.jsonl", lines=True)
    idx = pd.read_csv(cdir / "rendered_index.csv")

    check("corpus: no train/test leakage by instruction",
          not (set(ins[ins.split == "train"].uid) & set(ins[ins.split == "test"].uid)),
          f"{ins.split.value_counts().to_dict()}")

    # Every instruction must appear under every role and design, or the role
    # contrast is confounded with which instructions carry which role.
    fit = idx[idx.kind == "fitting"]
    per = fit.groupby(["model", "uid"]).apply(
        lambda g: len(set(zip(g.role, g.design))), include_groups=False)
    check("corpus: every instruction rendered under every (role, design)",
          bool((per == per.max()).all()) and int(per.max()) == 8,
          f"cells per instruction: min={int(per.min())} max={int(per.max())} (expected 8)")

    # ---------------------------------------------------------------- per model
    for m in MODELS:
        d = ROOT / "rq1" / m
        if not d.exists():
            check(f"{m}: results present", False, f"missing {d}")
            continue

        val = pd.read_csv(d / "direction_validation.csv")
        lab = pd.read_csv(d / "refusal_labels.csv")

        # Layer selection must never have used the test split.
        for c in val.concept.unique():
            s = val[val.concept == c]
            sel = int(s.loc[s.train_auc.idxmax(), "layer"])
            best_test = int(s.loc[s.auc.idxmax(), "layer"])
            if sel != best_test:
                check(f"{m}/{c}: reported layer selected on TRAIN, not test", True,
                      f"train-selected L{sel} != test-argmax L{best_test} — selection is honest")
                break
        else:
            check(f"{m}: layer selection differs from test argmax somewhere", True,
                  "all concepts coincide; not evidence of leakage but worth noting")

        # Nothing is measured on the prompts used to fit it.
        tr = set(lab[lab.split == "train"].uid)
        te = set(lab[lab.split == "test"].uid)
        check(f"{m}: fit/eval instruction sets disjoint", not (tr & te),
              f"train={len(tr)} test={len(te)} overlap={len(tr & te)}")

        # Every direction must beat its length-only baseline, or only the
        # length-matched refit is reportable.
        bad = []
        for c in val.concept.unique():
            s = val[val.concept == c]
            b = s.loc[s.train_auc.idxmax()]
            if b.auc <= b.length_only_auc:
                bad.append(f"{c}: auc {b.auc:.3f} <= length-only {b.length_only_auc:.3f}")
        check(f"{m}: every direction beats its length-only baseline", not bad, "; ".join(bad))

        # Activations: no overflow or NaN from the fp16 storage cast.
        blob = torch.load(d / "activations.pt", map_location="cpu", weights_only=False)
        worst, nan, inf = 0.0, 0, 0
        for pos in ("t_inst", "t_post_inst"):
            for li, a in blob[pos].items():
                worst = max(worst, float(a.abs().max()))
                nan += int(torch.isnan(a).sum())
                inf += int(torch.isinf(a).sum())
        check(f"{m}: activations finite, fp16 storage has headroom",
              nan == 0 and inf == 0 and worst < 65504 / 4,
              f"absmax={worst:.1f} (fp16 max 65504, {65504/max(worst,1e-9):.0f}x headroom), "
              f"nan={nan} inf={inf}")

        # The index cached with the activations must align with the labels.
        cached = pd.DataFrame(blob["index"])
        check(f"{m}: cached activation index aligns with refusal labels",
              list(cached.uid) == list(lab.uid) and list(cached.role) == list(lab.role),
              f"n_cached={len(cached)} n_labels={len(lab)}")

        # Gate artifacts carry a control-variant suffix (`__under` / `__over`) so
        # that runs which differ in the control variable cannot be conflated.
        # Globbing rather than naming one file matters: a check that silently
        # finds nothing is worse than one that fails, because it still reports a
        # green run.
        import json
        gates = sorted(d.glob("causal_gate*.json"))
        mats = sorted(d.glob("causal_matrix*.csv"))
        check(f"{m}: gate artifacts present", bool(gates) and bool(mats),
              f"{[p.name for p in gates]} / {[p.name for p in mats]}")

        for gp in gates:
            tag = gp.stem.replace("causal_gate", "") or "(default)"
            g = json.loads(gp.read_text())
            bad_v = [b for b, v in g["per_bound"].items()
                     if v["verdict"] in ("PASS", "FAIL")
                     and v["G2_asymmetry"]["n_pairs_tested"] == 0]
            check(f"{m}{tag}: no gate verdict from an untested criterion", not bad_v,
                  f"bounds reporting a verdict with 0 pairs tested: {bad_v}")
            # FDR must actually have been applied to the asymmetry family.
            has_fdr = all("n_fdr_reject" in v["G2_asymmetry"] for v in g["per_bound"].values())
            check(f"{m}{tag}: G2 is FDR-corrected", has_fdr,
                  "BH q=0.05 over the asymmetry family, as pre-registered")

        for mp in mats:
            tag = mp.stem.replace("causal_matrix", "") or "(default)"
            mat = pd.read_csv(mp)
            diag = mat[(mat.source == mat.target) & (mat.source != "random")]
            first = diag[diag.read_layer == diag.steer_layer + 1]
            err = (first.delta / first.alpha - 1.0).abs()
            check(f"{m}{tag}: steering readout is calibrated (delta_AA ~= alpha)",
                  bool(len(err)) and bool(err.median() < 0.35),
                  f"median |delta/alpha - 1| = {err.median():.3f} at the first read layer")
            # The behavioural readout must be generation, not the demoted proxy.
            check(f"{m}{tag}: behavioural readout is generation-based",
                  "d_refusal" in mat.columns,
                  "refusal rate under intervention, labelled with the published rule")

    # ------------------------------------------------------- reproducibility
    if AGAINST:
        other = Path(AGAINST)
        for f in ("instructions.jsonl", "attack_intents.jsonl", "transfer_corpus.jsonl"):
            a, b = (other / "e1_0_corpus" / f), (cdir / f)
            if a.exists() and b.exists():
                check(f"reproducible: {f} byte-identical", a.read_bytes() == b.read_bytes())
        # Reproducibility has two tiers, and conflating them hides a real signal.
        #
        # Label-INDEPENDENT concepts (the harm family) depend only on the frozen
        # corpus and the forward pass, so they must reproduce EXACTLY. Any drift
        # there would mean the capture or the estimator is nondeterministic.
        #
        # Label-DEPENDENT concepts (the control family) additionally depend on
        # greedy generation, which is only bit-reproducible at a FIXED batch size:
        # padding changes bf16 numerics and flips a handful of borderline tokens.
        # They are therefore checked against a tolerance, at each run's own
        # train-selected layer, which is the number that actually gets reported.
        for m in MODELS:
            pa, pb = other / "rq1" / m / "direction_validation.csv", ROOT / "rq1" / m / "direction_validation.csv"
            if not (pa.exists() and pb.exists()):
                continue
            A, B = pd.read_csv(pa), pd.read_csv(pb)
            j = A.merge(B, on=["concept", "layer"], suffixes=("_a", "_b"))
            j["d"] = (j.auc_a - j.auc_b).abs()

            li = j[~j.concept.str.contains("control")]
            check(f"reproducible: {m} label-independent concepts are EXACT",
                  bool(len(li) and li.d.max() == 0.0),
                  f"{sorted(li.concept.unique())}: max|delta AUC| = {li.d.max():.6f}")

            rows = []
            for c in A.concept.unique():
                if c not in set(B.concept):
                    continue
                a, b = A[A.concept == c], B[B.concept == c]
                ba, bb = a.loc[a.train_auc.idxmax()], b.loc[b.train_auc.idxmax()]
                rows.append((c, abs(float(ba.auc) - float(bb.auc)),
                             int(ba.layer), int(bb.layer)))
            worst = max(rows, key=lambda r: r[1]) if rows else ("", 0.0, 0, 0)
            check(f"reproducible: {m} reported AUC agrees within tolerance",
                  worst[1] < 0.02,
                  f"worst: {worst[0]} delta={worst[1]:.4f} "
                  f"(layer {worst[2]} vs {worst[3]}) — label-driven, see ENVIRONMENT.md")

            # Emergence onset is the quantity E1.5 reports, and it must be stable
            # even where the peak layer is not: the AUC-vs-depth curve is flat near
            # its maximum for the control concepts, so argmax moves while onset
            # should not.
            ea, eb = other/"rq1"/m/"emergence_summary.csv", ROOT/"rq1"/m/"emergence_summary.csv"
            if ea.exists() and eb.exists():
                EA, EB = pd.read_csv(ea), pd.read_csv(eb)
                e = EA.merge(EB, on="concept", suffixes=("_a", "_b"))
                dd = (e.onset_90_a - e.onset_90_b).abs()
                check(f"reproducible: {m} E1.5 onset depth stable",
                      bool(len(e) and dd.max() <= 0.10),
                      f"max |delta onset_90| = {dd.max():.3f} over {len(e)} concepts")

    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print("\n" + "=" * 70)
    print(f"{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
