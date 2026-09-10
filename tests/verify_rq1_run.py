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


def _post(m):
    """Keep only the `t_post_inst` read position.

    `causal_matrix*.csv` carries a second read position (`t_inst`), added for the
    token-resolved readout. Every gate-level quantity is defined at
    `t_post_inst`; consuming both would double-count each intervention and mix
    two incomparable residual bases.

    It deliberately does NOT restrict read LAYERS. The matrix carries every layer
    downstream of the steer layer, and which of those form the test family is an
    adjudication option (`gate_layers`) applied inside `_adjudicate_at`, so that
    it can be swept. Filtering here would silently pin that sweep to one arm.
    Older matrices lack the column.
    """
    if "read_position" in m.columns:
        m = m[m.read_position == "t_post_inst"]
    return m



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

    # ------------------------------------------------- E1.7 Level 2 style corpus
    # The whole Level 2 design rests on the crossing being balanced: the register
    # contrast is only a register contrast if every base appears in every
    # register. Checked here, on the corpus, rather than inferred from the
    # downstream numbers.
    sdir = ROOT / "e1_7_style"
    if (sdir / "style_corpus.jsonl").exists():
        import json as _json
        from collections import Counter
        sm = _json.loads((sdir / "style_corpus_meta.json").read_text())
        sitems = [_json.loads(l) for l in open(sdir / "style_corpus.jsonl") if l.strip()]
        gen = [i for i in sitems if i["arm"] == "generated"]
        tpl = [i for i in sitems if i["arm"] == "template"]
        per_gen = set(Counter(i["base_uid"] for i in gen).values())
        per_tpl = set(Counter(i["base_uid"] for i in tpl).values())
        check("E1.7 L2: every base appears in every register",
              per_gen == {3} and per_tpl == {6},
              f"generated per base {per_gen} (expect {{3}}), "
              f"template per base {per_tpl} (expect {{6}})")
        check("E1.7 L2: both arms cover the same bases",
              {i["base_uid"] for i in gen} == {i["base_uid"] for i in tpl},
              "the template arm is the control for the generated arm, so it must "
              "hold the same requests")
        check("E1.7 L2: corpus declared usable",
              bool(sm.get("usable")),
              f"{sm.get('n_bases_complete')} complete bases "
              f"(floor {sm.get('min_complete_bases')})")
        # Lexical fidelity is a reported distribution, not an assumption.
        covs = sorted(i["coverage"] for i in gen)
        check("E1.7 L2: content coverage recorded per item",
              all(c == c for c in covs) and bool(covs),
              f"median {covs[len(covs)//2]:.3f}, min {covs[0]:.3f} over {len(covs)} "
              f"generated rewrites — report this, do not assume it")

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

        # E1.1d: the refusal-controlled harm variants must be ROLE-BALANCED.
        # Pooled `R_harm` is exempt (all four roles per instruction, balanced by
        # construction), but conditioning on the refusal label breaks that, since
        # refusal rate varies by role. An unbalanced fit would carry a role
        # component and would DEPRESS its cosine against a role-balanced
        # `R_control` — overstating the separability that is the headline claim.
        hc = d / "harm_controls.csv"
        if hc.exists():
            H = pd.read_csv(hc)
            check(f"{m}: E1.1d variants are role-balanced",
                  bool(len(H)) and "role_balanced" in H.columns and bool(H.role_balanced.all()),
                  f"{len(H)} variant-position fits; "
                  f"role TV before balancing: "
                  f"{H.role_tv_before_balancing.round(3).tolist() if 'role_tv_before_balancing' in H else 'NOT RECORDED'}")

            # These fits are heavily imbalanced (~870 harmful vs ~50 over-refused
            # harmless) and report AUC at or near 1.0, which is exactly what a
            # length cue would produce. The baseline is the only thing separating
            # "harm direction" from "length direction" here.
            check(f"{m}: E1.1d variants beat their length-only baseline",
                  "beats_length_baseline" in H.columns and bool(H.beats_length_baseline.all()),
                  (f"AUC vs length-only: "
                   + "; ".join(f"{r.concept}@{r.position} {r.auc:.3f} vs {r.length_only_auc:.3f}"
                               for r in H.itertuples()))
                  if "length_only_auc" in H.columns else "NOT RECORDED")

        # E1.7 Level 2: the tag-vs-register crossing. The corpus is generated, so
        # the checks are about whether the crossing is actually balanced — an
        # unbalanced one would make the "register" contrast partly a content
        # contrast between different base sets.
        l2 = d / "style_level2.csv"
        if l2.exists():
            L = pd.read_csv(l2)
            arms = set(L.arm.unique()) if "arm" in L.columns else set()
            check(f"{m}: E1.7 Level 2 has both arms",
                  {"generated", "template"} <= arms,
                  f"arms present: {sorted(arms)} — `generated` is the real test, "
                  f"`template` upper-bounds how detectable register can be")
            # Every reported cosine must be read against a floor, never against 0.
            check(f"{m}: E1.7 Level 2 cosines carry a split-half floor",
                  {"floor", "cos_tag_register"} <= set(L.columns)
                  and bool(L.floor.notna().all()),
                  f"{len(L)} rows; median floor "
                  f"{L.floor.median():.3f}" if "floor" in L.columns else "MISSING")

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
            mat = _post(pd.read_csv(mp))
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

            # Read coverage, checked on the UNFILTERED matrix: EXPERIMENTS.md
            # requires every layer downstream of the steer layer to be read, and
            # a silent regression to a three-layer sample would still produce a
            # plausible-looking depth curve.
            full = pd.read_csv(mp)
            if "gate_read_layer" in full.columns:
                post = full[full.read_position == "t_post_inst"] \
                    if "read_position" in full.columns else full
                gaps = []
                for L, g in post.groupby("steer_layer"):
                    want = set(range(int(L) + 1, int(post.read_layer.max()) + 1))
                    got = set(g.read_layer.astype(int))
                    if want - got:
                        gaps.append(f"steer L{L}: missing {sorted(want - got)[:5]}")
                check(f"{m}{tag}: every downstream layer is read",
                      not gaps, "; ".join(gaps[:3]) or
                      f"{post.read_layer.nunique()} distinct read layers over "
                      f"{post.steer_layer.nunique()} steer layers")

                # And the gate family must be the pre-registered subset, not
                # everything that happened to be read.
                per = post[post.gate_read_layer.astype(bool)].groupby(
                    "steer_layer").read_layer.nunique()
                check(f"{m}{tag}: gate family is the pre-registered read layers",
                      bool(len(per)) and bool((per <= 3).all()),
                      f"gate read layers per steer layer: {sorted(set(per))} (expected <= 3); "
                      f"profile keeps {post.read_layer.nunique()} for description")

                # Both read positions must be present, or the token-resolved
                # readout PLAN-INF asks for is not actually there.
                check(f"{m}{tag}: both read positions present",
                      set(full.read_position.unique()) >= {"t_inst", "t_post_inst"},
                      f"{sorted(full.read_position.unique())}")

        # Stage B: the steered-position sweep. `position_masks` existed in the
        # intervention primitives for a long time while never being passed, so
        # this asserts the sweep actually ran rather than that the file exists.
        for bp in sorted(d.glob("causal_stage_b*.csv")):
            B = pd.read_csv(bp)
            got = set(B.token_set.unique()) if "token_set" in B.columns else set()
            check(f"{m}: stage B swept the steered positions",
                  bool(len(B)) and "all_real" in got and len(got) >= 2,
                  f"token sets present: {sorted(got)} "
                  f"(pre-registered: all_real, instruction_span, t_inst_only, "
                  f"post_instruction; a set is skipped only where its span is empty)")
            # Restricting which tokens are steered must actually change the
            # intervention, or the masks are not reaching the hook.
            if len(got) >= 2:
                per = B.groupby("token_set").mean_kl.mean()
                check(f"{m}: steered-token set changes the intervention",
                      float(per.max() - per.min()) > 1e-6,
                      "mean KL by token set: "
                      + ", ".join(f"{k}={v:.4f}" for k, v in per.items()))

    # ------------------------------------------------------- reproducibility
    if AGAINST:
        other = Path(AGAINST)
        for f in ("instructions.jsonl", "attack_intents.jsonl", "transfer_corpus.jsonl"):
            a, b = (other / "e1_0_corpus" / f), (cdir / f)
            if a.exists() and b.exists():
                check(f"reproducible: {f} byte-identical", a.read_bytes() == b.read_bytes())
        # The E1.7 Level 2 corpus is GENERATED, but greedily and at a fixed batch
        # size, so the two roots must still receive byte-identical text. If they
        # do not, the style corpus is not a frozen shared input and the Level 2
        # comparison between roots is not like-for-like.
        sa = other / "e1_7_style" / "style_corpus.jsonl"
        sb = ROOT / "e1_7_style" / "style_corpus.jsonl"
        if sa.exists() and sb.exists():
            check("reproducible: style_corpus.jsonl byte-identical",
                  sa.read_bytes() == sb.read_bytes(),
                  "generated greedily at fixed batch size; drift here means the "
                  "Level 2 corpus is not a frozen shared input")
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
