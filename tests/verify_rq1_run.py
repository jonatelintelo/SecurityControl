#!/usr/bin/env python
"""Post-run verification of an RQ1 result set.

Checks properties that the pipeline does not check itself, and that would each
silently invalidate a claim if violated. Run against a completed results root:

    RESULTS_ROOT=./results_verify python tests/verify_rq1_run.py
    VERIFY_AGAINST=./results     python tests/verify_rq1_run.py   # + reproducibility

Exit code 0 = every check passed.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from core.config import (MATCHED_CONTROL_VARIANT, MODELS as MODEL_SPECS,  # noqa: E402
                         RQ1_MODELS, roster_table)


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
MODELS = [m for m in os.environ.get("MODELS", ",".join(RQ1_MODELS)).split(",") if m]

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
    # The expected cell count is |roles| x |designs| PER MODEL, read off that
    # model's own renderings — not the constant 8. `tool` exists in few chat
    # templates, so a cross-family model may legitimately carry three role
    # classes; hardcoding four would fail such a model for being what it is,
    # while still not catching an incomplete crossing on a four-role one.
    # Comparing the count of distinct (role, design) pairs against the product
    # of that model's marginals is the actual completeness test, and it is
    # strictly stronger: it fails if ANY cell is missing, at any width.
    fit = idx[idx.kind == "fitting"]
    ok, det = True, []
    for m, g in fit.groupby("model"):
        want = g.role.nunique() * g.design.nunique()
        per = g.groupby("uid").apply(
            lambda h: len(set(zip(h.role, h.design))), include_groups=False)
        good = bool((per == want).all())
        ok &= good
        det.append(f"{m}: {g.role.nunique()} roles ({','.join(sorted(g.role.unique()))}) "
                   f"x {g.design.nunique()} designs = {want}; "
                   f"per instruction min={int(per.min())} max={int(per.max())}"
                   + ("" if good else "  <-- INCOMPLETE"))
    check("corpus: every instruction rendered under every (role, design)",
          ok, "\n         ".join(det))

    # A three-role model cannot express the `tool` class, and `tool` is the
    # stand-in for the plan's *untrusted external content* role (D1). That is a
    # scope limit on what R_role means for such a model, so it is surfaced by
    # the verifier rather than left for a reader to notice.
    no_tool = sorted({m for m, g in fit.groupby("model") if "tool" not in set(g.role)})
    if no_tool:
        check("corpus: models without the `tool` role are flagged (D1 scope limit)",
              True,
              f"{', '.join(no_tool)} render no `tool` role. For these, R_role is an "
              f"authority/speaker-identity variable and does NOT cover the "
              f"untrusted-external-content class. Must be stated in the write-up.")

    # ------------------------------------------------------------- roster
    # The cross-family claim is only as good as the roster that was actually
    # run. Checking it against the registry — rather than trusting the job
    # scripts — catches the case where a model silently failed and the run
    # still produced a complete-looking result set for the survivors.
    print("\nRoster under verification:")
    print(roster_table())
    ran = sorted(set(idx.model.unique()))
    check("roster: every registered RQ1 model was rendered",
          set(RQ1_MODELS) <= set(ran),
          f"expected {RQ1_MODELS}\n         rendered {ran}"
          + (f"\n         MISSING {sorted(set(RQ1_MODELS) - set(ran))}"
             if set(RQ1_MODELS) - set(ran) else ""))
    vendors = sorted({MODEL_SPECS[m].vendor for m in ran if m in MODEL_SPECS})
    kinds = sorted({MODEL_SPECS[m].kind for m in ran if m in MODEL_SPECS})
    check("roster: spans more than one vendor and both architectures",
          len(vendors) >= 2 and set(kinds) >= {"dense", "moe"},
          f"vendors={vendors} architectures={kinds}")

    # Every model must carry the SAME role classes, or `R_role` is not the same
    # variable across models and the cross-model comparison is not like-for-like.
    role_sets = {m: tuple(sorted(g.role.unique())) for m, g in fit.groupby("model")}
    distinct = sorted(set(role_sets.values()))
    check("roster: R_role is the same class set on every model",
          len(distinct) == 1,
          f"role sets: {role_sets}" if len(distinct) != 1 else
          f"all {len(role_sets)} models carry {distinct[0]}")
    if len(distinct) == 1 and "tool" not in distinct[0]:
        check("roster: D1 scope limit (no `tool` class anywhere)", True,
              "R_role is an authority/speaker-identity variable only; it does NOT "
              "cover the untrusted-external-content class. State in the write-up.")

    # ------------------------------------------------------------- template bleed
    # BPE merges one template character into the boundary tokens of the
    # instruction on the `tool` role. GATED on structure — tool role only, at
    # most one character each end — and the by-label RATE DIFFERENCE is
    # reported, not gated.
    #
    # An earlier version of this check gated the rate difference at 0.10. That
    # was wrong and it failed here: the difference is ~0.5, driven by final
    # punctuation (AdvBench imperatives vs XSTest questions), which no rendering
    # choice can change. Gating it would fail every run forever on a property of
    # the public datasets. It is controlled by the surface/length-only baseline
    # that R_harm must beat, and reported in the write-up. `experiments/
    # e1_0_corpus.py` makes the same distinction; the two must not disagree.
    if "bleed_tail" in idx.columns:
        b = fit.copy()
        b["bleed_head"] = b.bleed_head.fillna("").astype(str)
        b["bleed_tail"] = b.bleed_tail.fillna("").astype(str)
        b["has_bleed"] = b.bleed_head.ne("") | b.bleed_tail.ne("")
        viol, rep = [], []
        for (mm, role, design), g in b.groupby(["model", "role", "design"]):
            if not g.has_bleed.any():
                continue
            chars = sorted({c for c in g.bleed_tail.unique() if c}
                           | {c for c in g.bleed_head.unique() if c})
            rh = float(g[g.harmful].has_bleed.mean()) if g.harmful.any() else 0.0
            rl = float(g[~g.harmful].has_bleed.mean()) if (~g.harmful).any() else 0.0
            rep.append(f"{mm}/{role}/{design} {chars} harmful={rh:.3f} "
                       f"harmless={rl:.3f} |d|={abs(rh - rl):.3f}")
            if role != "tool":
                viol.append(f"{mm}/{role}/{design}: bleed outside the tool role")
            if any(len(c) > 1 for c in chars):
                viol.append(f"{mm}/{role}/{design}: bleed longer than one char {chars}")
        check("corpus: template bleed is structurally bounded (tool role, <=1 char)",
              not viol, "; ".join(viol[:4]) if viol else
              ("\n         ".join(rep) if rep else "no bleed on any model"))
        if rep:
            print("[info] bleed by-label rate difference is REPORTED, not gated — it "
                  "tracks source punctuation and is covered by the surface baseline")
    else:
        check("corpus: template bleed recorded", False,
              "rendered_index.csv has no bleed_head/bleed_tail columns — rebuild the corpus")

    # ------------------------------------------------- E1.7 Level 2 style corpus
    # The whole Level 2 design rests on the crossing being balanced: the register
    # contrast is only a register contrast if every base appears in every
    # register. Checked here, on the corpus, rather than inferred from the
    # downstream numbers.
    sdir = ROOT / "e1_7_style"
    if (sdir / "style_corpus.jsonl").exists():
        import json as _json
        from collections import Counter
        sm = json.loads((sdir / "style_corpus_meta.json").read_text())
        sitems = [json.loads(l) for l in open(sdir / "style_corpus.jsonl") if l.strip()]
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

        # Layer selection: REPORTED, not checked, and the distinction is the point.
        #
        # This was previously written as `check(..., True, ...)` — a call that
        # passes unconditionally whatever the data says. It could not fail, so it
        # was decoration. The mutation suite (tests/test_verifier_teeth.py) found
        # it by corrupting the artifact and observing that nothing complained.
        #
        # It cannot be made into a real check HERE, because
        # `direction_validation.csv` records no "which layer did we report"
        # field: both quantities below are derived from the same table by
        # applying the selection rule at read time, so the artifact cannot
        # disagree with itself. Selection-on-train is a property of the SOURCE,
        # and it is tested as one in tests/test_invariants.py.
        agree = []
        for c in val.concept.unique():
            sv = val[val.concept == c]
            sel = int(sv.loc[sv.train_auc.idxmax(), "layer"])
            best_test = int(sv.loc[sv.auc.idxmax(), "layer"])
            agree.append((c, sel, best_test))
        n_same = sum(1 for _, a, b in agree if a == b)
        print(f"[info] {m}: train-selected vs test-argmax layer per concept — "
              f"{n_same}/{len(agree)} coincide: "
              + ", ".join(f"{c}: L{a}/L{b}" for c, a, b in agree[:5]))

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

        # ---- the soft-refusal probe's held-out claim ----------------------
        # E (tools/soft_refusal_probe.py) asks where the `undetermined` items
        # project relative to the two poles R_control was fitted on. That is only
        # a held-out test if those items were genuinely excluded from the fit —
        # if they had leaked in, the projection would partly restate the fit and
        # the conclusion would be circular.
        #
        # Checked against the labels rather than assumed from the code: the fit's
        # own recorded class counts must equal the refused/complied counts on the
        # harmful side, with `undetermined` accounted for by being in NEITHER.
        und = int((lab[lab.harmful] .label == "undetermined").sum()) if "harmful" in lab.columns else 0
        rc = val[val.concept == "R_control"]
        if len(rc) and und:
            r0 = rc.iloc[0]
            h = lab[lab.harmful]
            want = {
                "n_positive": int((h[h.split == "train"].label == "refused").sum()),
                "n_negative": int((h[h.split == "train"].label == "complied").sum()),
                "n_test_pos": int((h[h.split == "test"].label == "refused").sum()),
                "n_test_neg": int((h[h.split == "test"].label == "complied").sum()),
            }
            got = {k: int(r0[k]) for k in want if k in rc.columns}
            check(f"{m}: R_control fit excludes the undetermined pool (E is held out)",
                  got == {k: v for k, v in want.items() if k in got},
                  f"fit counts {got} vs labels {want}; {und} undetermined items are in "
                  f"neither class"
                  + ("" if got == {k: v for k, v in want.items() if k in got}
                     else "  MISMATCH — the held-out projection test would be circular"))

        # ---- minimum n behind every reported interval --------------------
        # A bootstrap CI is computed from whatever is there, so a cell with a
        # handful of test instructions yields a confident-looking interval that
        # means nothing. The cluster bootstrap resamples BY INSTRUCTION, so the
        # instruction count — not the item count, which is 8x larger because of
        # the role x design crossing — is the real sample size.
        #
        # KEYED ON THE MINORITY CLASS, not the union. An earlier version of this
        # check read `n_test_instructions`, which counts instructions
        # contributing to EITHER class. For `R_control` that is 45-50 and passes
        # comfortably while the minority class has as few as 10 test ITEMS over
        # ~4 instructions — the contrast is imbalanced by construction (refusal
        # is rare among harmless prompts and compliance is rare among harmful
        # ones), so the union is always healthy and always the wrong number.
        # Power is set by the smaller side.
        MIN_TEST_INSTRUCTIONS = 20
        MIN_TEST_MINORITY_ITEMS = 20
        if {"n_test_pos", "n_test_neg"} <= set(val.columns):
            v = val.copy()
            v["minority_items"] = v[["n_test_pos", "n_test_neg"]].min(axis=1)
            # one row per concept: the layer that is actually reported
            rep = v.loc[v.groupby("concept").train_auc.idxmax()]
            thin = rep[rep.minority_items < MIN_TEST_MINORITY_ITEMS]
            check(f"{m}: every reported CI has >= {MIN_TEST_MINORITY_ITEMS} "
                  f"MINORITY-class test items",
                  thin.empty,
                  (", ".join(f"{r.concept}@L{int(r.layer)} minority={int(r.minority_items)} "
                             f"(pos={int(r.n_test_pos)}, neg={int(r.n_test_neg)})"
                             for r in thin.itertuples()))
                  if not thin.empty else
                  ", ".join(f"{r.concept} minority={int(r.minority_items)}"
                            for r in rep.itertuples()))
        else:
            check(f"{m}: per-class test sizes recorded", False,
                  "direction_validation.csv lacks n_test_pos / n_test_neg")

        if "n_test_instructions" in val.columns:
            thin2 = val[val.n_test_instructions < MIN_TEST_INSTRUCTIONS]
            check(f"{m}: every reported CI has >= {MIN_TEST_INSTRUCTIONS} test "
                  f"instructions (union, weaker check)",
                  thin2.empty,
                  f"min n_test_instructions = {int(val.n_test_instructions.min())} "
                  f"across {len(val)} direction-layer fits")

        # Activations: no overflow or NaN from the fp16 storage cast.
        _cj = d / "activations_cache.json"
        _bp = (Path(json.loads(_cj.read_text())["path"]) if _cj.exists()
               else d / "activations.pt")
        check(f"{m}: activation cache is present where the run recorded it",
              _bp.exists(), f"{_bp}" + ("" if _bp.exists() else "  MISSING"))
        # `continue` here would skip THE ENTIRE REST OF THIS MODEL'S CHECKS —
        # E1.1d, the gate, liveness, reproduction — on nothing more than a
        # missing cache file. That is the "check that cannot fail" failure mode
        # this file's mutation suite exists to catch, and it caught exactly this.
        # Only the activation-specific checks are skipped.
        blob = (torch.load(_bp, map_location="cpu", weights_only=False, mmap=True)
                if _bp.exists() else None)
        if blob is not None:
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

        # The pinned CONTROL_VARIANT must agree with THIS run's own labels.
        #
        # The variant is a data-dependent choice — `under` needs a populated
        # harmful-and-complied cell, `over` needs over-refusal on harmless — but
        # it is pinned per job in `blank_slate.sh` from counts measured in EARLIER
        # runs. In a clean start those counts no longer exist, so the pin is an
        # assumption until it is checked against the labels this run produced.
        #
        # One direction already fails loudly: pinning a variant whose direction
        # was not extracted raises in `stage_causal`. The other is silent — if a
        # richer contrast became available and we ran the weaker one anyway,
        # nothing would say so. That is what this checks.
        lc = d / "labels_checks.json"
        if lc.exists():
            rv = json.loads(lc.read_text()).get("checks", {}).get("refusal_variance", {})
            supported = {v for v, key in (("under", "within_harmful"),
                                          ("over", "within_harmless"))
                         if rv.get(key, {}).get("usable")}
            ran = {p.stem.replace("causal_gate__", "")
                   for p in d.glob("causal_gate__*.json")}
            # An arm that ran with CONTROL_VARIANT_OPTIONAL and found its
            # direction unavailable records a skip. That is a legitimate outcome
            # — but ONLY if this run's labels agree the contrast is not usable.
            # A skip that contradicts the labels means the direction went
            # missing somewhere between labelling and extraction, which would
            # otherwise pass unnoticed as "we just didn't run that arm".
            skipped = {p.stem.replace("causal_skipped__", "")
                       for p in d.glob("causal_skipped__*.json")}
            check(f"{m}: every variant run is supported by this run's labels",
                  ran <= supported or not ran,
                  f"ran {sorted(ran) or '-'}; labels support {sorted(supported) or '-'}"
                  + (f"  UNSUPPORTED: {sorted(ran - supported)}" if ran - supported else ""))
            check(f"{m}: skipped variants are ones the labels call unusable",
                  not (skipped & supported),
                  f"skipped {sorted(skipped) or '-'}; labels support "
                  f"{sorted(supported) or '-'}"
                  + (f"  CONTRADICTION: {sorted(skipped & supported)} was skipped but "
                     f"the labels say it is estimable" if skipped & supported else ""))
            missed = supported - ran - skipped
            check(f"{m}: no usable control contrast was left unrun",
                  not missed,
                  f"labels support {sorted(supported)} but only {sorted(ran)} was run"
                  f" — {sorted(missed)} is estimable on this run's data and would be a "
                  f"stronger contrast" if missed else
                  f"ran every supported variant: {sorted(ran)}"
                  + (f" (skipped, unusable: {sorted(skipped)})" if skipped else ""))

        # The matched cross-model arm must exist on EVERY model, or the
        # cross-model gate comparison is not like-for-like — it would set
        # under-refusal on one model against over-refusal on another and read
        # the difference as a failure to replicate.
        check(f"{m}: matched cross-model arm '{MATCHED_CONTROL_VARIANT}' was run",
              (d / f"causal_gate__{MATCHED_CONTROL_VARIANT}.json").exists(),
              f"causal_gate__{MATCHED_CONTROL_VARIANT}.json "
              + ("present" if (d / f"causal_gate__{MATCHED_CONTROL_VARIANT}.json").exists()
                 else "MISSING — this model cannot enter the cross-model comparison"))

        # Gate artifacts carry a control-variant suffix (`__under` / `__over`) so
        # that runs which differ in the control variable cannot be conflated.
        # Globbing rather than naming one file matters: a check that silently
        # finds nothing is worse than one that fails, because it still reports a
        # green run.
        # `.superseded.json` are pre-fix BACKUPS kept for audit by
        # tools/readjudicate_gates.py — archives, not live artifacts.
        gates = sorted(g for g in d.glob("causal_gate*.json")
                       if ".superseded" not in g.name)
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

            # ---- LIVENESS: every criterion must be ABLE to fire -------------
            # Generalises the assertion added after the G3 bug, which is the
            # paradigm failure this project has to defend against: a null band
            # was looked up with a scalar key while the dict was keyed by a
            # 1-tuple, every lookup missed, the default was `inf`, and
            # `shift > inf` was therefore ALWAYS False. G3 could not fire, so
            # every GATE 1 verdict was FAIL by construction — with no error, no
            # NaN in any CSV, and a complete, plausible artifact set.
            #
            # A band that is non-finite, non-positive, or MISSING FOR AN ALPHA
            # THAT THE MATRIX ACTUALLY CONTAINS is that bug. It is checked here
            # against the matrix's own alphas rather than against the band's
            # self-report, because the failure was precisely that the band
            # looked complete while being unreachable.
            mat_for_tag = d / f"causal_matrix{gp.stem.replace('causal_gate', '')}.csv"
            alphas_in_matrix = set()
            if mat_for_tag.exists():
                _mm = pd.read_csv(mat_for_tag)
                _acol = "abs_alpha" if "abs_alpha" in _mm.columns else "alpha"
                alphas_in_matrix = {round(abs(float(a)), 6)
                                    for a in _mm[_acol].dropna().unique()}

            dead, unreachable = [], []
            for b, v in g["per_bound"].items():
                for band_name in ("null_band_delta", "null_band_behaviour"):
                    band = v.get(band_name)
                    if not isinstance(band, dict):
                        dead.append(f"bound {b}: {band_name} absent")
                        continue
                    # A zero band the adjudication EXPLICITLY EXCLUDED is
                    # handled, not a fault. `_adjudicate_at` records those alphas
                    # in `degenerate_alphas_excluded` and drops their cells from
                    # G3, so the criterion is not satisfiable by a shift that
                    # merely exceeds zero. What must still fail is a zero or
                    # non-finite band that nothing accounted for — the original
                    # bug, where the band was `inf` and G3 could never fire.
                    g3blk = next((v[k2] for k2 in v if k2.startswith("G3")), {})
                    excluded = {str(float(a)) for a in
                                g3blk.get("degenerate_alphas_excluded", [])}
                    for k, val in band.items():
                        if np.isfinite(float(val)) and float(val) > 0:
                            continue
                        try:
                            handled = str(float(str(k).strip("()',"))) in excluded
                        except ValueError:
                            handled = False
                        if not handled:
                            dead.append(f"bound {b}: {band_name}[{k}]={val} "
                                        f"(not in degenerate_alphas_excluded)")
                # Every alpha the matrix carries must have a behavioural band,
                # or G3 is silently unfireable at that alpha.
                beh = v.get("null_band_behaviour") or {}
                have = set()
                for k in beh:
                    try:
                        have.add(round(abs(float(str(k).strip("()',"))), 6))
                    except ValueError:
                        pass
                missing = sorted(alphas_in_matrix - have)
                if alphas_in_matrix and missing:
                    unreachable.append(f"bound {b}: no behavioural band for alpha {missing}")

            check(f"{m}{tag}: every null band is finite and positive (G3-bug guard)",
                  not dead, "; ".join(dead[:6]) if dead else
                  f"all bands finite across {len(g['per_bound'])} capability bounds")
            check(f"{m}{tag}: every alpha in the matrix has a behavioural null band",
                  not unreachable, "; ".join(unreachable[:4]) if unreachable else
                  f"alphas {sorted(alphas_in_matrix)} all covered")

            # Report, do not gate, the per-criterion counts: a zero here can be
            # a real null result OR an unfireable criterion, and the bands above
            # are what distinguish them. Printing makes a zero impossible to miss.
            for b, v in sorted(g["per_bound"].items()):
                g2 = v.get("G2_asymmetry", {})
                g3 = v.get("G3_behavioural", v.get("G3", {}))
                print(f"         liveness {m}{tag} bound={b}: G2 tested={g2.get('n_pairs_tested')} "
                      f"ci_disjoint={g2.get('n_cis_disjoint')} beyond_null={g2.get('n_beyond_null')} "
                      f"fdr={g2.get('n_fdr_reject')} -> holds={g2.get('holds')} | "
                      f"G3 holds={g3.get('holds')} n={g3.get('n_moved', g3.get('n'))}")

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
