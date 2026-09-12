#!/usr/bin/env python
"""Full-scope RQ1 readout — every pillar, every model, every arm, as results land.

WHY THIS EXISTS
`rq1_walkthrough.py` answers "did the artifact appear". It does not answer "what
does it say". Watching a run through the control arm alone is a narrow lens:
RQ1 asks whether role perception, harmfulness recognition and refusal/compliance
control are causally distinguishable, and investigates their GEOMETRY,
DIMENSIONALITY, EMERGENCE and CAUSAL INTERACTIONS. Three of those four do not
involve `R_control`'s cell sizes at all, and are fully powered on every model
regardless of how the control arm lands.

So this prints the headline number for each criterion C1-C10, per model, from
whatever has been written so far. It reads CSV/JSON only — never the activation
blob — so it is cheap and safe to run against a live run.

    python tools/rq1_status.py
    RESULTS_ROOT=./results_verify python tools/rq1_status.py
    python tools/rq1_status.py --csv   # one row per (model, criterion, quantity)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from core.config import MATCHED_CONTROL_VARIANT, MODELS as SPEC, RQ1_MODELS  # noqa: E402

ROOT = Path(os.environ.get("RESULTS_ROOT", "./results"))
ROWS: list[dict] = []


def rec(model: str, crit: str, quantity: str, value, note: str = "") -> None:
    ROWS.append({"model": model, "criterion": crit, "quantity": quantity,
                 "value": value, "note": note})


def _csv(p: Path):
    try:
        return pd.read_csv(p) if p.exists() else None
    except Exception:
        return None


def _json(p: Path):
    try:
        return json.loads(p.read_text()) if p.exists() else None
    except Exception:
        return None


def model_report(m: str) -> None:
    d = ROOT / "rq1" / m
    spec = SPEC.get(m)
    head = f"{m}  [{spec.vendor if spec else '?'} · {spec.kind if spec else '?'}]"
    print(f"\n{'=' * 78}\n{head}\n{'=' * 78}")
    if not d.exists():
        print("  no results yet")
        return

    # ---------------------------------------------------------------- labels
    lab = _csv(d / "refusal_labels.csv")
    if lab is not None:
        print("\n-- BEHAVIOUR (gates which control arms are estimable) --")
        for side, mask, pos, neg in [("under  (within harmful )", lab.harmful, "refused", "complied"),
                                     ("over   (within harmless)", ~lab.harmful, "refused", "complied")]:
            g = lab[mask]
            n_p = int((g.label == pos).sum())
            n_n = int((g.label == neg).sum())
            n_u = int((g.label == "undetermined").sum())
            mino, which = (n_p, pos) if n_p < n_n else (n_n, neg)
            instr = int(g[g.label == which].uid.nunique())
            ok = "OK " if instr >= 20 else "THIN"
            print(f"   {side}: refused={n_p:5d} complied={n_n:5d} undetermined={n_u:5d}"
                  f"  -> minority {mino:4d} items / {instr:3d} instr  [{ok}]")
            rec(m, "arms", side.split()[0], instr, f"minority={which}, {ok}")
        # `undetermined` is not degeneracy — it is Guard-verified harmless
        # non-refusal (soft refusal). Reported because it is a finding, not noise.
        if "label_reason" in lab.columns:
            u = lab[lab.label == "undetermined"].label_reason.value_counts().to_dict()
            if u:
                print(f"   undetermined reasons: {u}")

    # ---------------------------------------------------------- C1 recovery
    val = _csv(d / "direction_validation.csv")
    if val is not None and len(val):
        print("\n-- C1  variables recoverable and validated (best TRAIN-selected layer) --")
        for c in sorted(val.concept.unique()):
            s = val[val.concept == c]
            b = s.loc[s.train_auc.idxmax()]
            beats = "yes" if b.auc > b.length_only_auc else "NO"
            mino = int(min(b.get("n_test_pos", 0), b.get("n_test_neg", 0)))
            print(f"   {c:22s} L{int(b.layer):3d}  AUC {b.auc:.3f} "
                  f"[{b.get('auc_ci_low', float('nan')):.3f},{b.get('auc_ci_high', float('nan')):.3f}]"
                  f"  length-only {b.length_only_auc:.3f} (beats: {beats})"
                  f"  split-half {b.get('split_half_cos', float('nan')):.3f}"
                  f"  minority n={mino}")
            rec(m, "C1", f"{c}.auc", round(float(b.auc), 4),
                f"L{int(b.layer)}, beats_length={beats}, minority_n={mino}")
    probe = _csv(d / "role_probe.csv")
    if probe is not None and len(probe):
        b = probe.loc[probe.train_accuracy.idxmax()]
        ch = float(b.chance) if "chance" in probe.columns else 0.25
        print(f"   R_role_probe (multiclass)  L{int(b.layer):3d}  acc {b.accuracy:.3f} "
              f"[{b.acc_ci_low:.3f},{b.acc_ci_high:.3f}]  chance {ch:.3f}")
        rec(m, "C1", "R_role_probe.acc", round(float(b.accuracy), 4), f"chance={ch}")

    # ------------------------------------------------- C2 harm vs behaviour
    hc = _csv(d / "harm_controls.csv")
    if hc is not None and len(hc):
        print("\n-- C2  R_harm separable from refusal behaviour (E1.1d) --")
        bal = bool(hc.role_balanced.all()) if "role_balanced" in hc.columns else None
        for v in sorted(hc.variant.unique()) if "variant" in hc.columns else []:
            s = hc[hc.variant == v]
            if not len(s):
                continue
            b = s.loc[s.auc.idxmax()] if "auc" in s.columns else s.iloc[0]
            print(f"   {v:26s} AUC {b.get('auc', float('nan')):.3f}  "
                  f"length-only {b.get('length_only_auc', float('nan')):.3f}")
            rec(m, "C2", v, round(float(b.get("auc", float("nan"))), 4))
        print(f"   role-balanced: {bal}")

    # ------------------------------------------------------------ C3 geometry
    geo = _csv(d / "geometry_cosines.csv")
    if geo is not None and len(geo):
        print("\n-- C3  pairwise geometry (max |cos| over layers) --")
        # The role direction is NOT called `R_role`: `_role_direction` names it
        # `R_role_<a>_vs_<b>@<position>` per contrast and read position. A
        # hardcoded "R_role" matches nothing, which silently reduced this
        # section to the single R_harm/R_control pair — one of the three pairs
        # the geometry pillar is about.
        #
        # Positions are kept MATCHED. `R_harm` lives at t_inst and `R_control` at
        # t_post_inst; projecting one onto the other's basis is the cross-position
        # comparison the analysis protocol forbids, so R_harm is compared against
        # role at t_inst, and R_control against role at t_post_inst, with
        # `R_harm_at_post` carrying the harm/control comparison.
        def _pairs(lhs: str, rx: str):
            out = []
            for r in geo.itertuples():
                aa, bb = r.concept_a, r.concept_b
                if aa == lhs and rx in bb:
                    out.append((bb, r))
                elif bb == lhs and rx in aa:
                    out.append((aa, r))
            return out

        def _show(lhs, rhs_exact=None, rhs_contains=None, label=""):
            if rhs_exact:
                s2 = geo[((geo.concept_a == lhs) & (geo.concept_b == rhs_exact)) |
                         ((geo.concept_a == rhs_exact) & (geo.concept_b == lhs))]
                cands = [(rhs_exact, s2.loc[s2.abs_cosine.idxmax()])] if len(s2) else []
            else:
                grouped = {}
                for nm, r in _pairs(lhs, rhs_contains):
                    if nm not in grouped or r.abs_cosine > grouped[nm].abs_cosine:
                        grouped[nm] = r
                cands = sorted(grouped.items())
            for nm, r in cands:
                nullp = r.null_p97_5 if hasattr(r, "null_p97_5") else float("nan")
                floor = r.split_half_floor if hasattr(r, "split_half_floor") else float("nan")
                # "Above the random band" is true of almost everything and says
                # little on its own: 0.12 and 0.85 are both above 0.032, and
                # calling both "shares structure" flattens the only distinction
                # that matters. The informative quantity is WHERE the cosine sits
                # between the two reference points the design already provides:
                #
                #   null p97.5   = what two RANDOM directions score  -> fully distinct
                #   split-half   = what the SAME direction scores when refit on
                #                  two halves of its own data       -> identical
                #
                # so  pos = (|cos| - null) / (floor - null)  reads as 0 = distinct,
                # 1 = the same direction. Reported alongside the raw cosine, never
                # instead of it.
                span = (floor - nullp)
                pos = (r.abs_cosine - nullp) / span if span and span > 1e-9 else float("nan")
                if pos != pos:
                    verdict = "no floor available"
                elif pos <= 0.10:
                    verdict = f"pos {pos:+.2f} — DISTINCT (at chance)"
                elif pos <= 0.40:
                    verdict = f"pos {pos:+.2f} — mostly distinct"
                elif pos <= 0.70:
                    verdict = f"pos {pos:+.2f} — PARTIALLY OVERLAPPING"
                else:
                    verdict = f"pos {pos:+.2f} — strongly overlapping"
                print(f"   {lhs:16s} vs {nm:34s} max|cos| {r.abs_cosine:.3f} @L{int(r.layer)}"
                      f"  null {nullp:.3f}  floor {floor:.3f}  -> {verdict}")
                rec(m, "C3", f"cos({lhs},{nm})", round(float(r.abs_cosine), 4), verdict)

        print("   [harm vs control, both at t_post_inst]")
        _show("R_harm_at_post", rhs_exact="R_control")
        _show("R_harm_at_post", rhs_exact="R_control_harmless")
        print("   [harm vs role, both at t_inst]")
        _show("R_harm", rhs_contains="R_role_") if True else None
        print("   [control vs role, both at t_post_inst]")
        _show("R_control", rhs_contains="R_role_")
    sub = _json(d / "geometry_subspace_summary.json")
    if sub:
        ks = sub.get("k_star", {})
        print(f"   subspace k*: {ks}")

    # ------------------------------------------------------ C4 dimensionality
    dim = _csv(d / "dimensionality.csv")
    if dim is not None and len(dim):
        print("\n-- C4  dimensionality --")
        # `r_eff_stratified`, not `r_eff`. The old fallback took the LAST column,
        # which is `mean_pairwise_cos` — so this printed cosines (0.92-0.99)
        # labelled as effective rank, hiding the real values (2.7-13.3).
        # r_eff >= 1 by construction, so anything below 1 was proof of the bug.
        col = next((c for c in ("r_eff_stratified", "r_eff") if c in dim.columns), None)
        if col is None:
            print("   no r_eff column found; columns are " + str(list(dim.columns)))
            return
        for c in sorted(dim.concept.unique()) if "concept" in dim.columns else []:
            s = dim[dim.concept == c]
            extra = ""
            if "auc_after_projecting_out_top1" in dim.columns:
                extra = (f"  AUC after removing top-1: "
                         f"{s.auc_after_projecting_out_top1.median():.3f}")
            print(f"   {c:22s} r_eff {s[col].median():.2f} "
                  f"(range {s[col].min():.2f}-{s[col].max():.2f}){extra}")
            rec(m, "C4", f"{c}.r_eff_median", round(float(s[col].median()), 3))
    bk = _json(d / "behavioural_k.json")
    if bk:
        print(f"   behavioural k*: {json.dumps(bk)[:220]}")

    # ---------------------------------------------------------- C5 emergence
    em = _csv(d / "emergence_summary.csv")
    if em is not None and len(em):
        print("\n-- C5  emergence across layers (onset depth, bootstrap CI) --")
        for r in em.itertuples():
            o = getattr(r, "onset_90", getattr(r, "onset_80", float("nan")))
            print(f"   {r.concept:22s} peak {getattr(r, 'peak_auc', float('nan')):.3f} "
                  f"@depth {getattr(r, 'peak_depth', float('nan')):.2f}   onset90 depth {o}")
            rec(m, "C5", f"{r.concept}.onset90", o)

    # ------------------------------------------------------------- C6 causal
    print("\n-- C6  causal distinguishability -> GATE 1 --")
    gates = sorted(d.glob("causal_gate*.json"))
    skipped = sorted(d.glob("causal_skipped__*.json"))
    if not gates and not skipped:
        print("   (causal stage not reached yet)")
    for gp in gates:
        g = _json(gp) or {}
        tag = gp.stem.replace("causal_gate", "") or "(default)"
        print(f"   {gp.name}: verdict {g.get('verdict')}  ({g.get('n_cells')} cells)")
        for bnd, v in sorted((g.get("per_bound") or {}).items()):
            g2 = v.get("G2_asymmetry", {})
            g3 = v.get("G3_behavioural_dissociation", v.get("G3_behavioural", v.get("G3", {})))
            print(f"      bound {bnd}: G2 holds={g2.get('holds')} "
                  f"({g2.get('n_asymmetric')}/{g2.get('n_pairs_tested')} pairs) | "
                  f"G3 holds={g3.get('holds')}  -> {v.get('verdict')}")
        rec(m, "C6", f"gate{tag}", g.get("verdict"))
    for sp in skipped:
        s = _json(sp) or {}
        print(f"   {sp.name}: SKIPPED — {s.get('reason')}")
        rec(m, "C6", sp.stem, "skipped", s.get("reason", ""))

    # ------------------------------------------------------ C7 role vs style
    st = _csv(d / "style_vs_metadata.csv")
    if st is not None and len(st):
        print("\n-- C7  role is metadata, not linguistic style --")
        print(f"   Level 1: {len(st)} rows")
    l2 = _csv(d / "style_level2.csv")
    if l2 is not None and len(l2):
        s2 = _json(d / "style_level2_summary.json") or {}
        print(f"   Level 2 (controlled register): {len(l2)} rows  {json.dumps(s2)[:200]}")

    # --------------------------------------------------------- C9 literature
    zh = _csv(d / "zhao_replication.csv")
    if zh is not None and len(zh):
        # No `train_auc` column here — the old `.iloc[0]` silently reported layer 0.
        b = zh.loc[zh.auc.idxmax()] if "auc" in zh.columns else zh.iloc[0]
        print(f"\n-- C9  Zhao replication (advbench|alpaca): AUC "
              f"{b.get('auc', float('nan')):.3f} @L{int(b.get('layer', -1))}")
        rec(m, "C9", "zhao_auc", round(float(b.get("auc", float("nan"))), 4))

    fid = _json(d / "fidelity.json")
    if fid:
        print(f"\n-- fidelity (post_attention_layernorm reproduction): "
              f"{json.dumps(fid)[:200]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", action="store_true", help="also write rq1_status.csv")
    args = ap.parse_args()

    print(f"RQ1 FULL-SCOPE STATUS — {ROOT}")
    print("Pillars: geometry (C3) · dimensionality (C4) · emergence (C5) · "
          "causal (C6); recovery C1/C2, style C7, cross-model C8, literature C9.")
    print("NOTE: C1/C3/C4/C5 do not depend on R_control's cell sizes and are fully "
          "powered on every model regardless of how the control arm lands.")

    for m in RQ1_MODELS:
        model_report(m)

    # ------------------------------------------------------------- C8 across
    print(f"\n{'=' * 78}\nC8  CROSS-MODEL / CROSS-ARCHITECTURE\n{'=' * 78}")
    df = pd.DataFrame(ROWS)
    if len(df):
        for crit in ["arms", "C1", "C3", "C6"]:
            s = df[df.criterion == crit]
            if not len(s):
                continue
            piv = s.pivot_table(index="quantity", columns="model", values="value",
                                aggfunc="first")
            print(f"\n-- {crit} --")
            print(piv.to_string())
        if args.csv:
            out = ROOT / "rq1_status.csv"
            df.to_csv(out, index=False)
            print(f"\nwrote {out}")
    else:
        print("  nothing to compare yet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
