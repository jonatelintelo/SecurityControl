#!/usr/bin/env python
"""Independent recomputation of RQ1's headline numbers from raw artifacts.

WHY THIS IS NOT ANOTHER VERIFIER
`tests/verify_rq1_run.py` checks PROPERTIES (no leakage, layer selected on train,
CI has enough data). `tests/test_estimators.py` checks the estimator functions
against known answers. Neither checks that the NUMBERS IN THE ARTIFACTS were
produced correctly from the data — and a readout that reads the wrong column, or
an aggregation that drops half its rows, passes both.

That is not hypothetical. Auditing the status tool by recomputation found five
such bugs in one pass: an r_eff below 1 (impossible for a participation ratio,
caused by a positional column fallback), `G3 holds=None` under a `PASS` verdict
(key-name miss), half the geometry pairs silently absent (literal-prefix match),
Zhao reported at layer 0 (missing `train_auc` falling through to `iloc[0]`), and
a verdict string that said "overlaps null" for a value 7x ABOVE the null.

So this recomputes each quantity FROM THE RAW DATA with a deliberately separate
implementation — plain numpy/pandas, no `core.extract` — and compares. Where the
two disagree, one of them is wrong and it says so.

    python tools/audit_independent.py                 # all models with artifacts
    python tools/audit_independent.py --model yi-6b-chat
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

from core.config import RQ1_MODELS  # noqa: E402

ROOT = Path(os.environ.get("RESULTS_ROOT", "./results"))
RESULTS: list[tuple[str, str, bool, str]] = []


def rec(model: str, name: str, ok: bool, detail: str) -> None:
    RESULTS.append((model, name, ok, detail))
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}\n          {detail}")


# ---------------------------------------------------------------- primitives
def auc_bruteforce(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney U by explicit pairwise comparison. O(n*m) and obviously
    correct, which is the point — it shares no code with the implementation
    under test."""
    p = np.asarray(pos, dtype=np.float64).ravel()
    n = np.asarray(neg, dtype=np.float64).ravel()
    if p.size == 0 or n.size == 0:
        return float("nan")
    gt = (p[:, None] > n[None, :]).sum()
    eq = (p[:, None] == n[None, :]).sum()
    return float((gt + 0.5 * eq) / (p.size * n.size))


def r_eff_numpy(x: np.ndarray) -> float:
    xc = x - x.mean(0, keepdims=True)
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    ev = np.linalg.eigvalsh(cov).clip(min=0)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 1e-20 else float("nan")


def bh_reject(p: np.ndarray, q: float = 0.05) -> int:
    """Benjamini-Hochberg, written out rather than imported."""
    p = np.sort(np.asarray(p, dtype=float))
    n = p.size
    if n == 0:
        return 0
    thresh = q * np.arange(1, n + 1) / n
    below = np.nonzero(p <= thresh)[0]
    return int(below[-1] + 1) if below.size else 0


# ---------------------------------------------------------------- the audit
def audit(m: str) -> None:
    d = ROOT / "rq1" / m
    print(f"\n{'=' * 78}\n{m}\n{'=' * 78}")
    if not (d / "direction_validation.csv").exists():
        print("  (no artifacts yet)")
        return

    val = pd.read_csv(d / "direction_validation.csv")
    lab = pd.read_csv(d / "refusal_labels.csv")

    # ---- 1. the 2x2 the labels stage recorded vs the labels themselves ----
    lc = d / "labels_checks.json"
    if lc.exists():
        j = json.loads(lc.read_text())
        rv = (j.get("checks") or {}).get("refusal_variance", {})
        got = {}
        for side, mask in (("within_harmful", lab.harmful), ("within_harmless", ~lab.harmful)):
            g = lab[mask]
            got[side] = (int((g.label == "refused").sum()), int((g.label == "complied").sum()))
        det = []
        agree = True
        for side, (r_, c_) in got.items():
            blk = rv.get(side, {})
            er, ec = blk.get("refused"), blk.get("complied")
            same = (er is None and ec is None) or (er == r_ and ec == c_)
            agree &= same
            det.append(f"{side}: labels say {r_}/{c_}, checks say {er}/{ec}")
        rec(m, "labels 2x2 matches the recorded check", agree, "; ".join(det))

    # ---- 2. AUC recomputed from activations + the saved direction ----------
    cj = d / "activations_cache.json"
    blob_p = Path(json.loads(cj.read_text())["path"]) if cj.exists() else d / "activations.pt"
    if blob_p.exists() and (d / "directions.pt").exists():
        import torch
        blob = torch.load(blob_p, map_location="cpu", weights_only=False, mmap=True)
        dirs = torch.load(d / "directions.pt", map_location="cpu", weights_only=False)
        idx = pd.DataFrame(blob["index"])
        L = lab[["uid", "role", "design", "label"]]
        idx = idx.merge(L, on=["uid", "role", "design"], how="left", suffixes=("", "_l"))
        lcol = "label_l" if "label_l" in idx.columns else "label"
        test = (idx.split == "test").to_numpy()

        # R_harm at t_inst: harmful vs harmless on the held-out split
        s = val[val.concept == "R_harm"]
        if len(s) and "R_harm" in dirs:
            b = s.loc[s.train_auc.idxmax()]
            layer = int(b.layer)
            v = dirs["R_harm"][layer]
            vec = (v.vector if hasattr(v, "vector") else v).float().numpy()
            acts = blob["t_inst"][layer].float().numpy()
            proj = acts @ vec
            pos = proj[test & idx.harmful.to_numpy().astype(bool)]
            neg = proj[test & ~idx.harmful.to_numpy().astype(bool)]
            mine = auc_bruteforce(pos, neg)
            rec(m, f"R_harm AUC recomputed from activations (L{layer})",
                abs(mine - b.auc) < 2e-3,
                f"independent {mine:.4f} vs artifact {b.auc:.4f} "
                f"(n_pos={pos.size}, n_neg={neg.size})")

            # r_eff on the same layer, as an order-of-magnitude sanity check
            re_mine = r_eff_numpy(acts[test])
            rec(m, "r_eff is a valid participation ratio (>=1)",
                np.isfinite(re_mine) and re_mine >= 1.0,
                f"independent r_eff on held-out activations @L{layer} = {re_mine:.2f}")

        # ---- 3. cosines recomputed from the saved direction vectors -------
        geo = d / "geometry_cosines.csv"
        if geo.exists():
            G = pd.read_csv(geo)
            bad, checked = [], 0
            for r in G.sample(min(40, len(G)), random_state=0).itertuples():
                a, b_ = r.concept_a, r.concept_b
                if a not in dirs or b_ not in dirs:
                    continue
                try:
                    va = dirs[a][int(r.layer)]
                    vb = dirs[b_][int(r.layer)]
                except Exception:
                    continue
                va = (va.vector if hasattr(va, "vector") else va).float().numpy()
                vb = (vb.vector if hasattr(vb, "vector") else vb).float().numpy()
                c = float(abs(va @ vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))
                checked += 1
                if abs(c - r.abs_cosine) > 2e-3:
                    bad.append(f"{a}~{b_}@L{int(r.layer)}: mine {c:.4f} vs {r.abs_cosine:.4f}")
            rec(m, "geometry cosines recomputed from the saved vectors",
                not bad, f"{checked} sampled pairs recomputed"
                + (f"; MISMATCHES: {bad[:3]}" if bad else "; all agree to 2e-3"))
        del blob

    # ---- 4. GATE 1 arithmetic recomputed from the matrix -------------------
    for gp in sorted(d.glob("causal_gate__*.json")):
        tag = gp.stem.replace("causal_gate__", "")
        mp = d / f"causal_matrix__{tag}.csv"
        if not mp.exists():
            continue
        gate = json.loads(gp.read_text())
        M = pd.read_csv(mp)
        if "read_position" in M.columns:
            M = M[M.read_position == "t_post_inst"]
        for bnd, v in sorted(gate.get("per_bound", {}).items()):
            g2 = v.get("G2_asymmetry", {})
            live = M[(M.kl_harmless <= float(bnd)) & (M.source != "random")]
            # POSITIVE alpha only, and each unordered pair counted once.
            #
            # G2 asks whether effect(A->B) differs from effect(B->A). Both signs
            # of alpha describe the same pair, so counting them separately
            # double-counts; and {A,B} is the same test as {B,A}. Omitting the
            # sign filter gave 346 against the artifact's 287 — the audit was
            # wrong, not the gate. Verified: with the filter, 287 and 521 at
            # bounds 0.1 and 0.5, matching exactly.
            pairs = 0
            for (_sl, _rl, _aa), g in live.groupby(["steer_layer", "read_layer", "abs_alpha"]):
                gg = g[g.alpha > 0]
                srcs = sorted(gg.source.unique())
                for i, A in enumerate(srcs):
                    for B in srcs[i + 1:]:
                        if len(gg[(gg.source == A) & (gg.target == B)]) and \
                           len(gg[(gg.source == B) & (gg.target == A)]):
                            pairs += 1
            rec(m, f"gate[{tag}] bound {bnd}: pair count recomputed from the matrix",
                pairs == g2.get("n_pairs_tested"),
                f"independent {pairs} off-diagonal pairs vs artifact "
                f"n_pairs_tested={g2.get('n_pairs_tested')}")
            break   # one bound is enough to catch a systematic error

    # ---- 5. emergence onset recomputed from the curves ---------------------
    ec, es = d / "emergence_curves.csv", d / "emergence_summary.csv"
    if ec.exists() and es.exists():
        C, S = pd.read_csv(ec), pd.read_csv(es)
        bad = []
        for r in S.itertuples():
            s = C[C.concept == r.concept].sort_values("layer")
            if not len(s) or not hasattr(r, "peak_auc"):
                continue
            # The TEST auc at the TRAIN-selected layer — NOT max(test auc).
            #
            # The first version of this check took the maximum test AUC over
            # layers and flagged a "mismatch" of 0.9408 vs 0.9377. The summary
            # was right and the check was wrong: taking the test maximum IS
            # selection on the evaluation set, the exact thing the pipeline is
            # built to avoid (see tests/test_invariants.py, which enforces
            # train-selection at the source level). An audit that demands the
            # test-max would push the code toward the bug it guards against.
            mine = float(s.loc[s.train_auc.idxmax()].auc)
            if abs(mine - r.peak_auc) > 2e-3:
                bad.append(f"{r.concept}: train-selected {mine:.4f} vs summary {r.peak_auc:.4f}")
        rec(m, "emergence peak AUC == test AUC at the train-selected layer",
            not bad, f"{len(S)} concepts checked" + (f"; {bad[:3]}" if bad else "; all agree"))

    # ---- 6. every direction must be unit length ---------------------------
    if (d / "directions.pt").exists():
        import torch
        dirs = torch.load(d / "directions.pt", map_location="cpu", weights_only=False)
        bad = []
        for name, byl in dirs.items():
            for layer, v in (byl.items() if isinstance(byl, dict) else enumerate(byl)):
                vec = (v.vector if hasattr(v, "vector") else v).float().numpy()
                n = float(np.linalg.norm(vec))
                if abs(n - 1.0) > 1e-3:
                    bad.append(f"{name}@L{layer}: |v|={n:.5f}")
        rec(m, "every saved direction is unit length", not bad,
            f"{sum(len(b) if isinstance(b, dict) else 0 for b in dirs.values())} vectors"
            + (f"; {bad[:3]}" if bad else "; all |v|=1"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None)
    args = ap.parse_args()
    print(f"INDEPENDENT RECOMPUTATION AUDIT — {ROOT}")
    print("Every number below is recomputed from raw artifacts with a separate")
    print("implementation (numpy/pandas only, no core.extract) and compared.\n")
    for m in (args.model or RQ1_MODELS):
        try:
            audit(m)
        except Exception as e:
            import traceback
            print(f"  AUDIT ERROR for {m}: {type(e).__name__}: {e}")
            traceback.print_exc()
            RESULTS.append((m, "audit ran", False, str(e)[:120]))
    bad = [r for r in RESULTS if not r[2]]
    print(f"\n{'=' * 78}")
    print(f"{len(RESULTS) - len(bad)}/{len(RESULTS)} independent checks agree with the artifacts")
    for m, n, _, det in bad:
        print(f"  DISAGREES: {m} — {n}\n             {det}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
