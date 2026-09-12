"""Unit tests for the estimators, against cases with KNOWN answers.

WHY THIS EXISTS
Every other check in this project verifies *structure* or *consistency*: that a
column is present, that two runs agree, that a layer was selected on train. None
of them verifies that a number is CORRECT. An off-by-one in the whitening of
`data_canonical_correlations`, or a transposed matrix in `principal_angles`,
would leave every artifact well-formed, every schema intact, and both results
roots in perfect agreement — because both roots would be wrong identically.

So each test here constructs data whose answer is known analytically or by an
independent construction, and asserts the implementation reproduces it. Closed
form where one exists; an independent second implementation where it does not.

Run:  python tests/test_estimators.py
Exit 0 = every estimator reproduces its known answer.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from core import extract  # noqa: E402

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILURES.append(name)
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


def close(a: float, b: float, tol: float = 1e-5) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= tol)


# ---------------------------------------------------------------- diff of means
def t_diff_of_means() -> None:
    # Two clouds whose means differ by a known vector; the estimator must return
    # that vector normalised, regardless of within-class scatter.
    g = torch.Generator().manual_seed(0)
    d = 16
    offset = torch.zeros(d); offset[3] = 2.0; offset[7] = -1.0
    neg = torch.randn(500, d, generator=g)
    pos = torch.randn(500, d, generator=g) + offset
    v = extract.diff_of_means(pos, neg, "c", 0, "residual", "t_inst", "p", "n")
    want = offset / offset.norm()
    # `.vector` is the unit direction; `.raw()` is that times `raw_norm`, i.e.
    # the UN-normalised difference. Using raw() here would compute a projection
    # length and call it a cosine.
    cos = float(torch.dot(v.vector.float(), want))
    check("diff_of_means recovers the known mean offset (cos -> 1)",
          cos > 0.97, f"cos={cos:.4f} vs analytic offset, n=500/class")
    check("diff_of_means .vector is unit length",
          close(float(v.vector.norm()), 1.0), f"|vector|={float(v.vector.norm()):.8f}")
    check("diff_of_means .raw() is the un-normalised difference",
          close(float(v.raw().norm()), float(v.raw_norm), 1e-4),
          f"|raw|={float(v.raw().norm()):.6f} raw_norm={v.raw_norm:.6f}")

    # Sign convention: positive minus negative, not the reverse. Getting this
    # backwards flips every AUC to 1-AUC and would read as "the direction points
    # the wrong way", which the codebase treats as information rather than a bug.
    v2 = extract.diff_of_means(neg, pos, "c", 0, "residual", "t_inst", "n", "p")
    flip = float(torch.dot(v.vector.float(), v2.vector.float()))
    check("diff_of_means sign convention is positive - negative",
          flip < -0.97, f"swapping the classes flips the direction: cos={flip:.4f}")


# ---------------------------------------------------------------- AUC
def t_auc() -> None:
    # Perfect separation, perfect inversion, and exact chance.
    pos = torch.tensor([3.0, 4.0, 5.0])
    neg = torch.tensor([0.0, 1.0, 2.0])
    check("auc = 1.0 on perfect separation", close(extract.auc(pos, neg), 1.0),
          f"{extract.auc(pos, neg)}")
    check("auc = 0.0 on perfect inversion", close(extract.auc(neg, pos), 0.0),
          f"{extract.auc(neg, pos)}")
    # Interleaved: pos={1,3}, neg={0,2}. Pairs (p>n): (1,0),(3,0),(3,2) = 3 of 4.
    check("auc matches a hand-counted rank statistic (3/4)",
          close(extract.auc(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 2.0])), 0.75),
          f"{extract.auc(torch.tensor([1.0, 3.0]), torch.tensor([0.0, 2.0]))}")
    # Ties must count as half, which `argsort().argsort()` does NOT do by itself.
    # This is the test that catches a naive rank implementation.
    tied = extract.auc(torch.tensor([1.0, 1.0]), torch.tensor([1.0, 1.0]))
    check("auc = 0.5 when every score is tied", close(tied, 0.5, 1e-6),
          f"auc={tied} (a rank implementation that ignores ties returns 0.0 or 1.0 here)")

    # Against an independent implementation on random data.
    g = torch.Generator().manual_seed(1)
    p, n = torch.randn(200, generator=g) + 0.5, torch.randn(300, generator=g)
    brute = float((p[:, None] > n[None, :]).float().mean()
                  + 0.5 * (p[:, None] == n[None, :]).float().mean())
    check("auc agrees with a brute-force pairwise count",
          close(extract.auc(p, n), brute, 1e-6), f"{extract.auc(p, n):.8f} vs {brute:.8f}")

    # Invariance to a monotone shift — the documented rank-based property.
    check("auc is invariant to a constant offset",
          close(extract.auc(p, n), extract.auc(p + 10.0, n + 10.0), 1e-9))


# ---------------------------------------------------------------- Cohen's d
def t_cohens_d() -> None:
    # Unit-variance clouds separated by exactly 2.0 -> d = 2.0.
    g = torch.Generator().manual_seed(2)
    a = torch.randn(4000, generator=g)
    b = torch.randn(4000, generator=g) + 2.0
    d = extract.cohens_d(b, a)
    check("cohens_d recovers a known standardised gap of 2.0",
          close(d, 2.0, 0.12), f"d={d:.4f} (n=4000/class)")
    check("cohens_d is 0 for identical distributions",
          abs(extract.cohens_d(a, a.clone())) < 1e-6, f"{extract.cohens_d(a, a.clone())}")


# ---------------------------------------------------------------- effective rank
def t_effective_rank() -> None:
    g = torch.Generator().manual_seed(3)
    d = 32
    # Isotropic: r_eff -> d.
    iso = torch.randn(20000, d, generator=g)
    r_iso = extract.effective_rank(iso)
    check("effective_rank ~ d on isotropic data",
          abs(r_iso - d) / d < 0.05, f"r_eff={r_iso:.2f} vs d={d}")
    # Rank 1: r_eff -> 1.
    v = torch.randn(d, generator=g)
    r1 = extract.effective_rank(torch.randn(2000, 1, generator=g) * v)
    check("effective_rank ~ 1 on rank-one data", abs(r1 - 1.0) < 0.05, f"r_eff={r1:.4f}")
    # Exactly k equal directions -> k, the participation-ratio definition.
    k = 5
    basis = torch.linalg.qr(torch.randn(d, k, generator=g)).Q          # [d, k]
    x = torch.randn(20000, k, generator=g) @ basis.T
    rk = extract.effective_rank(x)
    check(f"effective_rank ~ k on k={k} equal-variance directions",
          abs(rk - k) / k < 0.05, f"r_eff={rk:.3f} vs k={k}")


# ---------------------------------------------------------------- subspace geometry
def t_principal_angles() -> None:
    d = 24
    g = torch.Generator().manual_seed(4)
    # Identical subspaces: every cos = 1, every angle 0, projection metric 1.
    A = torch.randn(3, d, generator=g)
    pa = extract.principal_angles(A, A.clone())
    # 0.05 deg, not 0: arccos is ill-conditioned near 1, so float32 cosine
    # precision (~1e-7) becomes ~sqrt(2e-7) rad ~ 0.03 deg of angle. That is the
    # resolution floor of any reported near-zero principal angle, and is worth
    # knowing when a small angle is interpreted as "these subspaces coincide".
    check("principal_angles: identical subspaces -> all angles ~0 (float32 floor)",
          max(pa["principal_angles_deg"]) < 0.05 and close(pa["projection_metric"], 1.0, 1e-5),
          f"max angle {max(pa['principal_angles_deg']):.2e} deg, "
          f"projection_metric={pa['projection_metric']:.6f}")

    # Orthogonal subspaces built from disjoint coordinate blocks: all angles 90.
    E = torch.eye(d)
    pa2 = extract.principal_angles(E[:3], E[3:6])
    check("principal_angles: orthogonal subspaces -> all angles 90 deg",
          close(min(pa2["principal_angles_deg"]), 90.0, 1e-3)
          and close(pa2["projection_metric"], 0.0, 1e-6),
          f"angles {[round(x,3) for x in pa2['principal_angles_deg']]}, "
          f"projection_metric={pa2['projection_metric']:.2e}")

    # A KNOWN angle: two lines at exactly 30 degrees.
    th = math.radians(30.0)
    a = torch.zeros(1, d); a[0, 0] = 1.0
    b = torch.zeros(1, d); b[0, 0] = math.cos(th); b[0, 1] = math.sin(th)
    pa3 = extract.principal_angles(a, b)
    check("principal_angles: two lines at 30 deg -> 30 deg",
          close(pa3["principal_angles_deg"][0], 30.0, 1e-3),
          f"{pa3['principal_angles_deg'][0]:.6f} deg")

    # Pass 2 must strictly generalise pass 1: for k=1 the cosine of the principal
    # angle IS |cos| between the unit vectors. If this fails, the two passes of
    # E1.2 disagree and the subspace pass cannot be compared to the vector pass.
    u = torch.randn(1, d, generator=g)
    w = torch.randn(1, d, generator=g)
    cos_vec = abs(float(torch.dot(u[0] / u[0].norm(), w[0] / w[0].norm())))
    check("principal_angles at k=1 reduces to |cos| between the vectors",
          close(extract.principal_angles(u, w)["cos_principal_angles"][0], cos_vec, 1e-5),
          f"{extract.principal_angles(u, w)['cos_principal_angles'][0]:.8f} vs {cos_vec:.8f}")

    # A partially-overlapping pair: share one axis, differ in the other, so the
    # spectrum must be exactly [1, 0] and the projection metric exactly 0.5.
    pa4 = extract.principal_angles(E[[0, 1]], E[[0, 2]])
    s = sorted(pa4["cos_principal_angles"], reverse=True)
    check("principal_angles: one shared axis of two -> spectrum [1, 0], metric 0.5",
          close(s[0], 1.0, 1e-6) and close(s[1], 0.0, 1e-6)
          and close(pa4["projection_metric"], 0.5, 1e-6),
          f"spectrum {[round(x, 6) for x in s]}, metric {pa4['projection_metric']:.6f}")


def t_canonical_correlations() -> None:
    g = torch.Generator().manual_seed(5)
    n = 4000
    # Xb is an invertible linear map of Xa -> canonical correlations all 1.
    Xa = torch.randn(n, 3, generator=g)
    M = torch.randn(3, 3, generator=g)
    cc = extract.data_canonical_correlations(Xa, Xa @ M)
    check("canonical correlations = 1 under an invertible linear map",
          len(cc) == 3 and min(cc) > 0.999, f"{[round(c, 6) for c in cc]}")

    # Independent data -> correlations near 0 (finite-sample noise ~ sqrt(k/n)).
    cc0 = extract.data_canonical_correlations(Xa, torch.randn(n, 3, generator=g))
    check("canonical correlations ~ 0 on independent data",
          max(cc0) < 0.15, f"max={max(cc0):.4f} (finite-sample floor)")

    # A KNOWN single correlation: second block shares one component at rho=0.6.
    rho = 0.6
    z = torch.randn(n, 1, generator=g)
    Ya = torch.cat([z, torch.randn(n, 1, generator=g)], 1)
    Yb = torch.cat([rho * z + math.sqrt(1 - rho ** 2) * torch.randn(n, 1, generator=g),
                    torch.randn(n, 1, generator=g)], 1)
    cc1 = extract.data_canonical_correlations(Ya, Yb)
    check("canonical correlations recover a planted rho=0.6",
          close(max(cc1), rho, 0.06), f"max={max(cc1):.4f} vs planted {rho}")

    # Guard: too few samples must return [] rather than a garbage spectrum.
    check("canonical correlations refuse an underdetermined fit",
          extract.data_canonical_correlations(torch.randn(3, 3), torch.randn(3, 3)) == [],
          "n < k+2 returns []")


# ---------------------------------------------------------------- stratum balance
def t_stratum_balanced() -> None:
    # THE BUG THIS GUARDS: R_control is refused-vs-complied pooled over role, and
    # refusal rate varies by role, so the plain estimator carries a role
    # component. Construct exactly that confound and require the balanced
    # estimator to remove it.
    g = torch.Generator().manual_seed(6)
    d = 32
    role_axis = torch.zeros(d); role_axis[0] = 1.0
    true_axis = torch.zeros(d); true_axis[1] = 1.0

    def make(n, role_level, signal):
        return (torch.randn(n, d, generator=g) * 0.1
                + role_level * role_axis + signal * true_axis)

    # positive is 90% role "A", negative is 10% role "A" -> heavy imbalance.
    pos = torch.cat([make(90, 1.0, 1.0), make(10, -1.0, 1.0)])
    neg = torch.cat([make(10, 1.0, -1.0), make(90, -1.0, -1.0)])
    ps = ["A"] * 90 + ["B"] * 10
    ns = ["A"] * 10 + ["B"] * 90

    plain = extract.diff_of_means(pos, neg, "c", 0, "residual", "t_post_inst", "p", "n")
    bal, meta = extract.stratum_balanced_diff_of_means(
        pos, neg, ps, ns, "c", 0, "residual", "t_post_inst", "p", "n")
    leak_plain = abs(float(torch.dot(plain.vector.float(), role_axis)))
    leak_bal = abs(float(torch.dot(bal.vector.float(), role_axis)))
    check("stratum balancing removes the role component the plain estimator carries",
          leak_bal < 0.1 and leak_plain > 0.5,
          f"|<v, role_axis>|: plain={leak_plain:.4f} -> balanced={leak_bal:.4f} "
          f"(lower is cleaner); balanced retains "
          f"|<v, true_axis>|={abs(float(torch.dot(bal.vector.float(), true_axis))):.4f}")
    check("stratum balancing reports that it balanced",
          bool(meta.get("role_balanced", meta.get("balanced", True))),
          f"meta keys: {sorted(meta)[:8]}")


# ---------------------------------------------------------------- nulls
def t_nulls() -> None:
    # Two random unit vectors in d dimensions have E[cos^2] = 1/d, so
    # E[|cos|] ~ sqrt(2/(pi*d)). A null band far from that is miscalibrated, and
    # a miscalibrated band is how a null result becomes a positive one.
    # The band is SIGNED and two-sided (mean ~ 0, symmetric quantiles), so the
    # analytic claim to check is its spread: sd(cos) = 1/sqrt(d) for two random
    # unit vectors. The function reports `analytic_sd` itself, so this also
    # checks the empirical draw against the function's own stated expectation.
    for d in (64, 512):
        band = extract.random_cosine_band(d, n_samples=4000, seed=0)
        want_sd = 1.0 / math.sqrt(d)
        check(f"random_cosine_band sd matches 1/sqrt(d) at d={d}",
              close(band["sd"], want_sd, 0.1 * want_sd)
              and close(band["analytic_sd"], want_sd, 1e-6),
              f"sd={band['sd']:.5f} analytic_sd={band['analytic_sd']:.5f} want={want_sd:.5f}")
        check(f"random_cosine_band is centred on 0 at d={d}",
              abs(band["mean"]) < 0.1 * want_sd, f"mean={band['mean']:+.6f}")
        check(f"random_cosine_band quantiles are symmetric at d={d}",
              close(abs(band["p2.5"]), abs(band["p97.5"]), 0.15 * want_sd),
              f"p2.5={band['p2.5']:.4f} p97.5={band['p97.5']:.4f}")

    # The subspace null must EXCEED the rank-1 null: two random k-planes overlap
    # more than two random lines. If it did not, every k>1 overlap would look
    # significant by construction — the exact error the function exists to avoid.
    d = 128
    r1 = extract.random_subspace_null(d, 1, 1, n_draws=400, seed=0)["mean"]
    r4 = extract.random_subspace_null(d, 4, 4, n_draws=400, seed=0)["mean"]
    check("random_subspace_null grows with k (k=4 overlaps more than k=1)",
          r4 > r1, f"mean projection metric: k=1 {r1:.5f} -> k=4 {r4:.5f}")
    # Analytic: E[projection metric] for random k-planes = k/d.
    check("random_subspace_null matches the analytic k/d expectation",
          close(r4, 4.0 / d, 0.3 * (4.0 / d)), f"got {r4:.5f} want ~{4.0/d:.5f}")


# ---------------------------------------------------------------- split-half
def t_split_half() -> None:
    # A direction fitted on two halves of the SAME signal must be stable;
    # fitted on two halves of pure noise it must not be. The floor is what
    # every geometry claim is compared against, so a floor that is always high
    # would make every cosine look meaningful.
    g = torch.Generator().manual_seed(7)
    d = 64
    axis = torch.zeros(d); axis[5] = 1.0
    pos = torch.randn(300, d, generator=g) * 0.5 + axis
    neg = torch.randn(300, d, generator=g) * 0.5 - axis
    try:
        strong = extract.split_half_stability(pos, neg, n_splits=20, seed=0)
        noise = extract.split_half_stability(torch.randn(300, d, generator=g),
                                             torch.randn(300, d, generator=g),
                                             n_splits=20, seed=0)
    except TypeError as e:
        check("split_half_stability signature", False, f"{e}")
        return
    sv = strong if isinstance(strong, float) else strong.get("mean_cosine", strong.get("mean"))
    nv = noise if isinstance(noise, float) else noise.get("mean_cosine", noise.get("mean"))
    check("split_half_stability is high on real signal and low on noise",
          sv > 0.8 and nv < 0.4, f"signal={sv:.4f}  noise={nv:.4f}")


def t_auc_ties_regression() -> None:
    """Regression guard for the tie bug found by this suite.

    `auc` used `argsort().argsort()`, which gives tied scores arbitrary DISTINCT
    ranks — counting a tie as a full win or a full loss instead of a half. It
    matters because `length_only_baseline` runs this AUC on integer token counts,
    where ties are abundant, and that baseline is what every direction must beat.
    """
    same = torch.tensor([5.0, 5.0, 5.0, 5.0])
    check("auc: every score tied -> exactly 0.5",
          close(extract.auc(same, same.clone()), 0.5, 1e-12),
          f"{extract.auc(same, same.clone())}")
    check("auc: identical multisets -> exactly 0.5",
          close(extract.auc(torch.tensor([1.0, 2.0, 3.0]),
                            torch.tensor([1.0, 2.0, 3.0])), 0.5, 1e-12))
    # Integer lengths, the real case: swapping which side is "positive" must give
    # exactly 1 - auc. Arbitrary tie-breaking violates this.
    g = torch.Generator().manual_seed(11)
    a = torch.randint(10, 25, (300,), generator=g).float()
    b = torch.randint(12, 30, (300,), generator=g).float()
    check("auc on integer lengths: auc(a,b) + auc(b,a) == 1 exactly",
          close(extract.auc(a, b) + extract.auc(b, a), 1.0, 1e-12),
          f"{extract.auc(a, b):.10f} + {extract.auc(b, a):.10f} = "
          f"{extract.auc(a, b) + extract.auc(b, a):.10f}")


def t_refusal_typography() -> None:
    """The refusal rule must not be defeated by a curly apostrophe.

    Arditi's prefix list is ASCII ("I'm sorry", "I can't"). NVIDIA-Nemotron-3
    emits U+2019 throughout, so unnormalised the rule called 1543 of its 1600
    harmful responses `complied` — including verbatim "I'm sorry, but I can't
    help with that." The labels gate caught it, but only after a 74-minute job.
    """
    from core import refusal

    curly = "I\u2019m sorry, but I can\u2019t help with that."
    ascii_ = "I'm sorry, but I can't help with that."
    for rule in refusal.LABEL_RULES:
        check(f"refusal rule `{rule}` sees a U+2019 refusal",
              refusal.has_refusal_marker(curly, rule),
              f"{curly!r}")
    check("the ASCII form is unaffected",
          all(refusal.has_refusal_marker(ascii_, r) for r in refusal.LABEL_RULES))
    check("a genuine compliance is still not a refusal",
          not refusal.has_refusal_marker(
              "Sure, here is how to do it: first gather the materials, then proceed.",
              refusal.PRIMARY_RULE))
    # Normalisation must be apostrophes ONLY — folding quotes or dashes as well
    # would be a wider change to the literature's instrument with no shown need.
    n = refusal.normalise_typography
    check("normalisation folds apostrophe variants onto ASCII",
          n("I\u2019m") == "I'm" and n("I\u02bcm") == "I'm" and n("I\u2018m") == "I'm")
    check("normalisation leaves other typography alone",
          n("\u201cquoted\u201d \u2014 dash") == "\u201cquoted\u201d \u2014 dash",
          "curly QUOTES and em-dashes are untouched; only apostrophes fold")


def main() -> int:
    for fn in (t_diff_of_means, t_auc, t_cohens_d, t_effective_rank,
               t_principal_angles, t_canonical_correlations,
               t_stratum_balanced, t_nulls, t_split_half,
               t_auc_ties_regression, t_refusal_typography):
        print(f"\n--- {fn.__name__}")
        try:
            fn()
        except Exception as e:
            import traceback
            FAILURES.append(fn.__name__)
            print(f"[FAIL] {fn.__name__} raised {type(e).__name__}: {e}")
            traceback.print_exc()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} — {FAILURES}")
        return 1
    print("PASS - every estimator reproduces its known answer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
