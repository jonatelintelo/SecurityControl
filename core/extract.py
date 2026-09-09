"""Direction and probe estimators.

Two estimators, one per source paper, applied to our crossed corpus:

* :func:`diff_of_means` — Zhao et al.'s estimator for `R_harm` and `R_control`.
* :func:`train_role_probe` — the role paper's multiclass logistic probe, with
  their hyperparameters (L2, ``C=5e-3``, ``max_iter=2000``).

Three constraints from the playbook are enforced here rather than left to callers:

1. **Every direction records which class is positive.** A flipped sign does not
   produce a wrong number, it produces "no effect" — unfalsifiable rather than
   false. It has already flipped once between designs.
2. **Separation is measured offset-free.** A raw mean projection carries a
   per-concept baseline offset that once made two conditions diverge for reasons
   unrelated to the intervention. AUC is rank-based and immune to it.
3. **Eigendecomposition casts to float32.** `eigvalsh` raises on bf16 — verified,
   ``"linalg_eigh_cpu" not implemented for 'BFloat16'``.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# directions
# ---------------------------------------------------------------------------
@dataclass
class Direction:
    """A unit direction plus everything needed to interpret its sign.

    `vector` is unit-length because every *measurement* here (AUC, cosine,
    correlation) is scale-invariant, and a unit vector makes them comparable
    across layers whose residual norms differ by an order of magnitude.

    `raw_norm` keeps the magnitude that normalisation would otherwise throw away:
    `||mean(positive) - mean(negative)||`. **Steering needs it.** Arditi et al.
    (2406.11717) and Zhao et al. (2507.11878) both add the *un-normalised*
    difference-in-means at coefficient 1.0 (`h' = h + v`), which is what makes
    that coefficient meaningful — it shifts an item by exactly one class-mean
    separation. Normalising and then re-scaling by a free parameter, as an earlier
    version of this code did, destroys that scale and replaces a published
    convention with a guess.
    """
    vector: torch.Tensor
    concept: str
    layer: int
    site: str
    position: str
    positive_class: str
    negative_class: str
    n_positive: int
    n_negative: int
    raw_norm: float = 1.0
    meta: Dict[str, object] = field(default_factory=dict)

    def project(self, acts: torch.Tensor) -> torch.Tensor:
        """Signed projection. Positive means "toward `positive_class`"."""
        return acts.float() @ self.vector.float()

    def raw(self) -> torch.Tensor:
        """The un-normalised difference-in-means, i.e. what the papers add at 1.0."""
        return self.vector.float() * self.raw_norm

    def to_record(self) -> Dict[str, object]:
        return {
            "concept": self.concept, "layer": self.layer, "site": self.site,
            "position": self.position, "positive_class": self.positive_class,
            "negative_class": self.negative_class,
            "n_positive": self.n_positive, "n_negative": self.n_negative,
            "norm": float(self.vector.norm()), **self.meta,
        }


def diff_of_means(
    positive: torch.Tensor,
    negative: torch.Tensor,
    concept: str,
    layer: int,
    site: str,
    position: str,
    positive_class: str,
    negative_class: str,
) -> Direction:
    """`v = mean(positive) - mean(negative)`, normalised to unit length.

    Zhao et al.'s estimator, verbatim:
        v^l_harmful = mu^l,t_inst_harmful   - mu^l,t_inst_harmless
        v^l_refuse  = mu^l,t_post-inst_refuse - mu^l,t_post-inst_accept
    """
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError(f"empty class for {concept}@L{layer}: "
                         f"pos={positive.shape} neg={negative.shape}")
    v = positive.float().mean(0) - negative.float().mean(0)
    n = v.norm()
    if n < 1e-8:
        raise ValueError(f"degenerate direction for {concept}@L{layer} (norm={n:.2e})")
    return Direction(
        vector=v / n, concept=concept, layer=layer, site=site, position=position,
        positive_class=positive_class, negative_class=negative_class,
        n_positive=len(positive), n_negative=len(negative), raw_norm=float(n),
    )


def stratum_balanced_diff_of_means(
    positive: torch.Tensor,
    negative: torch.Tensor,
    positive_strata: Sequence[str],
    negative_strata: Sequence[str],
    concept: str,
    layer: int,
    site: str,
    position: str,
    positive_class: str,
    negative_class: str,
) -> Tuple[Direction, Dict[str, object]]:
    """Difference-of-means with the two sides **post-stratified** to one common
    stratum distribution.

    Why `R_control` cannot use the plain estimator. It contrasts `refused` vs
    `complied` within harmful, pooled over role — but refusal rate varies by role,
    and the corpus-widening rule deliberately exploits that to populate the
    compliant cell. So P(role | refused) != P(role | complied), and a plain
    difference of means carries a role-composition component: the resulting
    direction is partly a role direction. That would inflate cos(R_control, R_role)
    — the headline RQ1 geometry quantity — purely by construction, and would weaken
    E1.6's gate, since steering role would then move R_control for definitional
    reasons.

    `R_harm` needs none of this: every instruction is rendered under all four
    roles, so role composition is identical on both sides of that contrast.

    Post-stratification rather than subsampling, because the compliant cell is the
    scarce one and discarding from it to match margins would cost exactly the data
    the variable is identified from. Each item is weighted so that both sides carry
    the **pooled** stratum distribution:

        w_i = target[s_i] / n_side[s_i],   normalised to sum to 1 within a side

    Strata present on only one side cannot be balanced and are dropped; which ones,
    and how much weight that removes, is returned rather than silently absorbed.
    """
    pos_s, neg_s = list(positive_strata), list(negative_strata)
    if len(pos_s) != len(positive) or len(neg_s) != len(negative):
        raise ValueError("stratum labels must align with the activation rows")

    pos_counts = Counter(pos_s)
    neg_counts = Counter(neg_s)
    shared = sorted(set(pos_counts) & set(neg_counts))
    dropped = sorted((set(pos_counts) | set(neg_counts)) - set(shared))
    if not shared:
        raise ValueError(f"no stratum present on both sides for {concept}@L{layer}: "
                         f"pos={sorted(pos_counts)} neg={sorted(neg_counts)}")

    # Target = pooled distribution over the shared strata. Using the pooled
    # marginal rather than a uniform one avoids upweighting a rare stratum into
    # dominating the estimate.
    total = sum(pos_counts[s] + neg_counts[s] for s in shared)
    target = {s: (pos_counts[s] + neg_counts[s]) / total for s in shared}

    def weighted_mean(x: torch.Tensor, strata: Sequence[str], counts: Counter) -> torch.Tensor:
        w = torch.tensor([target.get(s, 0.0) / counts[s] if s in target else 0.0
                          for s in strata], dtype=torch.float32)
        if float(w.sum()) <= 0:
            raise ValueError(f"all weight removed for {concept}@L{layer}")
        w = w / w.sum()
        return (x.float() * w.unsqueeze(1)).sum(0)

    v = (weighted_mean(positive, pos_s, pos_counts)
         - weighted_mean(negative, neg_s, neg_counts))
    n = v.norm()
    if n < 1e-8:
        raise ValueError(f"degenerate direction for {concept}@L{layer} (norm={n:.2e})")

    kept_pos = sum(pos_counts[s] for s in shared)
    kept_neg = sum(neg_counts[s] for s in shared)
    info = {
        "balanced_on": "stratum",
        "shared_strata": shared,
        "dropped_strata": dropped,
        "target_distribution": {s: round(target[s], 4) for s in shared},
        "positive_counts": dict(pos_counts),
        "negative_counts": dict(neg_counts),
        "weight_retained_positive": round(kept_pos / max(len(pos_s), 1), 4),
        "weight_retained_negative": round(kept_neg / max(len(neg_s), 1), 4),
    }
    return Direction(
        vector=v / n, concept=concept, layer=layer, site=site, position=position,
        positive_class=positive_class, negative_class=negative_class,
        n_positive=kept_pos, n_negative=kept_neg, raw_norm=float(n),
    ), info


def composition_imbalance(positive_strata: Sequence[str],
                          negative_strata: Sequence[str]) -> Dict[str, float]:
    """Total-variation distance between the two sides' stratum distributions.

    0 means the confound this balancing addresses is absent; large values mean the
    plain estimator would have encoded the stratum. Reported for every contrast,
    including the ones that need no balancing, so that "R_harm is unconfounded by
    role" is a measured statement rather than an assumed one.
    """
    pc, nc = Counter(positive_strata), Counter(negative_strata)
    np_, nn = max(sum(pc.values()), 1), max(sum(nc.values()), 1)
    keys = set(pc) | set(nc)
    tv = 0.5 * sum(abs(pc.get(k, 0) / np_ - nc.get(k, 0) / nn) for k in keys)
    return {"total_variation": round(float(tv), 4),
            "positive_shares": {k: round(pc.get(k, 0) / np_, 3) for k in sorted(keys)},
            "negative_shares": {k: round(nc.get(k, 0) / nn, 3) for k in sorted(keys)}}


def random_direction_like(d: Direction, seed: int = 0) -> Direction:
    """Matched random control: same dimensionality *and same magnitude*, no meaning.

    The magnitude match matters for steering. Interventions add the raw vector at
    coefficient 1.0, so a unit-norm random control would be a far *smaller*
    perturbation than the real direction (whose `raw_norm` is typically many
    units), and it would look inert for a reason that has nothing to do with its
    lack of semantic content. Matching `raw_norm` makes the comparison answer the
    intended question: same size of push, different direction.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d.vector.shape, generator=g)
    return Direction(
        vector=v / v.norm(), concept=f"random_{d.concept}", layer=d.layer, site=d.site,
        position=d.position, positive_class="random+", negative_class="random-",
        n_positive=d.n_positive, n_negative=d.n_negative, raw_norm=d.raw_norm,
        meta={"seed": seed},
    )


# ---------------------------------------------------------------------------
# separation, measured offset-free
# ---------------------------------------------------------------------------
def auc(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    """Probability a random positive outranks a random negative.

    Rank-based, so a constant offset in the projection cannot change it. 0.5 is
    chance; below 0.5 means the direction points the wrong way, which is
    information rather than an error.
    """
    p = positive_scores.detach().float().flatten()
    n = negative_scores.detach().float().flatten()
    if p.numel() == 0 or n.numel() == 0:
        return float("nan")
    allv = torch.cat([p, n])
    ranks = allv.argsort().argsort().float() + 1.0
    r_pos = ranks[: p.numel()].sum()
    n_p, n_n = float(p.numel()), float(n.numel())
    return float((r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))


def cohens_d(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> float:
    p, n = positive_scores.float(), negative_scores.float()
    if p.numel() < 2 or n.numel() < 2:
        return float("nan")
    pooled = torch.sqrt(((p.numel() - 1) * p.var() + (n.numel() - 1) * n.var())
                        / (p.numel() + n.numel() - 2))
    return float((p.mean() - n.mean()) / pooled) if pooled > 1e-12 else float("nan")


def separation(direction: Direction, positive: torch.Tensor, negative: torch.Tensor) -> Dict[str, float]:
    ps, ns = direction.project(positive), direction.project(negative)
    return {"auc": auc(ps, ns), "cohens_d": cohens_d(ps, ns),
            "mean_pos": float(ps.mean()), "mean_neg": float(ns.mean())}


# ---------------------------------------------------------------------------
# dimensionality (E1.4)
# ---------------------------------------------------------------------------
def effective_rank(x: torch.Tensor) -> float:
    """Participation ratio of the covariance eigenspectrum, STRUCT's `r_eff`:

        r_eff = (sum lambda_i)^2 / sum lambda_i^2

    ~1 means a single dominant axis; >>1 means a genuine subspace. The Motivation
    raises exactly this for refusal. Cast to float32 first — eigvalsh raises on
    bf16 and would otherwise crash mid-run.
    """
    xf = x.detach().float()
    xc = xf - xf.mean(0, keepdim=True)
    cov = (xc.T @ xc) / max(len(xc) - 1, 1)
    ev = torch.linalg.eigvalsh(cov).clamp(min=0)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 1e-20 else float("nan")


def separation_after_projecting_out(
    direction: Direction, positive: torch.Tensor, negative: torch.Tensor
) -> Dict[str, float]:
    """Separation remaining once the top direction is removed.

    If a concept is genuinely one-dimensional this collapses to chance; if
    separation survives, the concept occupies a subspace. Complements `r_eff` by
    asking a causal-flavoured question instead of a spectral one.
    """
    v = direction.vector.float()
    strip = lambda a: a.float() - (a.float() @ v).unsqueeze(1) * v.unsqueeze(0)
    p, n = strip(positive), strip(negative)
    resid = diff_of_means(p, n, f"{direction.concept}_resid", direction.layer,
                          direction.site, direction.position,
                          direction.positive_class, direction.negative_class)
    return separation(resid, p, n)


# ---------------------------------------------------------------------------
# role probe (the role paper's estimator)
# ---------------------------------------------------------------------------
@dataclass
class RoleProbe:
    """Multiclass logistic probe over role classes at one layer."""
    classes: List[str]
    coef: torch.Tensor      # [n_classes, d]
    intercept: torch.Tensor  # [n_classes]
    layer: int
    site: str
    accuracy: float
    n_train: int
    n_test: int

    def contrast(self, positive_role: str, negative_role: str) -> Direction:
        """`w_i - w_j`, the decision boundary between two roles.

        For a softmax probe this is exactly the axis separating the two classes,
        so it is the principled analogue of a difference-of-means and is what
        makes the role probe comparable with `R_harm` / `R_control` in E1.2.
        """
        i, j = self.classes.index(positive_role), self.classes.index(negative_role)
        v = (self.coef[i] - self.coef[j]).float()
        n = v.norm()
        if n < 1e-8:
            raise ValueError(f"degenerate role contrast {positive_role}-{negative_role}")
        return Direction(
            vector=v / n, concept=f"role_{positive_role}_vs_{negative_role}",
            layer=self.layer, site=self.site, position="content_tokens",
            positive_class=positive_role, negative_class=negative_role,
            n_positive=self.n_train, n_negative=self.n_train,
            # A probe weight difference is a decision boundary, not a
            # difference-in-means, so it carries no activation-space magnitude and
            # must NOT be steered with at coefficient 1.0. Use a diff-of-means
            # direction for interventions; this one is for geometry only.
            raw_norm=float("nan"),
            meta={"probe_accuracy": self.accuracy},
        )


def train_role_probe(
    x_train: torch.Tensor,
    y_train: Sequence[str],
    x_test: torch.Tensor,
    y_test: Sequence[str],
    layer: int,
    site: str,
    C: float = 5e-3,
    max_iter: int = 2000,
    seed: int = 0,
) -> RoleProbe:
    """The role paper's demo settings: L2 logistic regression, `C=5e-3`,
    `max_iter=2000`, no feature normalisation, multinomial over role classes."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(penalty="l2", C=C, max_iter=max_iter,
                             fit_intercept=True, random_state=seed)
    clf.fit(x_train.float().numpy(), list(y_train))
    acc = float(clf.score(x_test.float().numpy(), list(y_test)))
    return RoleProbe(
        classes=list(clf.classes_),
        coef=torch.tensor(np.asarray(clf.coef_), dtype=torch.float32),
        intercept=torch.tensor(np.asarray(clf.intercept_), dtype=torch.float32),
        layer=layer, site=site, accuracy=acc,
        n_train=len(y_train), n_test=len(y_test),
    )


def probe_transfer_accuracy(probe: RoleProbe, x: torch.Tensor, y: Sequence[str]) -> float:
    """Apply a probe to a different corpus — the cross-corpus transfer check.

    If a probe trained on our crossed corpus (instructions in role tags) fails to
    classify roles on the role paper's constant-content webtext, it is reading
    "instruction-ness" rather than role, and must be rebuilt.
    """
    logits = x.float() @ probe.coef.T + probe.intercept
    pred = [probe.classes[i] for i in logits.argmax(dim=1).tolist()]
    labels = list(y)
    if not labels:
        return float("nan")
    return sum(p == t for p, t in zip(pred, labels)) / len(labels)


def random_cosine_band(d_model: int, n_samples: int = 2000, seed: int = 0) -> Dict[str, float]:
    """Null band for E1.2.

    Cosines between random unit vectors in `R^d` concentrate near 0 with
    sd ~ 1/sqrt(d). Without this band, "harmfulness is anti-aligned with the
    system role" is eyeballing rather than a claim.
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(n_samples, d_model, generator=g)
    b = torch.randn(n_samples, d_model, generator=g)
    a, b = a / a.norm(dim=1, keepdim=True), b / b.norm(dim=1, keepdim=True)
    c = (a * b).sum(1)
    return {"mean": float(c.mean()), "sd": float(c.std()),
            "p2.5": float(c.quantile(0.025)), "p97.5": float(c.quantile(0.975)),
            "analytic_sd": float(1.0 / np.sqrt(d_model))}


# ---------------------------------------------------------------------------
# controls required by the playbook
# ---------------------------------------------------------------------------
def balanced_auc(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    n_boot: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """AUC on class-balanced subsamples.

    `R_control`'s classes are wildly unbalanced by construction — within harmful,
    Qwen2.5-7B refuses 1,180 and complies 102 — and AUC on a heavily skewed set is
    easy to misread. Subsample the larger class to the smaller, repeatedly, and
    report the mean with a percentile interval.
    """
    g = torch.Generator().manual_seed(seed)
    p, n = positive_scores.float(), negative_scores.float()
    k = min(len(p), len(n))
    if k < 2:
        return {"auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n_per_class": k}
    vals = []
    for _ in range(n_boot):
        pi = torch.randperm(len(p), generator=g)[:k]
        ni = torch.randperm(len(n), generator=g)[:k]
        vals.append(auc(p[pi], n[ni]))
    v = torch.tensor(vals)
    return {"auc": float(v.mean()), "ci_low": float(v.quantile(0.025)),
            "ci_high": float(v.quantile(0.975)), "n_per_class": k}


def split_half_stability(
    positive: torch.Tensor,
    negative: torch.Tensor,
    n_splits: int = 20,
    seed: int = 0,
) -> Dict[str, float]:
    """Cosine between directions fitted on disjoint halves of the same data.

    This is the **noise floor**: no cross-concept cosine can be called meaningful
    unless it is large relative to how much a direction varies against itself.
    A single fit gives no stability estimate at all.
    """
    g = torch.Generator().manual_seed(seed)
    cs = []
    for _ in range(n_splits):
        pi = torch.randperm(len(positive), generator=g)
        ni = torch.randperm(len(negative), generator=g)
        ph, nh = len(positive) // 2, len(negative) // 2
        if ph < 2 or nh < 2:
            return {"mean": float("nan"), "sd": float("nan"), "n_splits": 0}
        a = positive[pi[:ph]].float().mean(0) - negative[ni[:nh]].float().mean(0)
        b = positive[pi[ph:]].float().mean(0) - negative[ni[nh:]].float().mean(0)
        if a.norm() < 1e-8 or b.norm() < 1e-8:
            continue
        cs.append(float((a / a.norm()) @ (b / b.norm())))
    if not cs:
        return {"mean": float("nan"), "sd": float("nan"), "n_splits": 0}
    t = torch.tensor(cs)
    return {"mean": float(t.mean()), "sd": float(t.std()), "n_splits": len(cs)}


def length_only_baseline(
    lengths: Sequence[float],
    labels: Sequence[bool],
) -> Dict[str, float]:
    """How much of a contrast is explained by sequence length alone.

    Required for both `R_role` (authentic markup differs by role: the
    `<tool_response>` wrapper and Qwen3.5's think block give a 15-20 token spread)
    and `R_harm` (harmful instructions run longer: median 14 words vs 9). A
    direction that does not beat this is measuring length, not the concept.
    Reported as AUC so it is directly comparable with the direction's own AUC.
    """
    x = torch.tensor([float(v) for v in lengths], dtype=torch.float32)
    y = torch.tensor([bool(v) for v in labels], dtype=torch.bool)
    if y.numel() == 0 or y.sum() < 2 or (~y).sum() < 2:
        return {"auc": float("nan"), "n_pos": int(y.sum()), "n_neg": int((~y).sum())}
    return {"auc": auc(x[y], x[~y]), "n_pos": int(y.sum()), "n_neg": int((~y).sum())}


def length_matched_mask(
    lengths: Sequence[float],
    labels: Sequence[bool],
    n_bins: int = 12,
    seed: int = 0,
) -> torch.Tensor:
    """Boolean mask selecting a subset whose length distribution matches across classes.

    Needed because the authentic role markup makes length itself informative: the
    `<tool_response>` wrapper adds ~9 tokens, and measured on Qwen2.5-7B a
    length-only classifier separates tool from user at **AUC 0.95** — beating the
    role direction's own 0.879. Without matching, "the model represents role"
    cannot be distinguished from "the model tracks sequence length".

    Instruction lengths vary enough (10-71 tokens) that the class length
    distributions overlap, so a matched subset exists. Within each length bin we
    keep an equal number from each class, sampled deterministically.
    """
    x = torch.tensor([float(v) for v in lengths], dtype=torch.float32)
    y = torch.tensor([bool(v) for v in labels], dtype=torch.bool)
    keep = torch.zeros(len(x), dtype=torch.bool)
    if y.sum() == 0 or (~y).sum() == 0:
        return keep

    g = torch.Generator().manual_seed(seed)
    edges = torch.quantile(x, torch.linspace(0, 1, n_bins + 1))
    edges[0] -= 1.0
    for i in range(n_bins):
        in_bin = (x > edges[i]) & (x <= edges[i + 1])
        pos = torch.nonzero(in_bin & y).flatten()
        neg = torch.nonzero(in_bin & ~y).flatten()
        k = min(len(pos), len(neg))
        if k == 0:
            continue
        keep[pos[torch.randperm(len(pos), generator=g)[:k]]] = True
        keep[neg[torch.randperm(len(neg), generator=g)[:k]]] = True
    return keep


# ---------------------------------------------------------------------------
# cluster-aware resampling
# ---------------------------------------------------------------------------
def cluster_bootstrap_auc(
    scores: torch.Tensor,
    labels: Sequence[bool],
    clusters: Sequence,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Bootstrap AUC by resampling **clusters**, not observations.

    Observations here are not independent. Each instruction is rendered under 4
    roles x 2 designs, so an 800-item test split contains only 100 distinct
    instructions; the role probe is worse still, fitted per *token*, so 8 tokens
    per rendering means ~3,200 tokens over the same 100 instructions. Resampling
    observations treats these as independent and produces intervals that are far
    too narrow — the point estimate is unaffected, the uncertainty is understated.

    Clustering by instruction `uid` also respects the paired design: the same
    instruction appears on both sides of a role contrast, so resampling a uid
    carries its tool rendering and its user rendering together.

    `n_clusters` is the honest sample size and is reported alongside.
    """
    s = scores.detach().float().flatten()
    y = torch.tensor([bool(v) for v in labels], dtype=torch.bool)
    cl = list(clusters)
    if len(s) != len(y) or len(s) != len(cl):
        raise ValueError(f"length mismatch: scores={len(s)} labels={len(y)} clusters={len(cl)}")

    by_cluster: Dict[object, List[int]] = {}
    for i, c in enumerate(cl):
        by_cluster.setdefault(c, []).append(i)
    keys = sorted(by_cluster, key=str)
    if len(keys) < 2:
        return {"auc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "n_clusters": len(keys), "n_obs": len(s)}

    point = auc(s[y], s[~y]) if (y.any() and (~y).any()) else float("nan")

    g = torch.Generator().manual_seed(seed)
    vals: List[float] = []
    for _ in range(n_boot):
        pick = torch.randint(len(keys), (len(keys),), generator=g).tolist()
        idx = [i for k in pick for i in by_cluster[keys[k]]]
        if not idx:
            continue
        ss, yy = s[idx], y[idx]
        if yy.any() and (~yy).any():
            vals.append(auc(ss[yy], ss[~yy]))
    if not vals:
        return {"auc": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_clusters": len(keys), "n_obs": len(s)}
    v = torch.tensor(vals)
    return {"auc": point, "ci_low": float(v.quantile(alpha / 2)),
            "ci_high": float(v.quantile(1 - alpha / 2)),
            "n_clusters": len(keys), "n_obs": len(s)}


def cluster_bootstrap_accuracy(
    correct: Sequence[bool],
    clusters: Sequence,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Cluster-aware CI for the role probe's multiclass accuracy.

    Same reasoning as `cluster_bootstrap_auc`: tokens within an instruction are
    not independent draws.
    """
    ok = torch.tensor([bool(v) for v in correct], dtype=torch.bool)
    cl = list(clusters)
    if len(ok) != len(cl):
        raise ValueError(f"length mismatch: correct={len(ok)} clusters={len(cl)}")

    by_cluster: Dict[object, List[int]] = {}
    for i, c in enumerate(cl):
        by_cluster.setdefault(c, []).append(i)
    keys = sorted(by_cluster, key=str)
    point = float(ok.float().mean()) if len(ok) else float("nan")
    if len(keys) < 2:
        return {"accuracy": point, "ci_low": float("nan"), "ci_high": float("nan"),
                "n_clusters": len(keys), "n_obs": len(ok)}

    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_boot):
        pick = torch.randint(len(keys), (len(keys),), generator=g).tolist()
        idx = [i for k in pick for i in by_cluster[keys[k]]]
        if idx:
            vals.append(float(ok[idx].float().mean()))
    v = torch.tensor(vals)
    return {"accuracy": point, "ci_low": float(v.quantile(alpha / 2)),
            "ci_high": float(v.quantile(1 - alpha / 2)),
            "n_clusters": len(keys), "n_obs": len(ok)}


def random_direction_null(
    positive: torch.Tensor,
    negative: torch.Tensor,
    n_directions: int = 200,
    seed: int = 0,
) -> Dict[str, float]:
    """Null distribution of AUC over many random unit directions.

    A **single** random direction is not a usable control. High-dimensional
    activations are strongly anisotropic, so the null's median sits at 0.50 as
    expected but its upper tail runs far above it: one draw can separate the
    classes by luck, and a single-draw baseline is as likely to flatter a
    direction as to challenge it. Only the null DISTRIBUTION is interpretable.

    Reports the null's upper tail and an empirical p-value: the fraction of random
    directions scoring at least as high as the real one. A direction is only
    credible if it sits beyond the null's p95, not merely above one random draw.
    """
    g = torch.Generator().manual_seed(seed)
    d_model = positive.shape[1]
    vals = []
    p, n = positive.float(), negative.float()
    for _ in range(n_directions):
        v = torch.randn(d_model, generator=g)
        v = v / v.norm()
        vals.append(auc(p @ v, n @ v))
    t = torch.tensor(vals)
    return {"null_median": float(t.median()), "null_p95": float(t.quantile(0.95)),
            "null_max": float(t.max()), "null_min": float(t.min()),
            "n_directions": n_directions}


def null_pvalue(observed: float, positive: torch.Tensor, negative: torch.Tensor,
                n_directions: int = 200, seed: int = 0) -> float:
    """Fraction of random directions reaching at least `observed` AUC."""
    g = torch.Generator().manual_seed(seed)
    p, n = positive.float(), negative.float()
    hits = 0
    for _ in range(n_directions):
        v = torch.randn(positive.shape[1], generator=g)
        if auc(p @ (v / v.norm()), n @ (v / v.norm())) >= observed:
            hits += 1
    return (hits + 1) / (n_directions + 1)
