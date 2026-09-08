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

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# directions
# ---------------------------------------------------------------------------
@dataclass
class Direction:
    """A unit direction plus everything needed to interpret its sign."""
    vector: torch.Tensor
    concept: str
    layer: int
    site: str
    position: str
    positive_class: str
    negative_class: str
    n_positive: int
    n_negative: int
    meta: Dict[str, object] = field(default_factory=dict)

    def project(self, acts: torch.Tensor) -> torch.Tensor:
        """Signed projection. Positive means "toward `positive_class`"."""
        return acts.float() @ self.vector.float()

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
        n_positive=len(positive), n_negative=len(negative),
    )


def random_direction_like(d: Direction, seed: int = 0) -> Direction:
    """Matched random control: same dimensionality, unit norm, no class meaning."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(d.vector.shape, generator=g)
    return Direction(
        vector=v / v.norm(), concept=f"random_{d.concept}", layer=d.layer, site=d.site,
        position=d.position, positive_class="random+", negative_class="random-",
        n_positive=d.n_positive, n_negative=d.n_negative, meta={"seed": seed},
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


def bootstrap_auc_ci(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI. Interval estimates are a standing constraint."""
    g = torch.Generator().manual_seed(seed)
    p, n = positive_scores.float(), negative_scores.float()
    vals = []
    for _ in range(n_boot):
        pi = torch.randint(len(p), (len(p),), generator=g)
        ni = torch.randint(len(n), (len(n),), generator=g)
        vals.append(auc(p[pi], n[ni]))
    v = torch.tensor(vals)
    return float(v.quantile(alpha / 2)), float(v.quantile(1 - alpha / 2))


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


def cosine_matrix(directions: Sequence[Direction]) -> torch.Tensor:
    v = torch.stack([d.vector.float() / d.vector.float().norm() for d in directions])
    return v @ v.T


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
