import torch


class SubspaceEngine:
    """Handles direction/subspace extraction, projection operators, and geometric alignments."""

    @staticmethod
    def extract_direction(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        """Calculates normalized difference-in-means direction."""
        diff = torch.mean(pos_acts, dim=0) - torch.mean(neg_acts, dim=0)
        return diff / torch.norm(diff, p=2)

    @staticmethod
    def extract_subspace(centered_diffs: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Orthonormal basis for a k-dimensional subspace of the per-pair
        difference vectors (columns are basis vectors).

        RQ1 asks about the *dimensionality* of each security variable, and the
        motivation notes refusal may be mediated by several directions rather
        than one axis. The 1-D diff-of-means direction cannot address that, so
        phase 1 reports subspace dimensionality separately.
        """
        _, _, Vh = torch.linalg.svd(centered_diffs, full_matrices=False)
        return Vh[:k, :].T

    @staticmethod
    def spectrum_effective_rank(centered_diffs: torch.Tensor) -> dict:
        """Participation-ratio effective rank of the difference-vector spectrum,
        plus the variance share of the leading direction.

        r_eff near 1 means the concept really is a single axis; substantially
        greater than 1 means the 1-D direction used downstream is a projection
        of something higher-dimensional, which is itself an RQ1 result.
        """
        s = torch.linalg.svdvals(centered_diffs.float())
        lam = s**2
        total = lam.sum()
        if total <= 0:
            return {"r_eff_spectrum": float("nan"), "top1_variance_share": float("nan")}
        return {
            "r_eff_spectrum": float(((lam.sum() ** 2) / (lam**2).sum()).item()),
            "top1_variance_share": float((lam[0] / total).item()),
        }

    @staticmethod
    def principal_angles(basis_A: torch.Tensor, basis_B: torch.Tensor) -> torch.Tensor:
        """Principal canonical angles (radians) between two subspaces — the
        multi-dimensional generalization of cosine similarity, used to compare
        concept subspaces once they are >1-D."""
        if basis_A.ndim == 1:
            basis_A = basis_A.unsqueeze(1)
        if basis_B.ndim == 1:
            basis_B = basis_B.unsqueeze(1)
        Qa, _ = torch.linalg.qr(basis_A)
        Qb, _ = torch.linalg.qr(basis_B)
        _, S, _ = torch.linalg.svd(torch.matmul(Qa.T, Qb))
        return torch.acos(torch.clamp(S, -1.0, 1.0))

    @staticmethod
    def get_orthogonal_projector(basis: torch.Tensor) -> torch.Tensor:
        """
        Constructs orthogonal projector P_R.
        For 1D vector v: P_R = v v^T (outer product)
        For 2D basis Q: P_R = Q Q^T (QR decomposition)
        """
        if basis.ndim == 1:
            v = basis / torch.norm(basis, p=2)
            return torch.outer(v, v)
        elif basis.ndim == 2:
            Q, _ = torch.linalg.qr(basis)
            return torch.matmul(Q, Q.T)
        else:
            raise ValueError(f"Subspace basis must be 1D or 2D tensor, got shape {basis.shape}")

    @staticmethod
    def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
        """Calculates cos(theta) between two 1D vectors."""
        v1_n = v1.view(-1) / torch.norm(v1, p=2)
        v2_n = v2.view(-1) / torch.norm(v2, p=2)
        return torch.dot(v1_n, v2_n).item()

    @staticmethod
    def probe_validation_accuracy(direction: torch.Tensor, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> float:
        """Sign-based classification accuracy of the diff-of-means direction on held-out
        activations: threshold at the midpoint between the two training-set class means'
        projections, then score how often held-out pos/neg projections fall on the
        correct side. Serves as the "probe accuracy" baseline referenced in RQ5."""
        d = direction.view(-1) / torch.norm(direction, p=2)
        pos_proj = pos_acts @ d
        neg_proj = neg_acts @ d
        threshold = 0.5 * (pos_proj.mean() + neg_proj.mean())
        correct = (pos_proj > threshold).sum() + (neg_proj <= threshold).sum()
        total = pos_proj.numel() + neg_proj.numel()
        return (correct / total).item()
