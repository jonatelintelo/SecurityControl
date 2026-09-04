import torch


class SubspaceEngine:
    """Handles subspace extraction, projection operators, and geometric alignments."""

    @staticmethod
    def extract_direction(pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        """Calculates normalized difference-in-means direction."""
        diff = torch.mean(pos_acts, dim=0) - torch.mean(neg_acts, dim=0)
        return diff / torch.norm(diff, p=2)

    @staticmethod
    def extract_subspace(centered_diffs: torch.Tensor, k: int = 3) -> torch.Tensor:
        """Calculates an orthonormal basis for a k-dimensional subspace using SVD."""
        _, _, Vh = torch.linalg.svd(centered_diffs, full_matrices=False)
        return Vh[:k, :].T

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
    def canonical_subspace_angles(basis_A: torch.Tensor, basis_B: torch.Tensor) -> torch.Tensor:
        """Calculates principal canonical angles (radians) between two subspaces."""
        if basis_A.ndim == 1:
            basis_A = basis_A.unsqueeze(1)
        if basis_B.ndim == 1:
            basis_B = basis_B.unsqueeze(1)
        Qa, _ = torch.linalg.qr(basis_A)
        Qb, _ = torch.linalg.qr(basis_B)
        M = torch.matmul(Qa.T, Qb)
        _, S, _ = torch.linalg.svd(M)
        return torch.acos(torch.clamp(S, -1.0, 1.0))
