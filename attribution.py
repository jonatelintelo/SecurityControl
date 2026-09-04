import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from subspaces import SubspaceEngine


class QwenAttributionEngine:
    """Calculates directional neuron projections and structural architectural metrics."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = model.model.layers

    def compute_neuron_attributions(self, layer_idx: int, neuron_activations: torch.Tensor, subspace_basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Projects each Qwen MLP neuron output into residual subspace R:
        C_{i,l}^R = E [ || P_R (a_{i,l}(h) * w_{i,l}^out) ||_2 ]
        Returns:
            raw_contributions: (intermediate_size,)
            projected_w: (hidden_size, intermediate_size)
        """
        w_down = self.layers[layer_idx].mlp.down_proj.weight.detach()
        P_R = SubspaceEngine.get_orthogonal_projector(subspace_basis).to(w_down.device).type_as(w_down)

        projected_w = torch.matmul(P_R, w_down)
        w_norms = torch.norm(projected_w, p=2, dim=0)

        intermediate_size = w_down.shape[1]
        acts_flat = neuron_activations.reshape(-1, intermediate_size).to(w_down.device)
        mean_acts = torch.mean(torch.abs(acts_flat), dim=0)

        raw_contributions = (mean_acts * w_norms).cpu()
        return raw_contributions, projected_w.cpu()

    @staticmethod
    def compute_architectural_quantities(raw_contributions: torch.Tensor, projected_w: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Calculates C_R, N_eff, r_eff, and k_50 as defined in Section 1.4."""
        total = torch.sum(raw_contributions)
        if total == 0:
            return {"C_R": 0.0, "N_eff": 0.0, "r_eff": 0.0, "k_50": 0.0}

        # Normalized contributions: p_i = |w_{i,R}| / sum |w_{j,R}|
        p = raw_contributions / total

        # Functional concentration C_R = sum(p_i^2)
        C_R = torch.sum(p**2).item()
        N_eff = 1.0 / C_R if C_R > 0 else 0.0

        # Effective functional rank r_eff = (sum \lambda_i)^2 / sum \lambda_i^2
        r_eff = 1.0
        if projected_w is not None:
            # Gram matrix of active component projection vectors: G = W_proj^T @ W_proj
            # Truncated to top active components to save memory
            top_indices = torch.topk(raw_contributions, k=min(256, len(raw_contributions))).indices
            W_top = projected_w[:, top_indices]
            gram = torch.matmul(W_top.T, W_top)
            evals = torch.linalg.eigvalsh(gram)
            evals = torch.clamp(evals, min=1e-12)
            sum_evals = torch.sum(evals)
            sum_sq_evals = torch.sum(evals**2)
            r_eff = ((sum_evals**2) / sum_sq_evals).item()

        # Causal redundancy: k_50
        sorted_p, _ = torch.sort(p, descending=True)
        cum_sum = torch.cumsum(sorted_p, dim=0)
        k_50_indices = torch.where(cum_sum >= 0.5)[0]
        k_50 = float(k_50_indices[0].item() + 1) if len(k_50_indices) > 0 else float(len(sorted_p))

        return {
            "functional_concentration_C_R": C_R,
            "effective_component_count_N_eff": N_eff,
            "effective_functional_rank_r_eff": r_eff,
            "causal_redundancy_k_50": k_50,
        }
