import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from core.subspaces import SubspaceEngine


class AttributionEngine:
    """Calculates directional neuron projections and structural architectural metrics."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.layers = model.model.layers

    def compute_static_weight_alignment(self, layer_idx: int, target_direction: torch.Tensor) -> torch.Tensor:
        """Pure-weights cosine alignment between each neuron's down_proj output column
        and the target direction — no activation data involved. This is the "can we
        find the same safety neurons just by measuring alignment" baseline."""
        w_down = self.layers[layer_idx].mlp.down_proj.weight.detach()
        w_norm = w_down / torch.clamp(torch.norm(w_down, p=2, dim=0, keepdim=True), min=1e-9)
        v_norm = (target_direction / torch.clamp(torch.norm(target_direction, p=2), min=1e-9)).to(w_down.device).type_as(w_down)
        return torch.abs(torch.matmul(v_norm, w_norm)).float().cpu()

    def compute_neuron_attributions(self, layer_idx: int, neuron_activations: torch.Tensor, subspace_basis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Projects each MLP neuron's output column into residual subspace R and scales
        by the neuron's mean |activation| on the probe batch (causal/activation-weighted
        attribution — combines "does this neuron write into R" with "is it actually firing")."""
        w_down = self.layers[layer_idx].mlp.down_proj.weight.detach()
        P_R = SubspaceEngine.get_orthogonal_projector(subspace_basis).to(w_down.device).type_as(w_down)

        projected_w = torch.matmul(P_R, w_down)
        w_norms = torch.norm(projected_w, p=2, dim=0)

        intermediate_size = w_down.shape[1]
        acts_flat = neuron_activations.reshape(-1, intermediate_size).to(w_down.device)
        mean_acts = torch.mean(torch.abs(acts_flat), dim=0)

        raw_contributions = (mean_acts * w_norms).float().cpu()

        # Return the unprojected but activation-scaled weights to correctly measure effective rank
        active_w = w_down * mean_acts
        return raw_contributions, active_w.float().cpu()

    def compute_activation_diff_ranking(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor) -> torch.Tensor:
        """NeuroStrike-style neuron ranking: |mean(|act|) on positive prompts - mean(|act|)
        on negative prompts| per neuron, using ONLY activation contrasts (no weight
        geometry at all). Serves as the independent ground-truth ranking that
        compute_static_weight_alignment and compute_neuron_attributions are compared
        against in phase 3."""
        pos_flat = pos_acts.reshape(-1, pos_acts.shape[-1])
        neg_flat = neg_acts.reshape(-1, neg_acts.shape[-1])
        diff = torch.mean(torch.abs(pos_flat), dim=0) - torch.mean(torch.abs(neg_flat), dim=0)
        return torch.abs(diff).float().cpu()

    @staticmethod
    def compute_architectural_quantities(raw_contributions: torch.Tensor, active_w: Optional[torch.Tensor] = None, gram_top_k: int = 256) -> Dict[str, float]:
        """Calculates C_R (functional concentration), N_eff (effective component count),
        and r_eff (effective functional rank). Causal redundancy (k_50) is evaluated
        empirically via core.metrics.component_ablation_curve, not here."""
        total = torch.sum(raw_contributions)
        if total == 0:
            return {"functional_concentration_C_R": 0.0, "effective_component_count_N_eff": 0.0, "effective_functional_rank_r_eff": 0.0}

        p = raw_contributions / total
        C_R = torch.sum(p**2).item()
        N_eff = 1.0 / C_R if C_R > 0 else 0.0

        r_eff = 1.0
        if active_w is not None:
            # eigvalsh has no CPU bf16 kernel; active_w/raw_contributions are float32
            # from the .float() casts above, but be defensive against future callers.
            active_w = active_w.float()
            top_indices = torch.topk(raw_contributions, k=min(gram_top_k, raw_contributions.numel())).indices
            W_top = active_w[:, top_indices]
            gram = torch.matmul(W_top.T, W_top)
            evals = torch.linalg.eigvalsh(gram)
            evals = torch.clamp(evals, min=1e-12)
            sum_evals = torch.sum(evals)
            sum_sq_evals = torch.sum(evals**2)
            r_eff = ((sum_evals**2) / sum_sq_evals).item()

        return {"functional_concentration_C_R": C_R, "effective_component_count_N_eff": N_eff, "effective_functional_rank_r_eff": r_eff}
