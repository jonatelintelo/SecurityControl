import torch
import torch.nn as nn
from typing import List
from subspaces import SubspaceEngine


class QwenInterventionEngine:
    """Manages causal steering, NeuroStrike neuron ablation, and subspace causal rescue."""

    @staticmethod
    def steer_subspace(layer_module: nn.Module, direction: torch.Tensor, alpha: float) -> torch.utils.hooks.RemovableHandle:
        """Residual steering: h' = h + alpha * r."""
        unit_dir = (direction / torch.norm(direction, p=2)).detach()

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            steer_vec = (alpha * unit_dir).to(h.device).type_as(h)
            h_steered = h + steer_vec
            return (h_steered,) + output[1:] if isinstance(output, tuple) else h_steered

        return layer_module.register_forward_hook(hook)

    @staticmethod
    def ablate_neurons(mlp_module: nn.Module, neuron_indices: List[int]) -> torch.utils.hooks.RemovableHandle:
        idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)

        def pre_hook(module, args):
            acts = args[0].clone()
            dev_idx = idx_tensor.to(acts.device)
            acts[..., dev_idx] = 0.0
            return (acts,)

        return mlp_module.down_proj.register_forward_pre_hook(pre_hook)

    @staticmethod
    def install_causal_rescue(layer_module: nn.Module, r_control_basis: torch.Tensor, delta_h: torch.Tensor) -> torch.utils.hooks.RemovableHandle:
        """
        Executes Causal Rescue:
        h'_l = h_l + P_{R_control} * \Delta h_l
        Restores missing refusal representation downstream while neurons remain ablated.
        """
        P_R = SubspaceEngine.get_orthogonal_projector(r_control_basis).detach()

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            P_R_dev = P_R.to(h.device).type_as(h)
            delta_dev = delta_h.to(h.device).type_as(h)
            rescue_signal = torch.matmul(delta_dev, P_R_dev.T)
            h_rescued = h + rescue_signal
            return (h_rescued,) + output[1:] if isinstance(output, tuple) else h_rescued

        return layer_module.register_forward_hook(hook)
