import torch
import torch.nn as nn
from typing import List

from core.subspaces import SubspaceEngine


class InterventionEngine:
    """Manages causal steering, NeuroStrike-style neuron ablation, and subspace causal rescue."""

    @staticmethod
    def steer_subspace(layer_module: nn.Module, direction: torch.Tensor, alpha: float, relative: bool = False) -> torch.utils.hooks.RemovableHandle:
        """Residual steering.

        relative=False: h' = h + alpha * unit(direction)   (absolute magnitude)
        relative=True:  h' = h + alpha * ||h|| * unit(direction)

        Use relative=True whenever alpha is compared ACROSS LAYERS. Residual
        norms grow strongly with depth (measured ~8x from layer 10 to 21 on
        Qwen3.5-9B), so a fixed absolute alpha is a much larger perturbation
        early than late, and an apparent "early layers are more vulnerable"
        result would be a norm artifact rather than a property of the safety
        architecture.
        """
        unit_dir = (direction / torch.norm(direction, p=2)).detach()

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            unit = unit_dir.to(h.device).type_as(h)
            if relative:
                scale = h.norm(dim=-1, keepdim=True)  # (batch, seq, 1)
                h_steered = h + alpha * scale * unit
            else:
                h_steered = h + alpha * unit
            return (h_steered,) + output[1:] if isinstance(output, tuple) else h_steered

        return layer_module.register_forward_hook(hook)

    @staticmethod
    def ablate_neurons(mlp_module: nn.Module, neuron_indices: List[int]) -> torch.utils.hooks.RemovableHandle:
        """NeuroStrike-style intervention: zero the selected neurons' contribution
        right before down_proj (i.e. force their output column's contribution to 0)."""
        idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)

        def pre_hook(module, args):
            acts = args[0].clone()
            dev_idx = idx_tensor.to(acts.device)
            acts[..., dev_idx] = 0.0
            return (acts,)

        return mlp_module.down_proj.register_forward_pre_hook(pre_hook)

    @staticmethod
    def install_causal_rescue(layer_module: nn.Module, r_control_basis: torch.Tensor, delta_h: torch.Tensor) -> torch.utils.hooks.RemovableHandle:
        """Executes causal rescue specifically at the prompt boundary to avoid corrupting
        single-token generation steps: re-injects delta_h (projected onto r_control_basis)
        into the last prefill position's residual stream."""
        P_R = SubspaceEngine.get_orthogonal_projector(r_control_basis).detach()

        def hook(module, args, output):
            h = output[0] if isinstance(output, tuple) else output

            # Guard: only apply rescue injection during prefill (prompt processing), not single-token generation
            if h.shape[1] > 1:
                P_R_dev = P_R.to(h.device).type_as(h)
                delta_dev = delta_h.to(h.device).type_as(h)
                rescue_signal = torch.matmul(delta_dev, P_R_dev.T)

                h_rescued = h.clone()
                h_rescued[:, -1, :] += rescue_signal.squeeze()
                return (h_rescued,) + output[1:] if isinstance(output, tuple) else h_rescued

            return output

        return layer_module.register_forward_hook(hook)
