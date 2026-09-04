import torch
import torch.nn as nn
from typing import Dict, List, Optional


class QwenHookEngine:
    """Manages residual stream and intermediate MLP neuron hooks for Qwen models."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.residual_cache: Dict[int, torch.Tensor] = {}
        self.mlp_act_cache: Dict[int, torch.Tensor] = {}
        self.active_hooks: List[torch.utils.hooks.RemovableHandle] = []

        # Resolve Qwen transformer blocks
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        else:
            raise AttributeError("Target model does not expose standard 'model.layers' hierarchy.")

        self.num_layers = len(self.layers)
        self.d_model = model.config.hidden_size

    def hook_residual_stream(self, layers: Optional[List[int]] = None) -> None:
        """Captures transformer block residual output (post-addition)."""
        target_layers = layers or list(range(self.num_layers))
        for l in target_layers:

            def _make_hook(layer_idx: int):
                def hook(module, args, output):
                    h = output[0] if isinstance(output, tuple) else output
                    self.residual_cache[layer_idx] = h.detach()

                return hook

            handle = self.layers[l].register_forward_hook(_make_hook(l))
            self.active_hooks.append(handle)

    def hook_mlp_neurons(self, layers: Optional[List[int]] = None) -> None:
        """Captures activations entering mlp.down_proj: silu(gate) * up."""
        target_layers = layers or list(range(self.num_layers))
        for l in target_layers:
            mlp = self.layers[l].mlp

            def _make_pre_hook(layer_idx: int):
                def hook(module, args):
                    # args[0] shape: (batch, seq, intermediate_size)
                    self.mlp_act_cache[layer_idx] = args[0].detach()

                return hook

            handle = mlp.down_proj.register_forward_pre_hook(_make_pre_hook(l))
            self.active_hooks.append(handle)

    def clear(self) -> None:
        self.residual_cache.clear()
        self.mlp_act_cache.clear()

    def remove_all_hooks(self) -> None:
        for h in self.active_hooks:
            h.remove()
        self.active_hooks.clear()
        self.clear()
