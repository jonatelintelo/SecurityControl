import torch
import torch.nn as nn
from typing import Dict, List, Optional

TensorDict = Dict[int, torch.Tensor]


class HookEngine:
    """Manages residual-stream and MLP-neuron forward hooks on a Llama-style
    decoder (any model exposing model.model.layers[i].mlp.{gate,up,down}_proj —
    covers Qwen, Llama, Mistral, and most other dense HF causal LMs)."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.residual_cache: Dict[int, torch.Tensor] = {}
        self.mlp_act_cache: Dict[int, torch.Tensor] = {}
        self.active_hooks: List[torch.utils.hooks.RemovableHandle] = []

        if hasattr(model, "model") and hasattr(model.model, "layers"):
            self.layers = model.model.layers
        else:
            raise AttributeError("Target model does not expose a standard 'model.layers' hierarchy.")

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


def capture_last_token_residuals(model, hook_engine: HookEngine, tokenizer, texts: List[str], device: str, layers: Optional[List[int]] = None) -> TensorDict:
    """Batched forward pass over `texts`, returning the last-token residual-stream
    activation at each requested layer as a [len(texts), d_model] tensor. Requires
    a left-padding tokenizer (set in core.model_io.load_model) so index -1 is the
    true last token regardless of each sequence's length. Removes its own hooks
    before returning so callers don't need to manage HookEngine state."""
    target_layers = layers if layers is not None else list(range(hook_engine.num_layers))
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)

    hook_engine.hook_residual_stream(target_layers)
    with torch.no_grad():
        _ = model(**inputs)
    acts = {l: hook_engine.residual_cache[l][:, -1, :].detach().cpu() for l in target_layers}
    hook_engine.remove_all_hooks()
    return acts


def capture_mlp_neuron_activations(model, hook_engine: HookEngine, tokenizer, texts: List[str], device: str, layer: int) -> torch.Tensor:
    """Batched forward pass capturing pre-down_proj MLP activations at one layer,
    masked to real (non-padding) token positions and flattened to
    [num_valid_tokens, intermediate_size]. Padding-aware version of the
    single-prompt capture in the original codebase — needed once prompts are
    batched with left-padding, since padded positions carry meaningless
    activations that would otherwise bias the mean used for attribution."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
    hook_engine.hook_mlp_neurons([layer])
    with torch.no_grad():
        _ = model(**inputs)
    acts = hook_engine.mlp_act_cache[layer]  # (batch, seq, intermediate)
    mask = inputs["attention_mask"].bool()
    valid = acts[mask]  # (num_valid_tokens, intermediate)
    hook_engine.remove_all_hooks()
    return valid.detach().cpu()
