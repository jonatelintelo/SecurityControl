"""Model-shape introspection that survives nested / multimodal configs.

Qwen3.5 is not a plain causal LM. Verified from the published configs:

    Qwen/Qwen2.5-7B-Instruct  Qwen2ForCausalLM               28 layers, d=3584  (flat config)
    Qwen/Qwen3.5-9B           Qwen3_5ForConditionalGeneration 32 layers, d=4096  (config.text_config)
    Qwen/Qwen3.5-35B-A3B      Qwen3_5MoeForConditionalGeneration 40 layers, d=2048, 256 experts

For the Qwen3.5 family `config.num_hidden_layers` and `config.hidden_size` are
**None** — the real values live under `config.text_config`, alongside a
`vision_config` for the vision tower. Any code doing `model.config.hidden_size`
silently gets `None` on the primary model, and `model.model.layers` may not be
the decoder stack either. Both are resolved here rather than assumed at each
call site.
"""
from typing import Any, List, Optional

import torch.nn as nn

# Ordered by specificity: the first attribute chain that yields a ModuleList of
# decoder blocks wins. Covers plain causal LMs and the multimodal wrappers.
_LAYER_PATHS = (
    "model.layers",
    "model.language_model.layers",
    "language_model.model.layers",
    "model.text_model.layers",
    "transformer.h",
)


def text_config(config: Any) -> Any:
    """The sub-config describing the text decoder.

    Multimodal wrappers nest it under `text_config`; flat causal LMs are their own
    text config. Detected by presence of a real layer count, not by model_type, so
    new architectures do not need to be enumerated.
    """
    nested = getattr(config, "text_config", None)
    if nested is not None and getattr(nested, "num_hidden_layers", None) is not None:
        return nested
    return config


def num_layers(config: Any) -> int:
    n = getattr(text_config(config), "num_hidden_layers", None)
    if n is None:
        raise ValueError(
            "could not resolve num_hidden_layers; config exposes "
            f"{[k for k in vars(config) if 'config' in k or 'layer' in k]}"
        )
    return int(n)


def hidden_size(config: Any) -> int:
    d = getattr(text_config(config), "hidden_size", None)
    if d is None:
        raise ValueError("could not resolve hidden_size from config")
    return int(d)


def num_experts(config: Any) -> Optional[int]:
    """Expert count for MoE models, else None. Used by the deferred MoE arm."""
    return getattr(text_config(config), "num_experts", None)


def _resolve(obj: Any, path: str) -> Optional[Any]:
    cur = obj
    for part in path.split("."):
        cur = getattr(cur, part, None)
        if cur is None:
            return None
    return cur


def find_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder block stack.

    Raises with the paths tried rather than returning something wrong — a hook
    registered on the wrong module produces plausible-looking numbers, which is
    the worst failure mode available.
    """
    for path in _LAYER_PATHS:
        candidate = _resolve(model, path)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return candidate
    raise AttributeError(
        f"could not locate decoder layers on {type(model).__name__}; tried {list(_LAYER_PATHS)}"
    )


def layer_path(model: nn.Module) -> str:
    """Which path `find_layers` resolved — recorded in run manifests."""
    for path in _LAYER_PATHS:
        candidate = _resolve(model, path)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return path
    raise AttributeError(f"could not locate decoder layers on {type(model).__name__}")


def describe(model_id: str, config: Any) -> dict:
    """Shape summary for the run manifest."""
    return {
        "model_id": model_id,
        "architecture": (getattr(config, "architectures", None) or [None])[0],
        "model_type": getattr(config, "model_type", None),
        "num_layers": num_layers(config),
        "hidden_size": hidden_size(config),
        "num_experts": num_experts(config),
        "is_multimodal": getattr(config, "vision_config", None) is not None,
    }
