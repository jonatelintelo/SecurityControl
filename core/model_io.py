"""Model and tokenizer loading.

Rewritten for the Qwen3.5 family, whose configs are nested under `text_config`
and whose checkpoints are `*ForConditionalGeneration` wrappers around a text
decoder. The previous version read `model.config.hidden_size` directly, which is
`None` on those models, and assumed `model.model.layers` for the decoder stack.
Both go through `core.model_meta` now.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from core import model_meta


def load_tokenizer(model_id: str):
    """Left padding is required: batched extraction reads `t_post-inst` at index
    -1, and right padding would put pad tokens there."""
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(
    model_id: str,
    device: str = "cuda",
    dtype: Optional[torch.dtype] = None,
    logger=None,
) -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    dtype = dtype or (torch.bfloat16 if device.startswith("cuda") else torch.float32)
    tok = load_tokenizer(model_id)

    cfg = AutoConfig.from_pretrained(model_id)
    shape = model_meta.describe(model_id, cfg)
    if logger:
        logger.info(f"Loading {model_id}: {shape}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, device_map=device if device.startswith("cuda") else None,
        )
    except (ValueError, KeyError):
        # Multimodal checkpoints may not register under AutoModelForCausalLM.
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_id, dtype=dtype, device_map=device if device.startswith("cuda") else None,
        )
    if not device.startswith("cuda"):
        model = model.to(device)
    model.eval()

    # Fail here, loudly, rather than at the first hook: a hook on the wrong module
    # returns plausible activations and silently invalidates everything downstream.
    layers = model_meta.find_layers(model)
    if len(layers) != shape["num_layers"]:
        raise RuntimeError(
            f"located {len(layers)} decoder layers via '{model_meta.layer_path(model)}' "
            f"but config declares {shape['num_layers']}"
        )
    if logger:
        logger.info(f"Decoder stack at '{model_meta.layer_path(model)}' ({len(layers)} layers)")
    return model, tok


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


"""Note: thread capping deliberately does NOT live here.

It must happen before torch initialises its native libraries, and this module
imports torch at module level — so importing a helper from here to set the caps
would initialise torch first and accomplish nothing. Entry points set
`OMP_NUM_THREADS` / `MKL_NUM_THREADS` themselves, before any project import.
"""
