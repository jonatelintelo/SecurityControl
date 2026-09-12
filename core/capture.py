"""Activation capture at explicit token positions.

Written against the current position API (`Rendered`, `batch_positions`). Zhao
read harmfulness at `t_inst` and refusal at `t_post-inst`, so a last-token-only
capture cannot express the design at all.

Two read sites:

* ``residual`` — the decoder block's output. **Primary for every direction**, so
  all three variables share one space: the attribution formalism `P_R f_{i,l}(h)`
  is only defined for a residual-stream subspace, and principal angles between the
  variables need a common basis.
* ``pre_mlp`` — the output of ``post_attention_layernorm``. Where the role paper's
  demo hooks, used only for the fidelity check that we reproduce their probe.

Hook-ordering note: capture hooks here are read-only. When steering is added
(E1.6) the steering hook **must be registered first** — forward hooks fire in
registration order, so a capture registered first observes the pre-steering value
and every delta comes out as exactly zero. Verified in `tests/test_invariants.py`.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from core import model_meta
from core.positions import Rendered, batch_positions

SITES = ("residual", "pre_mlp")


def has_site(layer: nn.Module, site: str) -> bool:
    """Can this architecture expose `site` at all?

    Not every decoder block has a `post_attention_layernorm`. NVIDIA's
    Nemotron-H is a Mamba/attention hybrid whose layer pattern is
    `MEMEM*EMEM...` — most layers are Mamba or MLP blocks, and every block
    exposes only `norm` and `mixer`. There is no post-attention layernorm
    because most layers have no attention.

    Callers use this to record "not applicable on this architecture" instead of
    dying. The distinction matters: a reproduction check that CANNOT be run is a
    scope limit to report, while one that runs and disagrees is a finding.
    """
    try:
        _site_module(layer, site)
        return True
    except AttributeError:
        return False


def _site_module(layer: nn.Module, site: str) -> nn.Module:
    if site == "residual":
        return layer
    if site == "pre_mlp":
        mod = getattr(layer, "post_attention_layernorm", None)
        if mod is None:
            raise AttributeError(
                f"{type(layer).__name__} has no post_attention_layernorm; cannot "
                "reproduce the role paper's read site")
        return mod
    raise ValueError(f"site must be one of {SITES}, got {site!r}")


class ActivationCapture:
    """Forward-hook capture over chosen layers at one read site."""

    def __init__(self, model: nn.Module, site: str = "residual"):
        self.model = model
        self.site = site
        self.layers = model_meta.find_layers(model)
        self.num_layers = len(self.layers)
        self.d_model = model_meta.hidden_size(model.config)
        self._cache: Dict[int, torch.Tensor] = {}
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    def _register(self, layers: Sequence[int]) -> None:
        for li in layers:
            def _hook(_m, _a, output, li=li):
                h = output[0] if isinstance(output, tuple) else output
                self._cache[li] = h.detach()
            self._handles.append(
                _site_module(self.layers[li], self.site).register_forward_hook(_hook))

    def _clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._cache.clear()

    @torch.no_grad()
    def at_positions(
        self,
        tokenizer,
        rendered: Sequence[Rendered],
        which: Sequence[str] = ("t_inst", "t_post_inst"),
        layers: Optional[Sequence[int]] = None,
        batch_size: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        log_every: int = 0,
        logger=None,
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        """One activation per example per requested position.

        Returns `{position_name: {layer: [n_examples, d_model]}}` on CPU. Indices
        are recomputed per batch from each example's own token count: the padded
        offset differs per row, and applying one offset to a whole batch reads pad
        tokens and yields silently meaningless activations.

        Both positions come from a **single forward pass** — capturing them in
        separate passes would double the cost for no benefit.
        """
        for w in which:
            if w not in ("t_inst", "t_post_inst"):
                raise ValueError(f"unknown position {w!r}")
        layers = list(layers) if layers is not None else list(range(self.num_layers))
        out: Dict[str, Dict[int, List[torch.Tensor]]] = {
            w: {li: [] for li in layers} for w in which}

        for start in range(0, len(rendered), batch_size):
            chunk = list(rendered[start:start + batch_size])
            enc = tokenizer([r.text for r in chunk], return_tensors="pt",
                            padding=True, add_special_tokens=False).to(device)
            padded = enc["input_ids"].shape[1]
            pos = batch_positions(chunk, padded, tokenizer.padding_side)
            rows = torch.arange(len(chunk), device=device)

            self._clear()
            self._register(layers)
            self.model(**enc)
            for w in which:
                idx = torch.tensor(pos[w], device=device)
                for li in layers:
                    out[w][li].append(self._cache[li][rows, idx, :].to(dtype).cpu())
            self._clear()

            if log_every and logger and (start // batch_size) % log_every == 0:
                logger.info(f"    captured {min(start + batch_size, len(rendered))}/{len(rendered)}")

        return {w: {li: torch.cat(v, 0) for li, v in d.items()} for w, d in out.items()}

    @torch.no_grad()
    def at_content_tokens(
        self,
        tokenizer,
        rendered: Sequence[Rendered],
        layers: Optional[Sequence[int]] = None,
        max_tokens_per_seq: int = 8,
        batch_size: int = 16,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        seed: int = 0,
        log_every: int = 0,
        logger=None,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Token-level capture over content spans, for the role probe.

        Spans exclude the role tags, so the probe cannot win by memorising tag
        tokens — which would make it a tag detector rather than a role probe.

        Returns `({layer: [n_tokens, d]}, owner[n_tokens])`, where `owner` maps
        each token back to its example so a split by instruction carries over.

        **Memory.** This is the dominant path: every content token x every layer.
        `max_tokens_per_seq` bounds it by sampling deterministically; at 8 tokens a
        full-layer run over 3,200 items is ~3.5 GB in fp16.
        """
        layers = list(layers) if layers is not None else list(range(self.num_layers))
        out: Dict[int, List[torch.Tensor]] = {li: [] for li in layers}
        owner: List[torch.Tensor] = []
        gen = torch.Generator().manual_seed(seed)

        for start in range(0, len(rendered), batch_size):
            chunk = list(rendered[start:start + batch_size])
            enc = tokenizer([r.text for r in chunk], return_tensors="pt",
                            padding=True, add_special_tokens=False).to(device)
            padded = enc["input_ids"].shape[1]
            pos = batch_positions(chunk, padded, tokenizer.padding_side)

            rows, cols, owners = [], [], []
            for i in range(len(chunk)):
                span = list(range(pos["content_start"][i], min(pos["content_end"][i], padded)))
                if not span:
                    continue
                if len(span) > max_tokens_per_seq:
                    pick = torch.randperm(len(span), generator=gen)[:max_tokens_per_seq].tolist()
                    span = [span[j] for j in sorted(pick)]
                rows += [i] * len(span)
                cols += span
                owners += [start + i] * len(span)

            if not rows:
                continue
            r = torch.tensor(rows, device=device)
            c = torch.tensor(cols, device=device)

            self._clear()
            self._register(layers)
            self.model(**enc)
            for li in layers:
                out[li].append(self._cache[li][r, c, :].to(dtype).cpu())
            self._clear()
            owner.append(torch.tensor(owners))

            if log_every and logger and (start // batch_size) % log_every == 0:
                logger.info(f"    tokens from {min(start + batch_size, len(rendered))}/{len(rendered)}")

        if not owner:
            raise ValueError("no content tokens captured; check span computation")
        return {li: torch.cat(v, 0) for li, v in out.items()}, torch.cat(owner)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._clear()
        return False
