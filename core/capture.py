"""Activation capture at explicit token positions.

Replaces the previous `HookEngine`, which only ever read index `-1`. Zhao et al.
read harmfulness at `t_inst` and refusal at `t_post-inst`, so a last-token-only
capture cannot express the design at all.

Two read sites:

* ``residual`` — the decoder block's output (post-addition). Primary for every
  direction, so all three variables live in one space and the E1.2 geometry is
  interpretable.
* ``pre_mlp`` — the output of ``post_attention_layernorm``, i.e. the MLP's input.
  This is where the role paper's demo hooks, so it is used for the fidelity check
  that we reproduce their probe.

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
from core.positions import Positions, batch_positions

SITES = ("residual", "pre_mlp")


def _site_module(layer: nn.Module, site: str) -> nn.Module:
    if site == "residual":
        return layer
    if site == "pre_mlp":
        mod = getattr(layer, "post_attention_layernorm", None)
        if mod is None:
            raise AttributeError(
                f"layer {type(layer).__name__} has no post_attention_layernorm; "
                "cannot reproduce the role paper's read site"
            )
        return mod
    raise ValueError(f"site must be one of {SITES}, got {site!r}")


class ActivationCapture:
    """Forward-hook capture over a chosen set of layers and one read site."""

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
            def _hook(_m, _args, output, li=li):
                h = output[0] if isinstance(output, tuple) else output
                self._cache[li] = h.detach()
            self._handles.append(_site_module(self.layers[li], self.site).register_forward_hook(_hook))

    def _clear(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._cache.clear()

    @torch.no_grad()
    def at_positions(
        self,
        tokenizer,
        positions: Sequence[Positions],
        which: str = "t_inst",
        layers: Optional[Sequence[int]] = None,
        batch_size: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> Dict[int, torch.Tensor]:
        """One activation per example, at `t_inst` or `t_post_inst`.

        Returns `{layer: [n_examples, d_model]}` on CPU. Indices are recomputed
        per batch from each example's own token count — the padded offset differs
        per row, and applying one offset to the batch reads pad tokens.
        """
        if which not in ("t_inst", "t_post_inst"):
            raise ValueError(f"which must be 't_inst' or 't_post_inst', got {which!r}")
        layers = list(layers) if layers is not None else list(range(self.num_layers))
        out: Dict[int, List[torch.Tensor]] = {li: [] for li in layers}

        for start in range(0, len(positions), batch_size):
            chunk = list(positions[start:start + batch_size])
            enc = tokenizer([p.text for p in chunk], return_tensors="pt",
                            padding=True, add_special_tokens=False).to(device)
            padded_len = enc["input_ids"].shape[1]
            t_inst, t_post = batch_positions(chunk, padded_len, tokenizer.padding_side)
            idx = torch.tensor(t_inst if which == "t_inst" else t_post, device=device)

            self._clear()
            self._register(layers)
            self.model(**enc)
            rows = torch.arange(len(chunk), device=device)
            for li in layers:
                out[li].append(self._cache[li][rows, idx, :].to(dtype).cpu())
            self._clear()

        return {li: torch.cat(v, dim=0) for li, v in out.items()}

    @torch.no_grad()
    def at_content_tokens(
        self,
        tokenizer,
        spans: Sequence[Tuple[str, int, int]],
        layers: Optional[Sequence[int]] = None,
        max_tokens_per_seq: int = 8,
        batch_size: int = 8,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        seed: int = 0,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Token-level capture over content spans, for the role probe.

        `spans` are `(text, start, end)` in unpadded coordinates, with role tags
        already excluded — the probe must not be able to win by memorising tag
        tokens, which would make it a tag detector rather than a role probe.

        Returns `({layer: [n_tokens, d_model]}, example_index[n_tokens])`. The
        second value maps each captured token back to its source example so a
        train/test split by instruction can be applied to tokens.

        **Memory.** Token-level capture across all layers is the expensive path:
        1,600 sequences x 32 layers x d=4096 in fp16 costs ~0.4 GB per token kept
        per sequence. `max_tokens_per_seq` bounds it by sampling content tokens
        deterministically; the default of 8 keeps a full-layer run near 3.5 GB.
        """
        layers = list(layers) if layers is not None else list(range(self.num_layers))
        out: Dict[int, List[torch.Tensor]] = {li: [] for li in layers}
        owner: List[torch.Tensor] = []
        gen = torch.Generator().manual_seed(seed)

        for start in range(0, len(spans), batch_size):
            chunk = list(spans[start:start + batch_size])
            enc = tokenizer([t for t, _, _ in chunk], return_tensors="pt",
                            padding=True, add_special_tokens=False).to(device)
            padded_len = enc["input_ids"].shape[1]
            lengths = enc["attention_mask"].sum(dim=1).tolist()

            sel_rows, sel_cols, sel_owner = [], [], []
            for i, ((_, s, e), n_tok) in enumerate(zip(chunk, lengths)):
                offset = (padded_len - n_tok) if tokenizer.padding_side == "left" else 0
                cols = list(range(s + offset, min(e + offset, padded_len)))
                if not cols:
                    continue
                if len(cols) > max_tokens_per_seq:
                    pick = torch.randperm(len(cols), generator=gen)[:max_tokens_per_seq].tolist()
                    cols = [cols[j] for j in sorted(pick)]
                sel_rows += [i] * len(cols)
                sel_cols += cols
                sel_owner += [start + i] * len(cols)

            if not sel_rows:
                continue
            r = torch.tensor(sel_rows, device=device)
            c = torch.tensor(sel_cols, device=device)

            self._clear()
            self._register(layers)
            self.model(**enc)
            for li in layers:
                out[li].append(self._cache[li][r, c, :].to(dtype).cpu())
            self._clear()
            owner.append(torch.tensor(sel_owner))

        if not owner:
            raise ValueError("no content tokens captured; check span computation")
        return ({li: torch.cat(v, dim=0) for li, v in out.items()}, torch.cat(owner))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._clear()
        return False
