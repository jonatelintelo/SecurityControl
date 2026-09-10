"""Residual-stream steering, for E1.6's causal test.

The intervention is the one the research plan specifies:

    h'_l = h_l + alpha * r^(l)          (main.tex, Experiment 2)

`alpha` is **not** a nuisance parameter to be eliminated. It carries two distinct
roles in the plan and both are required:

1. **Steering coefficient** in the cross-intervention causal matrix (PLAN-XINT).
   `main.tex` writes the repair intervention as `h_l' = h_l + alpha r_role^(l)`.

2. **A measured outcome.** The central prediction experiment (PLAN-CPE,
   `notes_from_mail.tex`) defines

       alpha* = min { |alpha| : ASR(alpha) >= tau }

   as the representation-level analogue of the sparse-attack budget `k*`, and
   makes `{C_R, N_eff,R, r_eff,R, k50_R} -> {k*, alpha*}` the paper's central
   predictive claim. Fixing alpha to a constant would make alpha* uncomputable
   and delete half of RQ5's dependent variable. `alpha_star` below computes it.

What was wrong in the earlier version was the *unit*, not the parameter. That
version applied alpha to a unit-normalised direction and re-scaled by
`mean_residual_norm`, so alpha had no natural scale and no value of it could be
justified — which is exactly why the capability window was impossible to pin
down. The literature supplies the missing unit:

    v = the RAW difference-in-means, and alpha = 1 is the published operating point

Zhao et al. (2507.11878) write exactly `h'_l = h_l + v_harmful`; Arditi et al.
(2406.11717) add the difference-in-means vector at the layer it was extracted at,
"across all token positions". Neither applies a coefficient beyond the vector's
own magnitude, so both are operating at alpha = 1. With `v` un-normalised, alpha
is denominated in **class-mean separations**: alpha = 1 shifts an item by exactly
the distance between the two class means, alpha = 0.5 by half of it. That makes
alpha = 1 a citable anchor and every other value interpretable, so the sweep that
produces alpha* is a sweep over a meaningful axis rather than over an arbitrary
one.

Three invariants are enforced here rather than left to callers, because all three
fail silently:

**Hook order.** The steering hook must be registered *before* any capture hook.
Forward hooks fire in registration order, so a capture registered first reads the
**pre**-steering value and every measured delta comes out as exactly `0.0` —
indistinguishable from "the intervention had no effect". Verified in
`tests/test_invariants.py`. `steer_and_capture` owns both registrations so the
order cannot be got wrong at a call site.

**Padding is excluded.** Steering every position of a left-padded batch corrupts
the keys and values that the final token attends over, and the damage compounds
across every subsequent layer. Measured before this was fixed: a *random*
direction drove KL(baseline || steered) to 6.59 nats and next-token entropy from
0.37 to 5.42 — a destroyed output distribution from a perturbation that should
have been inert. Fixing it brought the same measurement to KL 0.21.

**Steer with the layer's own direction.** A direction fitted at layer l lives in
layer l's basis. `steer_and_capture` takes a single layer and the caller is
responsible for pairing it with that layer's direction.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from core import model_meta


def _make_steer_hook(v_raw: torch.Tensor, alpha: float, attention_mask: torch.Tensor,
                     position_mask: Optional[torch.Tensor] = None):
    """`h' = h + alpha * v_raw`, on the selected real token positions only.

    `v_raw` is the un-normalised difference-in-means, so `alpha` is denominated in
    class-mean separations and `alpha=1.0` is the published operating point.

    Padding is always excluded: pad positions are still keys and values for the
    final token, so perturbing them corrupts what it attends over.

    `position_mask` ([batch, seq] bool) further restricts *which* real tokens are
    steered. `None` means all of them, which is Arditi et al.'s convention and the
    default. It is a swept dimension because the two sources disagree: Arditi steer
    "all token positions", while PLAN-INF describes steering "in early
    layer/tokens" — neither is established for our question, so both are run.
    """
    m = attention_mask if position_mask is None else (attention_mask.bool() & position_mask.bool())
    m = m.unsqueeze(-1)                       # [batch, seq, 1]

    def _steer(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        delta = (alpha * v_raw).to(h.dtype) * m.to(h.dtype)
        h = h + delta
        return (h,) + output[1:] if isinstance(output, tuple) else h

    return _steer


def _make_generation_steer_hook(v_raw: torch.Tensor, alpha: float,
                                attention_mask: torch.Tensor):
    """Steering hook that survives incremental decoding.

    `_make_steer_hook` closes over a `[batch, seq]` prompt mask, which is correct
    for a single forward pass and wrong during generation: after the prefill step
    the model processes one token at a time, so the hidden state is `[batch, 1, d]`
    and the prompt-shaped mask no longer aligns.

    Here the prefill step is masked to real prompt tokens (padding must never be
    perturbed — pad positions are still keys and values for the final token), and
    every subsequent decode step is steered in full, since a generated token is by
    definition a real one. That is Arditi et al.'s "all token positions"
    convention carried consistently through generation.
    """
    m = attention_mask.bool().unsqueeze(-1)
    prompt_len = attention_mask.shape[1]

    def _steer(_module, _args, output):
        h = output[0] if isinstance(output, tuple) else output
        delta = (alpha * v_raw).to(h.dtype)
        if h.shape[1] == prompt_len:                 # prefill
            h = h + delta * m.to(h.dtype)
        else:                                        # decode: all real tokens
            h = h + delta
        return (h,) + output[1:] if isinstance(output, tuple) else h

    return _steer


@torch.no_grad()
def generate_steered(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    direction,
    steer_layer: int,
    alpha: float = 1.0,
    device: str = "cuda",
    batch_size: int = 16,
    max_new_tokens: int = 48,
) -> List[str]:
    """Generate under a steering intervention, for a behavioural readout.

    The gate's behavioural condition must not rest on the single-token refusal
    logit margin: that is a proxy over two hand-chosen token ids, and the playbook
    demotes it to a cross-check that is never reported alone. The reportable
    readout is the published refusal-prefix rule applied to generated text, which
    requires actually generating under the intervention.

    `direction=None` or `alpha=0` gives the unsteered baseline through the same
    code path, so baseline and steered runs differ only in the perturbation.
    """
    layers = model_meta.find_layers(model)
    v = None if direction is None else _check_steerable(direction)
    out: List[str] = []

    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        handle = None
        if v is not None and alpha != 0.0:
            hook = _make_generation_steer_hook(v.to(device), alpha, enc["attention_mask"])
            handle = _resolve(layers, steer_layer).register_forward_hook(hook)
        try:
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
        finally:
            if handle is not None:
                handle.remove()
        for row in gen[:, enc["input_ids"].shape[1]:]:
            out.append(tokenizer.decode(row.tolist(), skip_special_tokens=True))
    return out


def _resolve(layers: nn.ModuleList, i: int) -> nn.Module:
    if not 0 <= i < len(layers):
        raise IndexError(f"layer {i} out of range for {len(layers)} layers")
    return layers[i]


def _check_steerable(direction) -> torch.Tensor:
    """Reject a direction that has no activation-space magnitude.

    A role-probe contrast is a decision boundary, not a difference-in-means; its
    `raw_norm` is NaN, so alpha would have no unit and alpha* would be
    uninterpretable. Fail loudly instead.
    """
    if isinstance(direction, torch.Tensor):
        # Deliberately refused. A bare tensor is almost always `Direction.vector`,
        # which is UNIT length: steering with it would silently apply alpha to a
        # magnitude of 1 instead of to one class-mean separation — the old scale
        # bug wearing a new hat, and invisible in the output. Callers who really
        # do mean an arbitrary vector (e.g. a subspace basis for E1.4's
        # behavioural half) must wrap it in a Direction and state `raw_norm`
        # explicitly, so the magnitude decision is always recorded.
        raise TypeError(
            "steer with a Direction, not a bare tensor: a raw tensor is usually "
            "`Direction.vector` (unit length), which silently changes what alpha "
            "means. Wrap it in a Direction with an explicit raw_norm.")
    rn = getattr(direction, "raw_norm", None)
    if rn is None:
        raise ValueError(
            f"direction {getattr(direction, 'concept', '?')} predates raw_norm; "
            "re-run the extract stage so the difference-in-means magnitude is kept")
    if rn != rn:  # NaN
        raise ValueError(
            f"direction {direction.concept!r} has no activation-space magnitude "
            "(it is a probe contrast, not a difference-in-means) and cannot be "
            "used to steer at coefficient 1.0")
    return direction.raw()


def _build_position_mask(attention_mask: torch.Tensor,
                         specs: Sequence[Sequence[bool]]) -> torch.Tensor:
    """Map per-item masks in UNPADDED coordinates onto the padded batch grid.

    `specs[i]` is one bool per *real* token of item `i`, in order. The padded
    sequence length changes from batch to batch and the models are left-padded, so
    a caller cannot pre-build a `[n_items, seq]` grid: real token `k` of an item
    does not sit at column `k`. This resolves each item's real columns from its own
    attention mask, which is correct under left *or* right padding.
    """
    out = torch.zeros_like(attention_mask, dtype=torch.bool)
    for i, spec in enumerate(specs):
        cols = torch.nonzero(attention_mask[i], as_tuple=False).flatten()
        if len(spec) != len(cols):
            raise ValueError(
                f"position mask for item {i} has {len(spec)} entries but the item "
                f"has {len(cols)} real tokens; masks must be in unpadded coordinates")
        sel = torch.tensor(spec, dtype=torch.bool, device=attention_mask.device)
        out[i, cols[sel]] = True
    return out


@torch.no_grad()
def steer_and_capture(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    direction,
    steer_layer: int,
    read_layers: Sequence[int],
    alpha: float = 1.0,
    device: str = "cuda",
    batch_size: int = 16,
    read_positions: Sequence[int] = (-1,),
    position_masks: Optional[Sequence[Sequence[bool]]] = None,
    read_rendered: Optional[Sequence] = None,
    read_which: Sequence[str] = ("t_post_inst",),
) -> Dict[int, torch.Tensor]:
    """Steer at one layer, read the residual stream at later layers.

    `alpha=0` or `direction=None` gives the unsteered baseline through the
    identical code path, so a baseline and a steered run differ only in the
    perturbation.

    Two ways to say *where* to read, and they are not interchangeable:

    * `read_positions` — flat offsets applied to every row. Safe only for
      negative offsets under left padding, where `-1` is the last real token of
      every row regardless of length.
    * `read_rendered` + `read_which` — the corpus's own **per-item** positions
      (`t_inst`, `t_post_inst`), resolved per batch through
      :func:`core.positions.batch_positions`, the same arithmetic
      `ActivationCapture.at_positions` uses. Required for any position other
      than the last token: `t_inst` sits at a different index in every row, so a
      single offset would read pad tokens for all but one of them.

    PLAN-INF asks for the intervention's effect "at later layers **and later
    tokens**", which is what the second form provides.

    Returns `{read_layer: [n_items, n_positions, d_model]}` on CPU, with the
    position axis ordered as `read_which` (or `read_positions`).
    """
    layers = model_meta.find_layers(model)
    read_layers = sorted(read_layers)
    bad = [l for l in read_layers if l <= steer_layer]
    if bad:
        raise ValueError(
            f"read layers {bad} are not downstream of steer_layer={steer_layer}; "
            "reading upstream of an intervention can only return zero deltas")

    v = None if direction is None else _check_steerable(direction)
    out: Dict[int, List[torch.Tensor]] = {l: [] for l in read_layers}

    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        handles: List[torch.utils.hooks.RemovableHandle] = []
        cache: Dict[int, torch.Tensor] = {}

        # 1. steering FIRST, and only on real tokens
        if v is not None and alpha != 0.0:
            pm = None if position_masks is None else _build_position_mask(
                enc["attention_mask"], position_masks[start:start + batch_size])
            handles.append(_resolve(layers, steer_layer).register_forward_hook(
                _make_steer_hook(v.to(device), alpha, enc["attention_mask"], pm)))

        # Resolve WHICH positions to keep before the forward pass, so the capture
        # hook can slice immediately. Retaining the full [batch, seq, d] per read
        # layer and slicing afterwards costs memory proportional to the number of
        # read layers — and E1.6 now reads every layer downstream of the steer
        # layer, which made that ~10x what it was. Slicing in the hook makes the
        # footprint independent of how many layers are read.
        n_pad = enc["input_ids"].shape[1]
        if read_rendered is not None:
            # Per-item positions, via the same arithmetic as
            # ActivationCapture.at_positions: `t_inst` sits at a different index
            # in every row, so one shared offset would read pad tokens.
            from core.positions import batch_positions
            chunk_r = list(read_rendered[start:start + batch_size])
            pos = batch_positions(chunk_r, n_pad, tokenizer.padding_side)
            sel = torch.tensor([pos[w] for w in read_which], device=device).T  # [rows, npos]
            rows_i = torch.arange(sel.shape[0], device=device).unsqueeze(1)
            take = lambda h: h[rows_i, sel, :]                      # noqa: E731
        else:
            idx = torch.tensor([p if p >= 0 else n_pad + p for p in read_positions], device=device)
            take = lambda h: h[:, idx, :]                           # noqa: E731

        # 2. capture SECOND, so it observes the steered value
        for li in read_layers:
            def _cap(_m, _a, output, li=li):
                h = output[0] if isinstance(output, tuple) else output
                cache[li] = take(h.detach()).float().cpu()
            handles.append(_resolve(layers, li).register_forward_hook(_cap))

        model(**enc)
        for h in handles:
            h.remove()

        for li in read_layers:
            out[li].append(cache[li])
        cache.clear()

    return {li: torch.cat(v_, 0) for li, v_ in out.items()}


@torch.no_grad()
def next_token_stats(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    direction,
    steer_layer: int,
    alpha: float = 1.0,
    device: str = "cuda",
    batch_size: int = 16,
    refusal_token: str = "I",
    comply_token: str = "Sure",
    position_masks: Optional[Sequence[Sequence[bool]]] = None,
) -> Dict[str, torch.Tensor]:
    """Behavioural readout at `t_post-inst` under the intervention.

    Reading behaviour at the final prompt token is Zhao et al.'s stated
    convention: "accepting or refusing examples refer to model behaviors at the
    t_post-inst position using the default prompting template."

    Returns the refusal logit margin, next-token entropy in **absolute nats**, and
    the log-probabilities so the caller can compute KL against the baseline.

    Entropy *ratio* is the wrong capability guard: an instruct model at the
    assistant turn is highly confident, so baseline entropy is near zero and any
    perturbation looks like a huge multiple while remaining small in absolute
    terms. Judge on absolute entropy and on KL from baseline instead.
    """
    layers = model_meta.find_layers(model)
    r_id = tokenizer.encode(refusal_token, add_special_tokens=False)[0]
    c_id = tokenizer.encode(comply_token, add_special_tokens=False)[0]
    v = None if direction is None else _check_steerable(direction)

    margins, entropies, logprobs = [], [], []
    for start in range(0, len(texts), batch_size):
        chunk = list(texts[start:start + batch_size])
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(device)
        handles = []
        if v is not None and alpha != 0.0:
            pm = None if position_masks is None else _build_position_mask(
                enc["attention_mask"], position_masks[start:start + batch_size])
            handles.append(_resolve(layers, steer_layer).register_forward_hook(
                _make_steer_hook(v.to(device), alpha, enc["attention_mask"], pm)))

        logits = model(**enc).logits[:, -1, :].float()
        for h in handles:
            h.remove()
        margins.append((logits[:, r_id] - logits[:, c_id]).cpu())
        lp = torch.log_softmax(logits, dim=-1)
        p = lp.exp()
        entropies.append((-(p * lp).sum(-1)).cpu())
        logprobs.append(lp.cpu())

    return {"refusal_margin": torch.cat(margins), "entropy": torch.cat(entropies),
            "logprobs": torch.cat(logprobs)}


def kl_from_baseline(baseline_logprobs: torch.Tensor, steered_logprobs: torch.Tensor) -> torch.Tensor:
    """KL(baseline || steered) per item, in nats.

    Arditi et al.'s `kl_score` is this quantity, computed on **harmless** prompts,
    and used to reject directions whose intervention "minimally disrupts" ordinary
    behaviour. We arrived at the same guard independently; using theirs makes the
    selection criterion citable rather than ad hoc.
    """
    p = baseline_logprobs.exp()
    return (p * (baseline_logprobs - steered_logprobs)).sum(-1)


@torch.no_grad()
def selection_scores(
    model: nn.Module,
    tokenizer,
    harmful_texts: Sequence[str],
    harmless_texts: Sequence[str],
    direction,
    steer_layer: int,
    alpha: float = 1.0,
    device: str = "cuda",
    batch_size: int = 16,
    baseline: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
) -> Dict[str, float]:
    """Arditi et al.'s three direction-selection metrics, at one layer.

    Their procedure sweeps every layer, scores each candidate direction on these
    three, and keeps the direction that bypasses refusal on harmful prompts,
    induces it on harmless prompts, and leaves harmless behaviour otherwise
    intact. Computing them per layer is what replaces our earlier hand-picked
    steering depth: the layer is *selected on evidence*, by a published criterion.

    * `bypass_score`  — drop in refusal margin on **harmful** prompts at `-alpha`
      (steering away from the positive class should suppress refusal).
    * `induce_score`  — rise in refusal margin on **harmless** prompts at `+alpha`.
    * `kl_score`      — KL(baseline || steered) on **harmless** prompts; small is
      good. This is a cost, not a benefit: it gates the other two.

    `baseline` may be passed in to avoid recomputing the unsteered pass.
    """
    if baseline is None:
        baseline = {
            "harmful": next_token_stats(model, tokenizer, harmful_texts, None,
                                        steer_layer, 0.0, device, batch_size),
            "harmless": next_token_stats(model, tokenizer, harmless_texts, None,
                                         steer_layer, 0.0, device, batch_size),
        }

    neg = next_token_stats(model, tokenizer, harmful_texts, direction,
                           steer_layer, -alpha, device, batch_size)
    pos = next_token_stats(model, tokenizer, harmless_texts, direction,
                           steer_layer, +alpha, device, batch_size)

    bypass = float((baseline["harmful"]["refusal_margin"] - neg["refusal_margin"]).mean())
    induce = float((pos["refusal_margin"] - baseline["harmless"]["refusal_margin"]).mean())
    kl = float(kl_from_baseline(baseline["harmless"]["logprobs"], pos["logprobs"]).mean())
    return {"bypass_score": bypass, "induce_score": induce, "kl_score": kl,
            "entropy_harmless_steered": float(pos["entropy"].mean()),
            "entropy_harmless_baseline": float(baseline["harmless"]["entropy"].mean())}


@torch.no_grad()
def steering_sanity(
    model: nn.Module,
    tokenizer,
    texts: Sequence[str],
    direction,
    steer_layer: int,
    device: str = "cuda",
    batch_size: int = 16,
    alphas: Sequence[float] = (0.5, 1.0, 2.0),
    n_random: int = 3,
    seed: int = 0,
) -> Dict[str, object]:
    """Precondition on the steering primitive. Run before trusting any effect.

    Two checks, both of which failed silently in an earlier implementation:

    * **determinism** — the unsteered forward pass run twice must give KL = 0.
      Anything else means the measurement itself is noisy.
    * **magnitude-matched random control** — a random direction of the *same
      magnitude* as the real one carries no class information, so the gap between
      its effect and the real direction's is the part attributable to content
      rather than to the size of the push. If a magnitude-matched random vector
      moves the model as much as the real direction does, the "causal effect" is
      measuring perturbation, not mechanism.

    Unlike the earlier `alpha` curve, this does not search for a usable strength:
    coefficient 1.0 is the published operating point and the others are reported
    only as sensitivity.
    """
    from core import extract as _extract

    base = next_token_stats(model, tokenizer, texts, None, steer_layer, 0.0, device, batch_size)
    again = next_token_stats(model, tokenizer, texts, None, steer_layer, 0.0, device, batch_size)
    determinism_kl = float(kl_from_baseline(base["logprobs"], again["logprobs"]).mean())

    curve = {}
    for c in alphas:
        real = next_token_stats(model, tokenizer, texts, direction, steer_layer,
                                c, device, batch_size)
        real_kl = float(kl_from_baseline(base["logprobs"], real["logprobs"]).mean())
        rnd = []
        for s in range(n_random):
            rv = _extract.random_direction_like(direction, seed=seed + s)
            st = next_token_stats(model, tokenizer, texts, rv, steer_layer,
                                  c, device, batch_size)
            rnd.append(float(kl_from_baseline(base["logprobs"], st["logprobs"]).mean()))
        curve[c] = {"real_kl": real_kl,
                    "random_kl_mean": sum(rnd) / len(rnd),
                    "random_kl_max": max(rnd),
                    "content_gap": real_kl - sum(rnd) / len(rnd)}

    return {"baseline_entropy": float(base["entropy"].mean()),
            "determinism_kl": determinism_kl,
            "alpha_curve": curve,
            "direction_raw_norm": float(getattr(direction, "raw_norm", float("nan"))),
            "deterministic": determinism_kl < 1e-4}


def alpha_star(
    alphas: Sequence[float],
    asr: Sequence[float],
    tau: float,
) -> Optional[float]:
    """`alpha* = min { |alpha| : ASR(alpha) >= tau }` — PLAN-CPE, verbatim.

    The representation-level analogue of the sparse-attack budget `k*`, and one of
    the two quantities the paper's central prediction must produce:

        {C_R, N_eff,R, r_eff,R, k50_R}  ->  {k*, alpha*}

    Returns `None` when no swept `alpha` reaches `tau`. That is a real answer —
    "this variable could not be pushed to threshold within the swept range" — and
    must be reported as right-censored rather than silently dropped or replaced by
    the largest alpha tried, either of which would bias the architecture-to-
    vulnerability relationship toward whatever the sweep happened to cover.

    `tau` is **undefined in the plan**, so callers sweep it too and report
    `alpha*(tau)` as a curve rather than committing to one threshold.
    """
    hits = [abs(a) for a, s in zip(alphas, asr) if s >= tau]
    return min(hits) if hits else None
