"""Prompt rendering and token-position resolution.

Two things everything downstream depends on:

1. **Uniform rendering across roles.** The crossed corpus puts the *same*
   instruction under `system` / `user` / `tool` / `assistant` tags. If the four
   were rendered by different code paths — say `apply_chat_template` for the
   roles it supports and a hand-built string for `tool` — then role would be
   confounded with template construction, and `R_role` would partly encode "which
   branch built this string". So all roles go through one ChatML formatter.

2. **Per-example token positions.** `t_inst` is *not* index -1. Zhao et al. read
   harmfulness at `t_inst` (last content token of the instruction) and refusal at
   `t_post-inst` (last token of the prompt); with batched padding these differ per
   row. `resolve_positions` returns explicit indices, and `batch_positions`
   converts them to padded-batch coordinates.

Positions are computed by *prefix length*, not by matching token strings:
tokenising the text up to and including the instruction gives `t_inst` directly.
This is robust to BPE merges that string matching would get wrong.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# ChatML, used by both Qwen2.5 and Qwen3.5. Kept explicit rather than pulled from
# the tokenizer's jinja template so that every role is built identically.
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

ROLES = ("system", "user", "tool", "assistant")


def _turn(role: str, content: str) -> str:
    return f"{IM_START}{role}\n{content}{IM_END}\n"


def render_prompt(
    instruction: str,
    role: str = "user",
    system: Optional[str] = None,
) -> Tuple[str, str]:
    """Render `instruction` inside a `role` turn, plus a trailing generation prompt.

    Returns `(prefix_through_instruction, full_text)`. The split point is what
    makes `t_inst` computable without token matching: `prefix_through_instruction`
    ends exactly at the last character of the instruction.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")

    head = _turn("system", system) if system is not None else ""
    prefix = f"{head}{IM_START}{role}\n{instruction}"
    full = f"{prefix}{IM_END}\n{IM_START}assistant\n"
    return prefix, full


@dataclass(frozen=True)
class Positions:
    """Token indices into an *unpadded* tokenisation of `text`."""
    text: str
    t_inst: int       # last content token of the instruction-bearing turn
    t_post_inst: int  # last token of the prompt (the newline after `assistant`)
    n_tokens: int


def resolve_positions(
    tokenizer,
    instruction: str,
    role: str = "user",
    system: Optional[str] = None,
) -> Positions:
    prefix, full = render_prompt(instruction, role=role, system=system)
    n_prefix = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    n_full = len(tokenizer(full, add_special_tokens=False)["input_ids"])
    if n_prefix == 0 or n_full <= n_prefix:
        raise ValueError(f"degenerate tokenisation: prefix={n_prefix} full={n_full}")
    return Positions(text=full, t_inst=n_prefix - 1, t_post_inst=n_full - 1, n_tokens=n_full)


def batch_positions(
    positions: Sequence[Positions],
    padded_len: int,
    padding_side: str = "left",
) -> Tuple[List[int], List[int]]:
    """Map per-example indices into padded-batch coordinates.

    With left padding a sequence of length `n` starts at `padded_len - n`, so every
    index shifts by that offset; with right padding indices are unchanged. Getting
    this wrong reads a pad token and yields a silently meaningless activation.
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")

    t_inst, t_post = [], []
    for p in positions:
        if p.n_tokens > padded_len:
            raise ValueError(f"sequence of {p.n_tokens} tokens exceeds padded length {padded_len}")
        offset = (padded_len - p.n_tokens) if padding_side == "left" else 0
        t_inst.append(p.t_inst + offset)
        t_post.append(p.t_post_inst + offset)
    return t_inst, t_post


def content_token_spans(
    tokenizer,
    instruction: str,
    role: str = "user",
    system: Optional[str] = None,
) -> Tuple[str, int, int]:
    """Half-open `[start, end)` token span covering only the instruction content.

    The role probe is trained on content tokens with the role tags filtered out
    (their demo does the same), so that the probe cannot simply memorise the tag
    tokens — which would make it a tag detector rather than a role probe.
    """
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    head = _turn("system", system) if system is not None else ""
    before = f"{head}{IM_START}{role}\n"
    prefix, full = render_prompt(instruction, role=role, system=system)
    start = len(tokenizer(before, add_special_tokens=False)["input_ids"])
    end = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    if end <= start:
        raise ValueError(f"empty content span for role={role!r}: [{start}, {end})")
    return full, start, end
