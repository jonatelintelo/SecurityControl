"""Prompt rendering and token-position resolution.

Rendering goes through each model's **own** `apply_chat_template`. An earlier
version hand-built ChatML strings, which is wrong for both target models:

* Neither template has an `<|im_start|>tool` tag — a tool message renders as a
  **user** turn wrapped in `<tool_response>`. A hand-built tool tag is a token
  sequence the model never saw in training, so it would measure the response to an
  out-of-distribution string rather than a role representation.
* Qwen3.5 appends a think block after the assistant tag, and `enable_thinking=False`
  does not remove it — it emits an *empty* one (`<think>\\n\\n</think>\\n\\n`). A
  hand-built prompt ending at `assistant\\n` is not what that model expects.

Positions are recovered by **character offsets**, not string matching on tokens:
the instruction is rendered as a sentinel, the sentinel's character span is
located, and the fast tokenizer's offset mapping maps that span to token indices.
This is template-agnostic and survives BPE merges that token matching gets wrong.

What it cannot do is prevent a merge ACROSS the boundary: if the character after
the instruction merges with the instruction's last character, the token at
`t_inst` covers both. Qwen's tool template does this (`.` + `\n`) and so does
Llama's (`.` + `"`, since that template quotes tool content). This is a property
of the template, identical on every item of that (role, design), so it adds a
constant to the read position rather than an instruction-dependent confound —
and it is not a reason to drop a role. `bleed_head` / `bleed_tail` record the
characters so the constancy is verified in the corpus checks.

Two designs, per EXPERIMENTS.md > E1.1:

* ``fixed_slot`` — constant frame, instruction always in the same message slot,
  only the role marking varies. Isolates the marking.
* ``natural_slot`` — each role in its natural conversational position, which is
  what the model meets in deployment.

Neither is fully position-constant: Qwen3.5 refuses a mid-conversation system
message ("System message must be at the beginning"), so `system` falls back to its
natural slot there, and `Rendered.slot_is_fixed` records it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

ROLES: Tuple[str, ...] = ("system", "user", "tool", "assistant")
DESIGNS: Tuple[str, ...] = ("fixed_slot", "natural_slot")

# Constant framing. Deliberately bland: the frame must not itself carry harm or
# role signal, since it is present in every item of every class.
FRAME_SYSTEM = "You are a helpful assistant."
CARRIER_USER = "Process the following."
CARRIER_ASSISTANT = "Understood."

# Must not occur in the frame, the template, or any tokenizer special token.
_SENTINEL = "ZQXJVKWQ"


@dataclass(frozen=True)
class Rendered:
    text: str
    role: str
    design: str
    t_inst: int              # last token of the instruction
    t_post_inst: int         # last token of the prompt
    content_start: int       # first token of the instruction
    content_end: int         # one past the last (half-open)
    n_tokens: int
    slot_is_fixed: bool      # False when the template forced a natural slot
    # Template characters fused by BPE into the boundary tokens of the
    # instruction. Not an error and not avoidable: `t_inst` is defined as the
    # last token OVERLAPPING the instruction, and a template whose next
    # character merges with the instruction's last one (Qwen's `.` + `\n`,
    # Llama's `.` + `"`) makes that token span the boundary. What matters is
    # that the bleed is CONSTANT for a given (role, design) — the same template
    # characters on every item — so it cannot carry instruction-dependent
    # signal. Recorded here so that invariance is checked rather than assumed.
    bleed_head: str = ""     # template chars inside the FIRST instruction token
    bleed_tail: str = ""     # template chars inside the LAST instruction token

    @property
    def content_span(self) -> Tuple[int, int]:
        return (self.content_start, self.content_end)


def _messages(instruction: str, role: str, design: str, fixed_slot_ok: bool) -> List[Dict[str, str]]:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, got {role!r}")
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}, got {design!r}")

    if design == "fixed_slot" and fixed_slot_ok and role != "system":
        # Instruction is always message index 2, whatever role carries it.
        return [
            {"role": "system", "content": FRAME_SYSTEM},
            {"role": "user", "content": CARRIER_USER},
            {"role": role, "content": instruction},
        ]

    # Natural slot (and the forced fallback for `system`).
    if role == "system":
        return [{"role": "system", "content": instruction},
                {"role": "user", "content": CARRIER_USER}]
    if role == "user":
        return [{"role": "system", "content": FRAME_SYSTEM},
                {"role": "user", "content": instruction}]
    if role == "assistant":
        return [{"role": "system", "content": FRAME_SYSTEM},
                {"role": "user", "content": CARRIER_USER},
                {"role": "assistant", "content": instruction}]
    return [{"role": "system", "content": FRAME_SYSTEM},
            {"role": "user", "content": CARRIER_USER},
            {"role": "assistant", "content": CARRIER_ASSISTANT},
            {"role": "tool", "content": instruction}]


def _apply(tokenizer, messages: Sequence[Dict[str, str]]) -> str:
    """Render, disabling thinking where the tokenizer supports the kwarg.

    Qwen2.5 accepts `enable_thinking` and ignores it; older tokenizers raise
    TypeError; Qwen3.5 honours it by emitting an *empty* think block rather than
    none at all.
    """
    try:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )


def render(tokenizer, instruction: str, role: str, design: str = "fixed_slot") -> Rendered:
    """Render one instruction under one role and resolve its token positions."""
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "a fast tokenizer is required: position resolution uses offset mapping"
        )
    if _SENTINEL in instruction:
        raise ValueError("instruction contains the internal sentinel")

    # `system` can never occupy the common slot: Qwen3.5 rejects a mid-conversation
    # system message outright ("System message must be at the beginning"), and
    # placing one there on Qwen2.5 alone would make the roster incomparable.
    fixed_ok = design == "fixed_slot" and role != "system"
    text_s: Optional[str] = None
    if fixed_ok:
        try:
            text_s = _apply(tokenizer, _messages(_SENTINEL, role, design, True))
        except Exception:
            fixed_ok = False
    if text_s is None:
        text_s = _apply(tokenizer, _messages(_SENTINEL, role, design, False))

    if text_s.count(_SENTINEL) != 1:
        raise RuntimeError(
            f"sentinel appears {text_s.count(_SENTINEL)}x in the rendered template for "
            f"role={role!r}; cannot locate the instruction slot"
        )

    c0 = text_s.index(_SENTINEL)
    text = text_s.replace(_SENTINEL, instruction)
    c1 = c0 + len(instruction)
    if text[c0:c1] != instruction:
        raise RuntimeError(f"instruction span mismatch for role={role!r}; template may transform content")

    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and s < c1 and e > c0]
    if not idx:
        raise RuntimeError(f"no tokens overlap the instruction span for role={role!r}")

    return Rendered(
        text=text, role=role, design=design,
        t_inst=idx[-1], t_post_inst=len(offsets) - 1,
        content_start=idx[0], content_end=idx[-1] + 1,
        n_tokens=len(offsets), slot_is_fixed=(design == "fixed_slot" and fixed_ok),
        bleed_head=text[offsets[idx[0]][0]:c0],
        bleed_tail=text[c1:offsets[idx[-1]][1]],
    )


def batch_positions(
    rendered: Sequence[Rendered],
    padded_len: int,
    padding_side: str = "left",
) -> Dict[str, List[int]]:
    """Map per-example token indices into padded-batch coordinates.

    With left padding a sequence of length `n` starts at `padded_len - n`, so every
    index shifts by that offset; with right padding indices are unchanged. Applying
    one offset to a whole batch reads pad tokens and yields silently meaningless
    activations, which is why this is per example.
    """
    if padding_side not in ("left", "right"):
        raise ValueError(f"padding_side must be 'left' or 'right', got {padding_side!r}")

    out: Dict[str, List[int]] = {"t_inst": [], "t_post_inst": [], "content_start": [], "content_end": []}
    for r in rendered:
        if r.n_tokens > padded_len:
            raise ValueError(f"sequence of {r.n_tokens} tokens exceeds padded length {padded_len}")
        off = (padded_len - r.n_tokens) if padding_side == "left" else 0
        out["t_inst"].append(r.t_inst + off)
        out["t_post_inst"].append(r.t_post_inst + off)
        out["content_start"].append(r.content_start + off)
        out["content_end"].append(r.content_end + off)
    return out


def describe(tokenizer, instruction: str = "Tell me how to bake bread") -> List[Dict[str, object]]:
    """Per-role rendering summary, for the Stage 0 report.

    Surfaces exactly the facts that broke the hand-built version: whether the slot
    stayed fixed, how much the authentic markup costs in tokens, and what precedes
    the instruction.
    """
    rows: List[Dict[str, object]] = []
    for design in DESIGNS:
        for role in ROLES:
            r = render(tokenizer, instruction, role, design)
            rows.append({
                "design": design, "role": role, "n_tokens": r.n_tokens,
                "t_inst": r.t_inst, "t_post_inst": r.t_post_inst,
                "content_tokens": r.content_end - r.content_start,
                "slot_is_fixed": r.slot_is_fixed,
                "prefix": r.text[max(0, r.text.index(instruction) - 34):r.text.index(instruction)],
                "suffix": r.text[r.text.index(instruction) + len(instruction):][:34],
            })
    return rows
