"""Preflight: every roster model expresses every role class it claims.

WHY THIS EXISTS
Three failure modes are possible when a role class meets a chat template, and
two of them are SILENT:

1. The template raises. Loud, and `render` propagates it.
2. The template has no branch for the role and drops the message entirely.
   `render` catches this (the sentinel appears 0x) — but only because the
   sentinel is checked; nothing else would notice that the prompt no longer
   contains the instruction. Qwen2.5 and gpt-oss-20b both do this for a `cot`
   role.
3. The template interpolates the role name into a tag it has never been
   trained on — Llama emits `<|start_header_id|>cot<|end_header_id|>`, Nemotron
   emits `<|im_start|>cot`. NOTHING catches this. The run completes, the
   numbers look plausible, and what was measured is the model's response to an
   out-of-distribution string rather than a role representation.

(3) is why `ROLES` is a closed set and why a new role class may not be added
without per-model evidence that the model was trained on that tag. `cot` is the
concrete case: the role paper (2603.12277) uses it, but on every model here it
is either dropped or rendered as a novel tag, because chain-of-thought is not a
chat-template role — it is a channel inside an assistant turn.

Run: python tests/smoke_roles.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_HOME", "/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir")

from core.config import MODELS, RQ1_MODELS, load_config, roster_table
from core.positions import DESIGNS, ROLES, render

PROBE = "Explain how photosynthesis converts sunlight into chemical energy."
# A role tag may absorb at most this many template characters through a BPE
# merge at the instruction boundary. One character is what the observed cases
# need (Qwen `\n`, Llama `"`); a larger overhang means the span resolution is
# wrong, not that the template is quirky.
MAX_BLEED_CHARS = 4

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    if not ok:
        FAILURES.append(name)
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


def main() -> int:
    from transformers import AutoTokenizer

    cfg = load_config()
    print("RQ1 roster under test:")
    print(roster_table())
    print()

    for slug in RQ1_MODELS:
        spec = MODELS[slug]
        roles = cfg.roles_for(slug)
        try:
            tok = AutoTokenizer.from_pretrained(spec.model_id)
        except Exception as e:
            check(f"{slug}: tokenizer loads", False, f"{type(e).__name__}: {str(e)[:120]}")
            continue

        renderings: dict[tuple, str] = {}
        problems, bleeds = [], {}
        for role in roles:
            for design in DESIGNS:
                try:
                    R = render(tok, PROBE, role, design)
                except Exception as e:
                    problems.append(f"{role}/{design}: {type(e).__name__}: {str(e)[:70]}")
                    continue
                # The instruction must be present, verbatim, exactly once.
                if R.text.count(PROBE) != 1:
                    problems.append(f"{role}/{design}: instruction appears "
                                    f"{R.text.count(PROBE)}x (dropped or duplicated)")
                    continue
                enc = tok(R.text, add_special_tokens=False, return_offsets_mapping=True)
                offs = enc["offset_mapping"]
                c0 = R.text.index(PROBE)
                c1 = c0 + len(PROBE)
                if not (offs[R.t_inst][0] < c1 <= offs[R.t_inst][1]):
                    problems.append(f"{role}/{design}: t_inst does not carry the "
                                    f"instruction's last character")
                if not (offs[R.content_start][0] <= c0 and offs[R.content_end - 1][1] >= c1):
                    problems.append(f"{role}/{design}: content span does not cover "
                                    f"the instruction")
                if len(R.bleed_head) + len(R.bleed_tail) > MAX_BLEED_CHARS:
                    problems.append(f"{role}/{design}: bleed {R.bleed_head!r}+"
                                    f"{R.bleed_tail!r} exceeds {MAX_BLEED_CHARS} chars")
                if R.bleed_head or R.bleed_tail:
                    bleeds[f"{role}/{design}"] = (R.bleed_head, R.bleed_tail)
                renderings[(role, design)] = R.text

        check(f"{slug}: all {len(roles)} roles x {len(DESIGNS)} designs resolve",
              not problems,
              "; ".join(problems[:4]) if problems else
              f"roles={','.join(roles)}"
              + (f"  bleed: {bleeds}" if bleeds else "  no bleed"))

        # Role marking must actually change the prompt. If two roles render
        # identically the role contrast is between identical inputs, and every
        # downstream separation number for that pair is meaningless.
        for design in DESIGNS:
            texts = {r: renderings.get((r, design)) for r in roles
                     if (r, design) in renderings}
            dupes = [(a, b) for i, a in enumerate(sorted(texts))
                     for b in sorted(texts)[i + 1:] if texts[a] == texts[b]]
            check(f"{slug}/{design}: every role renders distinctly",
                  not dupes, f"identical renderings: {dupes}" if dupes else
                  f"{len(texts)} distinct role renderings")

    # ---- the closed-set guard -------------------------------------------
    # Not a style rule. See the module docstring: an unvetted role name renders
    # as a novel tag on Llama and Nemotron with no error anywhere.
    check("ROLES is the vetted closed set",
          tuple(sorted(ROLES)) == ("assistant", "system", "tool", "user"),
          f"ROLES={ROLES}")
    bad = []
    for name in ("cot", "thinking", "analysis", "reasoning", "developer"):
        try:
            render(None, PROBE, name, "fixed_slot")   # must fail before touching tok
            bad.append(name)
        except ValueError:
            pass          # correct: rejected as not in ROLES
        except Exception:
            pass          # also rejected, just later
    check("unvetted role names are rejected by render()", not bad,
          f"accepted without vetting: {bad}" if bad else
          "cot/thinking/analysis/reasoning/developer all rejected")

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("PASS - every roster model expresses every role class it claims")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
