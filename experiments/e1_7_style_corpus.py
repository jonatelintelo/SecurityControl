#!/usr/bin/env python
"""E1.7 Level 2 — build, verify and freeze the controlled-style corpus.

Spec: EXPERIMENTS.md > RQ1 > E1.7. PLAN-EXTRACT asks whether *"explicit role
metadata and linguistic style produce compatible or conflicting
representations."* Level 1 can only measure how the two co-vary in the corpus as
it happens to be; separating them needs a corpus where they are **crossed** — the
same request in three registers, under every role tag.

Why this is its own build and not part of E1.0: `e1_0_corpus.py` is offline and
model-free by design, so every downstream experiment inherits something checkable
without a GPU. Rewriting needs generation, so it lives here and produces its own
frozen artifact.

**Rewrites are generated, not templated.** A templated rewrite puts the register
in a fixed framing phrase at the edges of otherwise verbatim content, so a
contrast fitted across it is a framing-phrase contrast — orthogonality to that
would license only "R_role is not the framing phrase", which is not the claim.
Generation redistributes register through the text. The template arm is retained
as a **control**: templated register is trivially separable, so it upper-bounds
how detectable style can be.

**The rewriter is not a model under test.** Generating the corpus with a model
that is then probed on it would leave the register cue and the probe entangled
through the same weights.

    sbatch slurm/scripts/run_gpu.sh experiments/e1_7_style_corpus.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataclasses import asdict  # noqa: E402
from typing import List, Optional  # noqa: E402

from core import pools, refusal  # noqa: E402
from core.config import load_config  # noqa: E402
from core.io_utils import get_logger, save_json  # noqa: E402
from core.model_io import load_model, resolve_device  # noqa: E402

NAME = "e1_7_style"
CORPUS = "e1_0_corpus"

# Cached, instruction-tuned, and NOT in the RQ1 roster.
REWRITER = os.environ.get("STYLE_REWRITER", "Qwen/Qwen3-30B-A3B-Instruct-2507")
N_BASE = int(os.environ.get("N_STYLE_BASE", 100))
MAX_NEW = int(os.environ.get("STYLE_MAX_NEW_TOKENS", 128))
# Below this the register contrast is fitted on too few distinct requests.
MIN_COMPLETE_BASES = int(os.environ.get("MIN_COMPLETE_BASES", 40))


def _balanced(s: str, open_ch: str, close_ch: str) -> Optional[str]:
    """The first balanced `open_ch ... close_ch` block, or None."""
    i = s.find(open_ch)
    if i < 0:
        return None
    depth, in_str, esc = 0, False, False
    for j in range(i, len(s)):
        c = s[j]
        if in_str:
            if esc:            esc = False
            elif c == "\\":    esc = True
            elif c == '"':     in_str = False
            continue
        if c == '"':           in_str = True
        elif c == open_ch:     depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return s[i:j + 1]
    return None


def _clean(raw: str) -> str:
    """Strip the chatter a rewriter adds around the rewrite itself.

    **Structure-aware, and it has to be.** An earlier version ended with
    `s.split("\n\n")[0]`, which is right for prose — a rewrite followed by "Let
    me know if you want another version!" — but destroys a structured rewrite the
    moment it contains a blank line:

        {"action": "calculate_sales_tax_rate",\n\n "jurisdiction": "California"}
        -> {"action": "calculate_sales_tax_rate",

    That silently truncated most `tool_register` rewrites, which then failed
    content verification and were read as the rewriter being bad at structured
    register. It was the cleaner all along. So: if the output contains a balanced
    JSON-ish block, that block IS the rewrite and is returned whole; the
    paragraph rule applies only to prose.
    """
    s = raw.strip()
    for lead in ("Sure,", "Here is", "Here's", "Rewrite:", "Rewritten:"):
        if s.lower().startswith(lead.lower()):
            s = s.split(":", 1)[-1].strip() if ":" in s[:40] else s[len(lead):].strip()
    for o, c in (("{", "}"), ("[", "]")):
        block = _balanced(s, o, c)
        if block and len(block) >= 10:
            return block.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] == s[0]:
        s = s[1:-1].strip()
    return s.split("\n\n")[0].strip()


def main() -> int:
    cfg = load_config()
    out = cfg.dir(NAME)
    log = get_logger(NAME, out)
    from core.io_utils import write_run_manifest
    write_run_manifest(cfg, NAME)

    fitting, _, _ = pools.load_corpus(cfg.results_root / CORPUS)
    bases = pools.select_style_bases(fitting, N_BASE, seed=cfg.seed)
    log.info(f"selected {len(bases)} base instructions "
             f"(harmful {sum(b.harmful for b in bases)}, "
             f"test-side {sum(b.split == 'test' for b in bases)})")

    device = resolve_device()
    model, tok = load_model(REWRITER, device, logger=log)

    # One generation pass per register, so a register's prompt is constant across
    # every item and cannot vary with position in a mixed batch.
    generated: List[pools.StyleItem] = []
    for reg, instruction in pools.STYLE_REGISTERS.items():
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": f"{instruction}\n\n---\n{b.text}"}],
                tokenize=False, add_generation_prompt=True)
            for b in bases]
        log.info(f"[{reg}] generating {len(prompts)} rewrites (max_new={MAX_NEW})")
        raw, _, _ = refusal.generate(model, tok, prompts, device, MAX_NEW,
                                     cfg.batch_size, log_every=10, logger=log)
        for b, r in zip(bases, raw):
            generated.append(pools.StyleItem(
                text=_clean(r), base_uid=b.uid, base_text=b.text, register=reg,
                arm="generated", template_id=-1, harmful=bool(b.harmful),
                source=b.source, split=b.split))

    verified, report = pools.verify_style_items(generated)
    log.info(f"verification: kept {report['n_kept']}/{report['n_in']} "
             f"(rejected {report['n_rejected']}: "
             f"{report['n_rejected_rewriter_refused']} refusals, "
             f"{report['n_rejected_inverted']} inversions, "
             f"rest coverage/length); mean coverage {report['mean_coverage_kept']}")
    for ex in report["examples_rejected"]:
        log.info(f"  rejected cov={ex['coverage']}: {ex['base'][:58]!r} -> {ex['rewrite'][:58]!r}")

    # Balance by construction, not by threshold: a base is kept only if it
    # survived in EVERY register, so the register contrast compares the same
    # requests and the within-base pairing is well defined.
    kept, comp = pools.require_complete_registers(verified, list(pools.STYLE_REGISTERS))
    log.info(f"complete registers: {comp['n_bases_complete']}/{comp['n_bases_seen']} bases "
             f"({comp['completion_rate']:.1%}); missing-by-register {comp['missing_register_counts']}")

    complete_bases = {it.base_uid for it in kept}
    items = kept + [t for t in pools.build_template_arm(bases)
                    if t.base_uid in complete_bases]
    meta = {
        "rewriter": REWRITER, "n_base_selected": len(bases),
        "n_bases_complete": comp["n_bases_complete"],
        "n_generated_kept": len(kept), "n_template": len(items) - len(kept),
        "verification": report, "completion": comp,
        "crossing": "base x register x arm, then x role tag at render time",
        "arms": {"generated": "register redistributed through the text (primary)",
                 "template": "verbatim content in a fixed frame (control; "
                             "upper-bounds how detectable register can be)"},
        "min_complete_bases": MIN_COMPLETE_BASES,
        "usable": comp["n_bases_complete"] >= MIN_COMPLETE_BASES,
    }
    pools.save_style_corpus(out, items, meta)
    log.info(f"wrote {len(items)} style items ({len(kept)} generated + "
             f"{len(items)-len(kept)} template control) over "
             f"{comp['n_bases_complete']} complete bases to {out}")

    if comp["n_bases_complete"] < MIN_COMPLETE_BASES:
        log.error(f"E1.7 Level 2 NOT usable — only {comp['n_bases_complete']} bases "
                  f"survived in every register, below the {MIN_COMPLETE_BASES} needed "
                  f"to fit a register contrast. Raise N_STYLE_BASE and rerun.")
        return 1
    log.info("E1.7 Level 2 corpus FROZEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
