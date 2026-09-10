#!/usr/bin/env python
"""Does restricting WHICH tokens are steered actually change the intervention?

Stage B sweeps the pre-registered steered-position sets (all real / instruction
span / `t_inst` only / post-instruction). `position_masks` has existed on both
intervention primitives for a long time while never being passed by any
experiment, so this is the first code path to exercise it — and every way it can
fail is silent:

* a mask built against a padded width rather than `r.n_tokens` raises loudly
  (good), but one built against the wrong *span* does not;
* an all-False mask that reached the hook as "no mask" would steer the entire
  prompt while being reported as a restricted set;
* a mask that never reaches the hook makes all four token sets return identical
  numbers, which reads as "token position does not matter" — a finding, not a bug.

The checks below distinguish those cases from a working sweep.

    sbatch slurm/scripts/run_gpu.sh tests/smoke_steer_positions.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from core import extract  # noqa: E402
from core.interventions import kl_from_baseline, next_token_stats  # noqa: E402
from core.model_io import load_model, resolve_device  # noqa: E402
from core.positions import render  # noqa: E402
from experiments.rq1 import TOKEN_SETS, _steer_mask  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
TEXTS = [
    "Explain how a bicycle derailleur works.",
    "Summarise the causes of the 1929 crash.",
    "What is the capital of France?",
    "List three primes over one hundred.",
]

FAILURES = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILURES
    FAILURES += 0 if ok else 1
    print(f"[{'ok  ' if ok else 'FAIL'}] {name}" + (f"\n         {detail}" if detail else ""))


def main() -> int:
    device = resolve_device()
    model, tok = load_model(MODEL, device)
    rendered = [render(tok, t, "user", "fixed_slot") for t in TEXTS]
    texts = [r.text for r in rendered]

    # Masks must be in unpadded coordinates and must not run past the prompt.
    bad = []
    for r in rendered:
        for t in TOKEN_SETS:
            if t == "all_real":
                continue
            m = _steer_mask(r, t)
            if len(m) != r.n_tokens:
                bad.append(f"{t}: len {len(m)} != n_tokens {r.n_tokens}")
            elif any(m) and max(i for i, v in enumerate(m) if v) > r.t_post_inst:
                bad.append(f"{t}: span runs past t_post_inst")
    check("masks are in unpadded coordinates and stay inside the prompt", not bad,
          "; ".join(bad[:3]) or f"{len(rendered)} items x {len(TOKEN_SETS)-1} sets")

    layer = max(1, len(model.model.layers) // 3)
    a = torch.randn(model.config.hidden_size)
    d = extract.diff_of_means(torch.stack([a, a]), torch.stack([-a, -a]),
                              "smoke", layer, "residual", "t_post_inst", "p", "n")

    base = next_token_stats(model, tok, texts, None, layer, 0.0, device, batch_size=2)
    kls = {}
    for t in TOKEN_SETS:
        pm = None if t == "all_real" else [_steer_mask(r, t) for r in rendered]
        st = next_token_stats(model, tok, texts, d, layer, 2.0, device, batch_size=2,
                              position_masks=pm)
        kls[t] = float(kl_from_baseline(base["logprobs"], st["logprobs"]).mean())
        n = "all" if pm is None else f"{sum(sum(m) for m in pm)} tok"
        print(f"    {t:<18} steered {n:<10} KL {kls[t]:.5f}")

    check("every token set produces a non-zero effect",
          all(v > 1e-6 for v in kls.values()),
          "a zero here means the mask blocked every position")

    # The decisive one: masks must REACH the hook. If they did not, every set
    # would steer the whole prompt and return the same KL as all_real.
    distinct = {k: v for k, v in kls.items() if k != "all_real"}
    same_as_all = [k for k, v in distinct.items()
                   if abs(v - kls["all_real"]) < 1e-6]
    check("restricting the steered tokens changes the effect",
          not same_as_all,
          f"identical to all_real: {same_as_all}" if same_as_all else
          f"spread {min(kls.values()):.5f}..{max(kls.values()):.5f}")

    # Steering one token cannot plausibly do as much as steering every token.
    check("t_inst_only has a smaller effect than all_real",
          kls["t_inst_only"] < kls["all_real"],
          f"t_inst_only {kls['t_inst_only']:.5f} vs all_real {kls['all_real']:.5f}")

    print(f"\n{'PASS' if not FAILURES else f'{FAILURES} FAILURE(S)'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
