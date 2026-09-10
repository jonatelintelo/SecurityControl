#!/usr/bin/env python
"""Does the E1.6 readout read the tokens and layers it claims to?

Three things changed in the causal readout and each fails *silently* rather than
raising, which is why they get a test:

1. **Per-item positions.** `t_inst` sits at a different index in every row, so
   the readout resolves positions per item over left-padded batches. Get it
   wrong and you read pad tokens — plausible numbers, no exception.
2. **Every downstream layer is read**, not three of them.
3. **Slicing moved into the capture hook** so memory no longer scales with the
   number of read layers. If that reordering broke "steer first, capture second",
   the capture would silently observe the *unsteered* value.

Checks 1 and 2 are pinned against `ActivationCapture.at_positions`, which is
already trusted and runs its own forward pass — so agreement is evidence about
the indexing rather than a tautology. Check 3 is a direct before/after on a
non-zero alpha.

    sbatch slurm/scripts/run_gpu.sh tests/smoke_read_positions.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from core import extract  # noqa: E402
from core.capture import ActivationCapture  # noqa: E402
from core.interventions import steer_and_capture  # noqa: E402
from core.model_io import load_model, resolve_device  # noqa: E402
from core.positions import render  # noqa: E402

MODEL = os.environ.get("SMOKE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")

# Deliberately ragged. If every prompt tokenised to the same length a single
# shared offset would pass, and the per-item path would go unexercised.
TEXTS = [
    "Write a haiku.",
    "Explain, in considerable and unnecessary detail, how a bicycle derailleur works.",
    "Hi.",
    "List three primes over one hundred and explain your reasoning briefly.",
    "What is the capital of France?",
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
    lens = [r.n_tokens for r in rendered]
    check("prompts are ragged (so per-item indexing is actually exercised)",
          len(set(lens)) > 1, f"token counts {lens}")

    n_layers = len(model.model.layers) if hasattr(model, "model") else 24
    steer_layer = 1
    read_layers = list(range(steer_layer + 1, n_layers))
    which = ("t_inst", "t_post_inst")

    ref = ActivationCapture(model, site="residual").at_positions(
        tok, rendered, which, read_layers, batch_size=2, device=device,
        dtype=torch.float32)

    got = steer_and_capture(model, tok, [r.text for r in rendered], None,
                            steer_layer=steer_layer, read_layers=read_layers,
                            alpha=0.0, device=device, batch_size=2,
                            read_rendered=rendered, read_which=which)

    check("every downstream layer was read",
          sorted(got) == read_layers,
          f"{len(got)} layers, {read_layers[0]}..{read_layers[-1]}")

    worst, where = 0.0, ""
    for li in read_layers:
        for pi, w in enumerate(which):
            d = float((ref[w][li].float() - got[li][:, pi, :].float()).abs().max())
            if d > worst:
                worst, where = d, f"L{li} {w}"
    check("per-item read matches ActivationCapture.at_positions",
          worst < 1e-3, f"max |delta| = {worst:.3e} (worst at {where})")

    # Negative control: the two positions must differ, or the test would pass
    # even if both axes read the same token.
    d = float((got[read_layers[0]][:, 0, :] - got[read_layers[0]][:, 1, :]).abs().max())
    check("the two positions read different tokens", d > 1e-3, f"max |delta| = {d:.3e}")

    # Steering must actually reach the capture. If the in-hook slicing had
    # reordered registration, this delta would be ~0.
    v = torch.randn(model.config.hidden_size if hasattr(model.config, "hidden_size")
                    else got[read_layers[0]].shape[-1])
    d_dir = extract.diff_of_means(
        torch.stack([v, v]), torch.stack([-v, -v]), "smoke", steer_layer,
        "residual", "t_post_inst", "a", "b")
    steered = steer_and_capture(model, tok, [r.text for r in rendered], d_dir,
                                steer_layer=steer_layer, read_layers=read_layers,
                                alpha=2.0, device=device, batch_size=2,
                                read_rendered=rendered, read_which=which)
    shift = float((steered[read_layers[0]] - got[read_layers[0]]).abs().max())
    check("steering reaches the capture (hook order: steer then capture)",
          shift > 1e-3, f"max |steered - baseline| = {shift:.3e} at L{read_layers[0]}")

    print(f"\n{'PASS' if not FAILURES else f'{FAILURES} FAILURE(S)'}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
