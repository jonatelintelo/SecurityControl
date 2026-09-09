#!/usr/bin/env python
"""Drop recomputable activation caches, keep everything that is evidence.

`activations.pt` is a cache, not a result: it is regenerated deterministically by
`rq1.py --only extract` (verified byte-exact for label-independent concepts across
two independent runs). The CSV/JSON artifacts derived from it are the evidence and
are never touched.

The live root keeps its cache so CPU analysis stages can re-run without a GPU.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# root -> why its cache is disposable
PRUNE = {
    "results_verify": "reproduction run; its evidence is the CSV comparison, already verified",
    # results_smoke was removed outright once a real reproduction existed; the entry
    # is kept so the script still works if a smoke root is recreated.
    "results_smoke": "smoke runs at FAST_DEV scale; never result-bearing",
}
KEEP = {"results": "live — CPU stages re-read this cache without a GPU"}


def main() -> int:
    apply = "--apply" in sys.argv
    total = 0
    for name, why in PRUNE.items():
        d = ROOT / name
        if not d.exists():
            continue
        hits = sorted(d.rglob("activations.pt"))
        size = sum(p.stat().st_size for p in hits)
        total += size
        print(f"{name}: {len(hits)} cache file(s), {size/2**30:.1f} GiB — {why}")
        if apply:
            for p in hits:
                p.unlink()
    for name, why in KEEP.items():
        d = ROOT / name
        hits = sorted(d.rglob("activations.pt")) if d.exists() else []
        print(f"{name}: KEEPING {len(hits)} cache file(s) "
              f"({sum(p.stat().st_size for p in hits)/2**30:.1f} GiB) — {why}")
    print(f"\n{'reclaimed' if apply else 'would reclaim'}: {total/2**30:.1f} GiB")
    if apply:
        note = ROOT / "results_verify" / "CACHES_PRUNED.md"
        if (ROOT / "results_verify").exists():
            note.write_text(
                "# Activation caches pruned\n\n"
                "`activations.pt` removed from this root to reclaim disk. It is a\n"
                "**cache, not a result**: regenerate with\n\n"
                "    RESULTS_ROOT=./results_verify MODELS=<slug> \\\n"
                "        sbatch slurm/scripts/run_gpu.sh experiments/rq1.py --only extract\n\n"
                "Extraction is deterministic — an independent re-run reproduced every\n"
                "label-independent direction at max |delta AUC| = 0.000000 — so the\n"
                "regenerated cache is identical for those concepts. Every CSV/JSON\n"
                "artifact derived from it is retained here unchanged.\n")
    elif not apply:
        print("(dry run — pass --apply)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
