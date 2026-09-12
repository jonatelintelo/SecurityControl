#!/usr/bin/env python
"""Regenerate every GATE 1 artifact from its saved matrix, on CPU.

WHY THIS IS SOUND
A gate verdict is a deterministic function of `causal_matrix__<variant>.csv` and
the adjudication options — no model, no generation, no randomness beyond the
seeded bootstrap already baked into the matrix's CIs. So when the adjudication
logic changes, the verdicts can be recomputed from saved matrices instead of
re-running `causal` on six models across two roots, which is days of GPU.

WHY IT WAS NEEDED
G3 admitted alphas whose behavioural null band was exactly 0.0. A band of zero
carries no information and makes the criterion vacuous at that magnitude: any
non-zero shift clears it. On qwen3.5-35b-a3b the band at |alpha|=0.25 was 0.0 and
206 of 362 qualifying cells came from that alpha alone — every one a 0.01 shift
against a 0.0 threshold. The verdict survived on informative alphas, but the
count was inflated, and the next model to hit this might not be so lucky.

The previous artifacts are kept as `causal_gate__<variant>.superseded.json` so the
change is auditable rather than silent.

    python tools/readjudicate_gates.py            # rewrite in place, keeping backups
    python tools/readjudicate_gates.py --dry      # report what would change
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from core.config import RQ1_MODELS  # noqa: E402
from core.io_utils import get_logger  # noqa: E402
from experiments.rq1 import _adjudicate  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--roots", default="results,results_verify")
    args = ap.parse_args()
    log = get_logger("readjudicate", Path("results") / "readjudicate_gates.log")

    changed, same = 0, 0
    for root in args.roots.split(","):
        for m in RQ1_MODELS:
            d = Path(root) / "rq1" / m
            for mp in sorted(d.glob("causal_matrix__*.csv")):
                tag = mp.stem.replace("causal_matrix__", "")
                gp = d / f"causal_gate__{tag}.json"
                if not gp.exists():
                    continue
                old = json.loads(gp.read_text())
                mat = pd.read_csv(mp)
                new = _adjudicate(mat, log, m)
                o_v, n_v = old.get("verdict"), new.get("verdict")
                o_n = {b: v["G3_behavioural_dissociation"]["n"]
                       for b, v in old.get("per_bound", {}).items()}
                n_n = {b: v["G3_behavioural_dissociation"]["n"]
                       for b, v in new.get("per_bound", {}).items()}
                dropped = {b: v["G3_behavioural_dissociation"].get(
                    "n_cells_excluded_degenerate_band", 0)
                    for b, v in new.get("per_bound", {}).items()}
                moved = (o_v != n_v) or (o_n != n_n)
                tagline = (f"{root}/{m}/{tag}: verdict {o_v} -> {n_v}; "
                           f"G3 n {o_n} -> {n_n}; degenerate cells dropped {dropped}")
                if moved:
                    changed += 1
                    log.info("CHANGED  " + tagline)
                    if o_v != n_v:
                        log.warning(f"  VERDICT CHANGED for {root}/{m}/{tag}: "
                                    f"{o_v} -> {n_v} — this must be reported, not "
                                    f"quietly adopted")
                else:
                    same += 1
                    log.info("unchanged " + tagline)
                if not args.dry:
                    # Rewrite EVERY gate, not only the ones whose numbers moved.
                    #
                    # Writing only the changed ones leaves the artifact set split
                    # across two code states: some files carry the new provenance
                    # fields (`degenerate_alphas_excluded`,
                    # `qualifying_cells_by_abs_alpha`) and some do not, and a
                    # reader cannot tell whether a gate was adjudicated under the
                    # current logic or merely happened to be unaffected by it.
                    # Numerically identical is not the same as verifiably current.
                    if moved:
                        shutil.copy2(gp, gp.with_suffix(".superseded.json"))
                    gp.write_text(json.dumps(new, indent=2, default=str))

    log.info("-" * 78)
    log.info(f"{changed} gate artifact(s) changed, {same} unchanged"
             + (" (dry run — nothing written)" if args.dry else
                " (previous versions kept as *.superseded.json)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
