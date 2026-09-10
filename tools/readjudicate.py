#!/usr/bin/env python
"""Re-adjudicate every saved gate matrix with the current logic.

Adjudication is analysis, not measurement: the matrix is the measurement. So a
change to the criteria never requires re-running the sweep on GPU.
"""
import json, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from experiments.rq1 import _adjudicate

log = logging.getLogger("readj"); log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout); h.setFormatter(logging.Formatter("%(message)s")); log.addHandler(h)

for mp in sorted(Path("results/rq1").glob("*/causal_matrix*.csv")):
    slug = mp.parent.name
    tag = mp.stem.replace("causal_matrix", "").lstrip("_") or "default"
    print(f"\n=== {slug} / {tag} ===")
    v = _adjudicate(pd.read_csv(mp), log, f"{slug}/{tag}")
    (mp.parent / f"causal_gate__{tag}.json").write_text(json.dumps(v, indent=2, default=str))
