import sys, json, logging
from pathlib import Path
sys.path.insert(0, "/home/b6aj/jtelintelo.b6aj/SecurityControl")
import pandas as pd
from experiments.rq1 import _adjudicate
log = logging.getLogger("readj"); log.setLevel(logging.INFO)
h = logging.StreamHandler(sys.stdout); h.setFormatter(logging.Formatter("%(message)s")); log.addHandler(h)
for slug in ["qwen2.5-7b", "qwen3.5-9b"]:
    p = Path(f"results/rq1/{slug}/causal_matrix.csv")
    if not p.exists(): continue
    print(f"\n=== {slug} ===")
    v = _adjudicate(pd.read_csv(p), log, slug)
    Path(f"results/rq1/{slug}/causal_gate.json").write_text(json.dumps(v, indent=2, default=str))
