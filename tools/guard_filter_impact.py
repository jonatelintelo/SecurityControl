#!/usr/bin/env python
"""What does the Llama-Guard cross-check actually cost us?

It reclassifies `complied` -> `undetermined` when the response is judged not
harmful. That answer has two causes and the Guard cannot tell them apart:
  (a) the model did not really comply  -> correctly removed
  (b) the model DID comply, with a prompt whose compliance is not harmful
      -> wrongly removed
Case (b) would bias WHICH compliances survive, toward the most harmful prompts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd

for slug in ["qwen2.5-7b", "qwen3.5-9b", "qwen3.5-35b-a3b"]:
    p = Path("results/rq1")/slug/"refusal_labels.csv"
    if not p.exists(): continue
    d = pd.read_csv(p)
    h = d[d.harmful]
    pre = h[h.label_preguard == "complied"]
    kept = pre[pre.label == "complied"]
    drop = pre[pre.label_reason == "guard_says_response_safe"]
    print(f"\n=== {slug} ===")
    print(f"  prefix rule says complied : {len(pre)}")
    print(f"  kept by the Guard         : {len(kept)}")
    print(f"  removed by the Guard      : {len(drop)}  ({len(drop)/max(len(pre),1):.0%})")
    print("\n  SOURCE composition — does the filter change which prompts survive?")
    print(f"    {'source':<14}{'before':>8}{'after':>8}{'removed':>9}   share before -> after")
    for s in sorted(pre.source.unique()):
        b, a = int((pre.source==s).sum()), int((kept.source==s).sum())
        r = int((drop.source==s).sum())
        sb, sa = b/max(len(pre),1), a/max(len(kept),1)
        print(f"    {s:<14}{b:>8}{a:>8}{r:>9}   {sb:>6.0%} -> {sa:>4.0%}")
    # total-variation between the before/after source distributions
    keys = sorted(set(pre.source) | set(kept.source))
    tv = 0.5*sum(abs((pre.source==k).mean() - (kept.source==k).mean()) for k in keys)
    print(f"    total-variation shift in source composition: {tv:.3f}")
    if "role" in pre.columns:
        keys = sorted(set(pre.role))
        tvr = 0.5*sum(abs((pre.role==k).mean() - (kept.role==k).mean()) for k in keys)
        print(f"    total-variation shift in role composition  : {tvr:.3f}")
