#!/usr/bin/env python
"""Record a Claude-session adjudication of the blind audit sheet as an artifact.

WHAT THIS IS, AND ITS LIMIT
O-1 asks how good the refusal-label instrument is. The honest answer needs a
reader independent of the rule. Two LLM judges were run and agreed with each
other only 61.9%, which means neither alone is a usable reference — so a third,
more capable reader is worth having.

That reader is a Claude session, and the limitation is specific and must travel
with the numbers: **this adjudication is not reproducible by re-running code.**
A future run cannot regenerate it. It is therefore stored as DATA — a versioned
CSV keyed by `audit_id` — exactly as a human labeller's sheet would be, and the
`adjudicator` column names the session so the provenance is explicit.

It is also not a human audit. It is a third automated instrument, more capable
than the other two but sharing their basic failure mode: a language model
judging language-model output. O-1 stays open until a person reads a sample.

    python tools/record_claude_adjudication.py labels.json
      where labels.json is {"<audit_id>": "refused|complied|degenerate|unsure", ...}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

D = Path(os.environ.get("AUDIT_DIR", "results/label_audit"))
SESSION = os.environ.get("CLAUDE_SESSION", "claude-opus-5")
VALID = {"refused", "complied", "degenerate", "unsure"}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    labels = json.loads(Path(sys.argv[1]).read_text())
    bad = {k: v for k, v in labels.items() if v not in VALID}
    if bad:
        print(f"invalid labels (must be one of {sorted(VALID)}): {list(bad.items())[:5]}")
        return 1

    sheet = pd.read_csv(D / "audit_sheet.csv")
    out = D / "audit_claude.csv"
    rows = []
    missing = []
    for r in sheet.itertuples():
        aid = str(r.audit_id)
        if aid in labels:
            rows.append({"audit_id": r.audit_id,
                         "adjudicated_label": labels[aid],
                         "adjudicator": f"claude:{SESSION}"})
        else:
            missing.append(aid)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out}: {len(rows)} labels ({len(missing)} sheet rows unlabelled)")
    if missing:
        print(f"  first unlabelled: {missing[:8]}")
    from collections import Counter
    print("  distribution:", dict(Counter(r['adjudicated_label'] for r in rows)))
    print("\nNOTE: not reproducible by re-running code; stored as data, like a "
          "human labeller's sheet. Not a substitute for a human pass.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
