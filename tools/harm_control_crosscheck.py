#!/usr/bin/env python
"""Do the two refusal-controlled harm directions agree — compared AT THE SAME LAYER?

Both hold refusal constant but sample different populations: `in_refused`
contrasts clearly-harmful against benign-but-sensitive items that triggered
over-refusal; `in_complied` contrasts the mildest harmful items against ordinary
harmless ones. If a refusal-free harm direction exists, the two should converge.

An earlier version of this check compared each variant at its OWN best layer —
layers 11 and 26 — which the analysis protocol forbids: the residual basis differs
by layer, so such a cosine is uninterpretable. Both variants are refitted at every
layer here and compared only within a layer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np, pandas as pd, torch
from core import extract
from core.io_utils import load_torch, save_df

for slug in ["qwen2.5-7b", "qwen3.5-9b", "qwen3.5-35b-a3b"]:
    d = Path("results/rq1")/slug
    blob = load_torch(d/"activations.pt")
    idx = pd.DataFrame(blob["index"])
    dirs = load_torch(d/"directions.pt")
    val = pd.read_csv(d/"direction_validation.csv")
    train = idx.split.eq("train").to_numpy()
    harmful = idx.harmful.astype(bool).to_numpy()
    print(f"\n{'='*74}\n{slug}\n{'='*74}")
    rows = []
    for pos, pooled in (("t_inst", "R_harm"), ("t_post_inst", "R_harm_at_post")):
        for li in range(len(blob[pos])):
            a = blob[pos][li]
            vecs = {}
            for name, lab in (("refused", "in_refused"), ("complied", "in_complied")):
                m = train & idx.label.eq(name).to_numpy()
                p, n = torch.tensor(m & harmful), torch.tensor(m & ~harmful)
                if int(p.sum()) < 25 or int(n.sum()) < 25:
                    continue
                # Role-balanced, matching `stage_harm_controls`. Conditioning on
                # the refusal label breaks the by-construction role balance that
                # exempts pooled `R_harm` from stratification, so a plain
                # difference of means here would carry a role component — and
                # would not be the same estimator the stage reports.
                d, _ = extract.stratum_balanced_diff_of_means(
                    a[p], a[n],
                    idx["role"].to_numpy()[p.numpy()].tolist(),
                    idx["role"].to_numpy()[n.numpy()].tolist(),
                    lab, li, "residual", pos, "harmful", "harmless")
                vecs[lab] = d.vector
            row = {"model": slug, "position": pos, "layer": li}
            if len(vecs) == 2:
                row["cos_refused_vs_complied"] = float(vecs["in_refused"] @ vecs["in_complied"])
            if pooled in dirs:
                pv = dirs[pooled][li].vector
                for k, v in vecs.items():
                    row[f"cos_{k}_vs_pooled"] = float(v @ pv)
            rows.append(row)
    df = pd.DataFrame(rows)
    save_df(d/"harm_controls_by_layer.csv", df)
    for pos in ("t_inst", "t_post_inst"):
        s = df[df.position == pos]
        if "cos_refused_vs_complied" not in s or s.cos_refused_vs_complied.isna().all():
            print(f"  {pos}: only one variant estimable — no comparison"); continue
        c = s.cos_refused_vs_complied.dropna()
        floor = float(val[val.concept == ("R_harm" if pos=="t_inst" else "R_harm_at_post")].split_half_cos.max())
        print(f"  {pos}: cos(in_refused, in_complied) across {len(c)} layers — "
              f"min {c.min():+.3f} median {c.median():+.3f} max {c.max():+.3f}")
        print(f"     split-half floor {floor:.3f} -> "
              f"{'CONVERGE somewhere' if c.max() >= floor else 'DIVERGE at every layer'}")
        for k in ("in_refused", "in_complied"):
            col = f"cos_{k}_vs_pooled"
            if col in s and not s[col].isna().all():
                q = s[col].dropna()
                print(f"     cos({k}, pooled) — median {q.median():+.3f} max {q.max():+.3f}")
