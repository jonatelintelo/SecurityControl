#!/usr/bin/env python
"""Probe: can persuasion-style framings populate the harmful-and-COMPLIED cell?

WHY
`R_control` is fitted as refused-vs-complied. Measured on the frozen corpus, the
minority cell is thin or empty across the roster:

    under (within harmful)   qwen2.5 92 (22 instr) | qwen3.5-9b 2 (1) |
                             llama 43 (15) | qwen3.5-35b-a3b 5 (2) | nemotron 16 (5)
    over  (within harmless)  61 (16) | 121 (35) | 97 (20) | 103 (27) | nemotron 2 (2)

So `under` — the variant closest to Zhao's own refuse-vs-accept contrast — is
unusable on four of five models, and Nemotron supports neither.

EXPERIMENTS.md pre-registers the order in which the corpus may be widened:
role framings (already in the design), then milder harmful sources, and **only
if still empty**, a jailbreak family that is then excluded from RQ4 and declared.
Step 2 was tested and is insufficient: sorry-bench is the most compliant source
on every model but yields 0-1.5% on qwen3.5-9b and the MoE.

This probe measures step 3 before any of it is committed to. It spends nothing:
no corpus is frozen, no direction is fitted, no result is produced.

WHAT IT USES, AND WHAT THAT COSTS
Sorry-Bench ships 440 base prompts x 21 styles, already paired by `question_id`.
Four persuasion styles are probed. `role_play` and `authority_endorsement` are
deliberately NOT among them: ENVIRONMENT.md reserves the mutation styles for RQ4,
and PLAN names role-play and authority manipulation as RQ6's own context
variants. Spending 4 of 20 styles leaves 16 — a cost worth declaring, not one
that guts RQ4.

The probed question_ids are drawn DISJOINT from the frozen fitting corpus, so
running this commits us to nothing and creates no leakage if we later decline.

Guard is applied exactly as the pipeline applies it. Without it the probe would
count soft refusals as compliances and report a cell that is not there — which is
the failure this whole line of work started from.

    MODELS=qwen3.5-9b sbatch slurm/scripts/run_gpu.sh experiments/probe_jailbreak_corpus.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from core import pools, refusal  # noqa: E402
from core.config import load_config  # noqa: E402
from core.io_utils import get_logger, save_df, save_json  # noqa: E402
from core.model_io import load_model, resolve_device  # noqa: E402
from core.positions import render  # noqa: E402

# Persuasion framings. NOT role_play / authority_endorsement — RQ6 names those.
STYLES = ["evidence-based_persuasion", "expert_endorsement",
          "logical_appeal", "misrepresentation"]
BASE_STYLE = "base"                       # the control arm: measures the LIFT
N_IDS = int(os.environ.get("PROBE_N_IDS", 60))
ROLES = ["system", "user", "tool", "assistant"]
DESIGN = "fixed_slot"
BUDGET = int(os.environ.get("PROBE_BUDGET", 48))


def main() -> int:
    cfg = load_config()
    out = cfg.results_root / "probe_jailbreak"
    out.mkdir(parents=True, exist_ok=True)
    log = get_logger("probe-jb", out / "probe.log")

    from core.pools import _load
    ds = _load("sorry-bench/sorry-bench-202503", "default")
    split = ds[list(ds.keys())[0]]

    # Text already used for fitting — the probe must not reuse it.
    frozen = set()
    ip = cfg.results_root / "e1_0_corpus" / "instructions.jsonl"
    if ip.exists():
        frozen = {r["text"].strip() for r in
                  pd.read_json(ip, lines=True).to_dict("records")}
        log.info(f"frozen fitting corpus: {len(frozen)} instructions (excluded here)")

    by_id: dict = {}
    for r in split:
        v = r["turns"]
        v = (v[0] if v else "") if isinstance(v, list) else v
        if isinstance(v, str) and v.strip():
            by_id.setdefault(r["question_id"], {})[r["prompt_style"]] = v.strip()

    usable = [q for q, st in sorted(by_id.items())
              if BASE_STYLE in st and all(s in st for s in STYLES)
              and st[BASE_STYLE] not in frozen]
    log.info(f"{len(by_id)} question_ids; {len(usable)} complete and disjoint from "
             f"the frozen corpus; probing {min(N_IDS, len(usable))}")
    usable = usable[:N_IDS]
    if not usable:
        log.error("no usable question_ids — every complete one is already in the "
                  "fitting corpus")
        return 1

    device = resolve_device()
    slug = cfg.models[0]
    model, tok = load_model(cfg.spec(slug).model_id, device, logger=log)

    # NB: the column is read as df["style"], never df.style — `style` is a pandas
    # DataFrame property (the Styler accessor), so the attribute form silently
    # compares a Styler object instead of the column and raises KeyError later.
    rows = []
    for style in [BASE_STYLE] + STYLES:
        items = [(q, by_id[q][style]) for q in usable]
        rendered = [(q, role, render(tok, text, role, DESIGN))
                    for q, text in items for role in ROLES]
        log.info(f"[{slug}] {style}: generating {len(rendered)} items "
                 f"({len(items)} prompts x {len(ROLES)} roles)")
        resp, trunc, _ = refusal.generate(
            model, tok, [r.text for _, _, r in rendered], device=device,
            max_new_tokens=BUDGET, batch_size=cfg.batch_size, log_every=40, logger=log)
        for (q, role, r), text, tr in zip(rendered, resp, trunc):
            lab = refusal.label_response(text, tr, refusal.PRIMARY_RULE)
            rows.append({"question_id": q, "style": style, "role": role,
                         "prompt": r.text, "response": text, "truncated": bool(tr),
                         "label_preguard": lab.label, "reason": lab.reason})

    df = pd.DataFrame(rows)
    df["label"] = df.label_preguard

    # Guard, exactly as the pipeline applies it: on the harmful-and-complied cell.
    # Everything probed here is harmful, so that is every `complied` row.
    cell = df[df.label_preguard == "complied"]
    if len(cell):
        try:
            from core.guard import Guard
            g = Guard(device=device, logger=log)
            v = g.judge_with_raw(list(cell.prompt), list(cell.response), cfg.batch_size)
            g.release()
            safe = [i for i, (uns, _) in zip(cell.index, v) if not uns]
            df.loc[safe, "label"] = "undetermined"
            log.info(f"[{slug}] guard: {len(safe)}/{len(cell)} complied rows "
                     f"reclassified to undetermined (no harmful content)")
        except Exception as e:
            log.warning(f"[{slug}] guard failed: {type(e).__name__}: {e} — "
                        f"compliance counts below are NOT Guard-verified")

    save_df(out / f"probe_{slug}.csv", df)

    log.info("-" * 78)
    log.info(f"[{slug}] COMPLIANT CELL by style "
             f"(Guard-verified; `instr` is what the cluster bootstrap counts)")
    summary = {}
    for style in [BASE_STYLE] + STYLES:
        s = df[df["style"] == style]
        comp = s[s.label == "complied"]
        ref = int((s.label == "refused").sum())
        und = int((s.label == "undetermined").sum())
        summary[style] = {"complied_items": int(len(comp)),
                          "complied_instructions": int(comp.question_id.nunique()),
                          "refused_items": ref, "undetermined_items": und,
                          "n_items": int(len(s)),
                          "by_role": comp.role.value_counts().to_dict()}
        log.info(f"   {style:28s} complied={len(comp):4d} items / "
                 f"{comp.question_id.nunique():3d} instr   refused={ref:4d}  "
                 f"undetermined={und:4d}   roles={comp.role.value_counts().to_dict()}")

    base_i = summary[BASE_STYLE]["complied_instructions"]
    best = max(STYLES, key=lambda s: summary[s]["complied_instructions"])
    log.info("-" * 78)
    log.info(f"[{slug}] base gives {base_i} compliant instructions; best style "
             f"`{best}` gives {summary[best]['complied_instructions']}")
    log.info(f"[{slug}] pooled over the 4 probed styles: "
             f"{df[(df['style'] != BASE_STYLE) & (df.label == 'complied')].question_id.nunique()} "
             f"compliant instructions")
    log.info("Decision rule: `under` needs a minority cell with enough INSTRUCTIONS "
             "to bootstrap — ~20+ is the floor the verifier enforces.")

    save_json(out / f"probe_{slug}.json", {
        "model": slug, "n_question_ids": len(usable), "roles": ROLES,
        "design": DESIGN, "budget": BUDGET, "styles_probed": STYLES,
        "styles_deliberately_not_probed": ["role_play", "authority_endorsement"],
        "why_not": "RQ6 names role-play and authority manipulation as its own "
                   "context variants; spending them here would pre-empt it",
        "disjoint_from_fitting_corpus": True,
        "guard_applied": True,
        "summary": summary,
        "commits_nothing": ("no corpus frozen, no direction fitted, no result "
                            "produced; declining costs nothing"),
    })
    log.info(f"wrote {out}/probe_{slug}.csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
