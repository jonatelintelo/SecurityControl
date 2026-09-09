#!/usr/bin/env python
"""Prune Slurm logs to the runs that produced current artifacts, and record why.

Slurm logs are the only record of what a job actually did — the per-experiment
manifest stores just the LAST job id for a directory, so a pipeline run across
several jobs is otherwise untraceable. Deleting logs indiscriminately therefore
destroys provenance; keeping all of them buries it. This keeps the ones that
produced live results and writes the job -> artifact mapping the manifests cannot.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT, ERR = ROOT/"slurm/1_run_phase/out", ROOT/"slurm/1_run_phase/err"

# job id -> what it produced. Everything not listed is a superseded or failed run.
KEEP = {
    # --- live results/ ---
    "6421361": "E1.0 corpus, full scale -> results/e1_0_corpus (FROZEN)",
    "6422011": "E1.1 Stage 1 labels, qwen2.5-7b (full-generation Guard)",
    "6422012": "E1.1 Stage 1 labels, qwen3.5-9b (full-generation Guard)",
    "6422312": "E1.1 extraction, qwen2.5-7b",
    "6422314": "E1.1 extraction, qwen3.5-9b",
    "6426909": "E1.2-E1.4 analysis, qwen2.5-7b (emergence crashed; see 6426988)",
    "6426988": "E1.5 emergence, both models (after train_auc fix)",
    "6427076": "E1.2-E1.5 analysis, qwen3.5-9b",
    "6427102": "E1.1 fidelity + cross-corpus transfer, qwen2.5-7b",
    "6427103": "E1.1 fidelity + cross-corpus transfer, qwen3.5-9b",
    "6430069": "E1.6 gate, qwen2.5-7b (generation readout)",
    "6430070": "E1.6 gate, qwen3.5-9b (generation readout)",
    "6432187": "E1.4b behavioural k, qwen2.5-7b (with the behavioural-null guard)",
    "6432189": "E1.4b behavioural k, qwen3.5-9b (with the behavioural-null guard)",
    "6432742": "E1.7 Level 1 style vs metadata, qwen2.5-7b",
    "6432744": "E1.7 Level 1 style vs metadata, qwen3.5-9b",
    "6430805": "E1.6 gate, qwen2.5-7b, CONTROL_VARIANT=over (matched cross-model test)",
    "6430807": "E1.5 emergence with onset CIs, qwen2.5-7b",
    "6430873": "E1.5 emergence with onset CIs, qwen3.5-9b",
    # --- independent reproduction ---
    "6427895": "reproduction: E1.0 corpus -> results_verify",
    "6427897": "reproduction: full RQ1 pipeline, qwen2.5-7b",
    "6427899": "reproduction: full RQ1 pipeline, qwen3.5-9b",
    # --- tests and verification ---
    "6427824": "tests/test_invariants.py — 22/22 environment invariants",
    "6428149": "tests/verify_rq1_run.py on results/ — 16/16",
    "6429332": "tests/verify_rq1_run.py, reproduction vs original (first pass)",
    # --- third-model probe ---
    "6430103": "download meta-llama/Llama-3.1-8B-Instruct",
    "6430148": "E1.0 render sweep, llama3.1-8b — FAILED (tool role quote-wrapping)",
    "6430264": "llama3.1-8b under-refusal probe (user role only)",
}

def job_id(p: Path) -> str:
    m = re.match(r"(\d+)-", p.name)
    return m.group(1) if m else ""

def main() -> int:
    dry = "--apply" not in sys.argv
    kept, removed, empty = [], [], []
    for d, suf in ((OUT, ".out"), (ERR, ".err")):
        for p in sorted(d.glob(f"*{suf}")):
            jid = job_id(p)
            if jid in KEEP:
                kept.append(p)
            else:
                (empty if p.stat().st_size == 0 else removed).append(p)
    print(f"keep {len(kept)}  remove {len(removed)}  remove-empty {len(empty)}")
    if not dry:
        for p in removed + empty:
            p.unlink()
        lines = ["# Slurm logs retained", "",
                 "Kept because each produced a live artifact. Job id -> what it produced;",
                 "the per-experiment `run_manifest.json` records only the last job for a",
                 "directory, so this is the mapping for multi-job pipeline runs.", "",
                 "| job | produced |", "|---|---|"]
        for j, why in sorted(KEEP.items()):
            lines.append(f"| `{j}` | {why} |")
        lines += ["", f"Pruned {len(removed)} superseded and {len(empty)} empty log files.",
                  "Superseded runs are those whose outputs were withdrawn or",
                  "overwritten by a later fix; see docs/REMOVED_ARCHIVES.md."]
        (ROOT/"slurm/1_run_phase/KEPT.md").write_text("\n".join(lines) + "\n")
        print("wrote slurm/1_run_phase/KEPT.md")
    else:
        print("(dry run — pass --apply to delete)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
