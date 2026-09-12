# RQ1 — the gap list, closed

This file was written before the RQ1 run, as a list of what was missing. **Every
item on it is now closed.** It is kept as a closure record — each gap names the
artifact that proves it shut — because a list of gaps with no resolution is how a
silently-skipped stage stays invisible, and deleting it would remove the audit
trail rather than complete it.

Superseded content (the "to implement" designs, the open scope limits) is in git
history; what follows is the current state.

## The three gaps — all closed

| # | Gap as written | Closed by | Proof |
|---|---|---|---|
| GAP 1 | E1.2 pass 2 not implemented — no principal angles, projection metric or CCA | `extract.principal_angles`, `extract.data_canonical_correlations`, `extract.random_subspace_null`; stage `geometry_subspace` | `geometry_subspace.csv`, `geometry_subspace_summary.json`, all six models |
| GAP 2 | the later-token readout is never exercised — `read_positions` accepted but never passed | `steer_and_capture` gained `read_rendered`/`read_which`; slicing moved inside the capture hook so memory is independent of read-layer count | `causal_matrix__*.csv` carries **both** `t_inst` and `t_post_inst` |
| S3 | `zhao_replication.csv` marked NOT DONE in one place and downgraded in another | emitted from the `advbench\|alpaca` refit, swept layer-wise | `zhao_replication.csv`, all six models |

## The scope limits, as they actually resolved

| # | Limit as written | What happened |
|---|---|---|
| S1 | "cross-family variation is not testable — Qwen2.5 and Qwen3.5 are one family" | **Resolved.** The roster is six models across **four vendors** (Qwen, Meta, 01-AI, NVIDIA) and five template families. |
| S2 | "dense-versus-MoE not covered; MoE deferred" | **Resolved.** Two MoE models (`qwen3.5-35b-a3b`, `nemotron-3-nano-30b-a3b`) ran the full 14 stages in both roots. |

## The two minor issues

* **D1** — `tool` as the stand-in for PLAN-EXTRACT's *untrusted external content*
  class is now recorded as a decision in `EXPERIMENTS.md`, and is **discharged**:
  every roster model renders all four role classes, so the caveat applies to none
  of them. `tests/verify_rq1_run.py` still emits it if a future model lacks `tool`.
* **D2** — `relative_depth(li, len(common))` fixed.

## Artifact naming

The `Produces` lists in `EXPERIMENTS.md` named files the code does not write; the
content existed under other names in every case. Reconciled there, not here.

---

## Still open after the run — the one item

**O-1: the refusal label is not validated by a human.** Three independent readers
now exist (gemma-3-27b, Mixtral-8x7B, a Claude session), and they agree with each
other far better than with the prefix rule (kappa 0.67-0.89 between readers,
0.54-0.72 reader-versus-rule). That bounds the instrument but does not close the
item: all three are language models judging language-model output, and a person
has still read none of it. `results/label_audit/` holds the blind sheet, the
per-adjudicator labels and the agreement matrix, so a human pass would slot in
directly.
