# SecurityControl — the causal architecture of LLM safety

Research code and planning documents for a study of whether LLM safety is implemented by
a structured, measurable causal architecture (role perception, harmfulness recognition,
behavioural control) rather than a single refusal direction, and whether different attacks
compromise different stages of it.

## Documents

| File | Holds | Changes when |
|---|---|---|
| [plan.md](plan.md) | motivation, hypotheses, research questions, related work and novelty, contributions, the experiment programme at design level, gates, scope | the science changes |
| [experiments.md](experiments.md) | the methodology and exact experimental setup for every experiment named in the plan; model-agnostic; no results, no assumptions | an experiment's setup changes |
| [assumptions.md](assumptions.md) | every design assumption and environment fact the two files above rely on, with origin, use, verification procedure and status | a check is run |
| `rq1_findings.md` … `rq4_findings.md` | results, per research question (not yet written) | a run produces evidence |
| [archive/](archive/) | the superseded previous iteration; cross-checking only | never |

Read them in that order. A measured number never appears in the first two files.

## Code status

Everything under `core/`, `experiments/`, `tools/`, `tests/` and `slurm/` is **legacy from
the superseded iteration**. It is not extended. It is recycled or removed piece by piece
against `experiments.md`: a module survives only if an experiment there needs it and it
passes that experiment's verification. `third_party/` holds pinned external attack code
(NeuroStrike, L³) and is not ours to modify.
