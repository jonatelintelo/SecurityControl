# RQ1 — what is missing

Derived by auditing the RQ1 code against `notes_from_mail.tex` (scoped, wins on
conflict), `main.tex` (extraction and geometry method detail), `PLAN.md` and
`EXPERIMENTS.md`. Requirements already implemented are omitted; this file is the
work list only.

**Established from code and specification, never from results.** Whether an
earlier run happened to return a particular value cannot tell us whether the code
implements the specified experiment, so no result was consulted. Every item is
checkable by reading the repository.

---

## GAP 1 — E1.2 pass 2 is not implemented

**Required.** `PLAN-GEOM` specifies layer-wise overlap per pair via **principal
angles, projection measures and canonical correlations**, with `cos(r_a, r_b)` as
the one-dimensional special case, and states that *"the primary analysis will use
subspace-level measures where appropriate"*.

`EXPERIMENTS.md` breaks the circularity — subspace measures need `k`, and `k`
comes from E1.4 — by splitting E1.2 into two passes, and rules:

> If E1.4 returns `k = 1` for every concept, pass 2 is degenerate and pass 1 is
> the analysis — state that outcome explicitly rather than reporting principal
> angles between one-dimensional subspaces.

**Missing.** No pass-2 code exists: `principal_angle`, `canonical`,
`subspace_angle` and `cca` match nothing in the repository. `behavioural_k.json`
is written by `stage_dimensionality_behavioural` and **consumed by no analysis
stage** — only by `tools/rq1_walkthrough.py`, for display.

The pipeline therefore cannot execute the specified branch under *either*
outcome. It cannot report subspace geometry when `k > 1`, and it cannot record the
degeneracy finding when `k = 1`, because nothing reads `k` to make the
determination.

**To implement.** A stage that reads `k` per concept, builds each concept's rank-`k`
subspace from the same stratified construction `dim_behavioural` uses, and
computes between every pair:

- principal angles `θ_1 … θ_k`
- a projection measure — `‖P_A P_B‖_F / √min(k_A, k_B)`
- canonical correlations

against a **matched random-subspace null** (not a random-vector null), and against
the split-half floor. When every `k = 1` it must state the degeneracy explicitly
and defer to pass 1, rather than silently skipping.

---

## GAP 2 — the later-token readout is never exercised

**Required.** `PLAN-INF` specifies the active experiment as: steer at an early
layer/token, then measure the effect on projections onto other concepts
downstream, **"at later layers and later tokens"**.

**Missing.** `core/interventions.py:steer_and_capture` accepts `read_positions`,
defaulting to `(-1,)`. `experiments/rq1.py` calls it at lines 1717 and 1759
**without** that argument and indexes the single captured position as
`caps[l][:, 0, :]`. The causal matrix varies the read *layer* but never the read
*token*.

The machinery exists and is simply not used, so half of a named requirement is
unimplemented.

**To implement.** Pass a set of read positions to both the baseline and steered
capture, and carry a `read_position` column through `causal_matrix*.csv` so the
delta can be resolved per token as well as per layer.

---

## Scope limits — decisions to record, not code to write

| # | Limit | Note |
|---|---|---|
| S1 | Cross-family variation (PLAN-GEOM) is not testable | Qwen2.5-7B and Qwen3.5-9B are one family. `PLAN-SCOPE` targets ≥4 families; the scoped plan narrows the first pass to one dense + one MoE |
| S2 | Dense-versus-MoE not covered | `qwen3.5-35b-a3b` registered, deferred until dense RQ1–4 settle |
| S3 | `zhao_replication.csv` | Downgraded with reasons in `EXPERIMENTS.md` O-11, **but** the E1.1 checklist (line 1412) still marks it NOT DONE. Reconcile: one of the two statements must go |

## Documentation to reconcile

`EXPERIMENTS.md`'s `Produces` lists name artifacts the code does not write. The
content exists in every case, so this is a naming fix, not missing work — but a
stale list is how a silently-skipped stage stays invisible.

| Spec name | Where the content actually lives |
|---|---|
| `length_baselines.csv` | `length_only_auc` column of `direction_validation.csv` |
| `splithalf.csv` | `split_half_cos` column of `direction_validation.csv` |
| `role_probe_accuracy.csv` | `role_probe.csv` |
| `role_probe_transfer.csv` | `fidelity.json`, both directions |
| `harm_refusal_2x2.csv` | `labels_checks.json` → `strata` (richer: adds instruction and test-item counts) |
| `guard_sensitivity.csv` | `labels_checks.json` → `guard`, plus the `R_control_preguard` row |
| `null_band.json` | `geometry_null_band.json` |
| `corpus.json` | `corpus_meta.json` |

## Minor code issues

| # | Issue | Location |
|---|---|---|
| D1 | *Untrusted external content* is a fifth role class in `PLAN-EXTRACT`. `tool`, rendered as a `<tool_response>` block, is its natural stand-in, but that equivalence is nowhere stated as a decision | `core/config.py`, `ENVIRONMENT.md` |
| D2 | `relative_depth(li, len(common))` passes the number of *shared* layers as the model's layer count. Correct only when a pair shares every layer | `experiments/rq1.py` `stage_geometry` |

---

## Already applied during this audit

Code changes made before the rerun, recorded so the diff is traceable.

| Defect | Consequence had it stood |
|---|---|
| `tools/asymmetry_structure.py` read the pre-rename `causal_matrix.csv` and `continue`d when absent | Printed nothing while exiting 0 ever since the `__under`/`__over` rename; any finding sourced from it was stale |
| E1.1d fitted its refusal-conditioned harm variants with the plain estimator | Conditioning on the refusal label breaks the role balance that exempts pooled `R_harm`. The variants carried a role component, which **depresses** their cosine against a role-balanced `R_control` and so **overstates** separability |
| E1.1d recorded no length-only baseline | Its variants are fitted on a heavily imbalanced contrast where a length cue alone could produce near-perfect separation; nothing distinguished "harm direction" from "length direction" |
| `run_manifest.json` and `rq1.log` opened in truncate mode in a directory shared by per-model jobs | Each model's job erased the previous one's provenance record |
