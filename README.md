# SecurityControl — The Causal Architecture of LLM Safety

Research code asking whether LLM safety is implemented by a **structured, measurable
causal architecture** rather than a single "refusal direction", and whether the
*structure* of that architecture predicts how easily it can be broken.

Hypothesis under test: safety decomposes into interacting latent variables
(`R_role`, `R_harm`, `R_control`); different attacks compromise different parts of
that architecture; and its concentration/redundancy predicts the intervention
budget needed to disable it.

---

## Start here

Documentation is split by **epistemic status**, because a previous single-file
playbook let results contaminate the specification. A claim's kind determines
which file it lives in.

| File | Holds | Changes when |
|---|---|---|
| **[PLAN.md](PLAN.md)** | The research plan, extracted from the two `.tex` sources, with their conflicts resolved | the sources change |
| **[ENVIRONMENT.md](ENVIRONMENT.md)** | Verified properties of models, chat templates, datasets, cluster — plus the traps that silently produce wrong numbers | never; re-verified, not re-decided |
| **[EXPERIMENTS.md](EXPERIMENTS.md)** | Designs, pre-registered criteria, parameters, status. **RQ1–RQ4** | we argue ourselves out of a decision |
| **[docs/RQ1_FINDINGS_TEMPLATE.md](docs/RQ1_FINDINGS_TEMPLATE.md)** | The claims RQ1 must support (C1-C10), each with the artifact that settles it. **The findings document itself is not written yet** — the reporting decisions wait until RQ2-RQ4 are in, so that what goes in 9 pages of main body is decided once | a run produces new evidence |

Read them in that order. **The only numbers in `EXPERIMENTS.md` are thresholds
fixed before a run** — a measured value appearing there is a bug.

The most important thing to understand about this project: **almost every headline
claim so far has been killed or reshaped by a control.** The controls are the main
experimental apparatus, not decoration. The corrections are recorded where they were made — in the code comments that
explain why a check exists, in `docs/REMOVED_ARCHIVES.md`, and in
`slurm/evidence/`. Several are worth knowing before reading any number: a gate
criterion that could never fire, an AUC that mis-ranked ties, a refusal rule
defeated by a typographic apostrophe, and a projection result that reversed once
the pool it ran on was split by what the items actually were.

---

## Where RQ1 stands

**Every RQ1 stage is settled**, on six models across four vendors and both
architectures (dense and MoE), reproduced in two independent results roots. In
brief:

- Three variables are **representationally distinct** — recoverable against a
  1000-draw anisotropy null, length-only baselines and cluster-bootstrapped CIs,
  with a positive control that reaches the noise floor when two directions really
  are the same variable.
- **Gate 1 PASSES** — G2 (off-diagonal asymmetry) ∧ G3 (behavioural dissociation)
  — on all six models at all five capability bounds, on the matched
  `CONTROL_VARIANT=over` arm. Robust in **3,168 of 3,168** alpha-matched
  adjudication settings; only pooling the null (a known error, swept to show it is
  the one load-bearing choice) reverses it.
- One directed relation replicates across the roster: **`R_harm → R_control` in
  920 of 947 qualifying cells** (97.1%, KL ≤ 0.5), and at **100% on five of the six
  models** (yi-6b-chat 185/212). `R_role`'s outgoing edges are the weakest in the
  matrix and are MIXED or merely leaning on five of six models — only Nemotron
  gives both role edges a consistent direction. So the plan's candidate chain
  `R_role → R_harm → R_control` gets its **second link supported and its first link
  not**. That is RQ2's opening question, not RQ1's answer.
- The variables are **causally distinguishable while geometrically entangled**.
  On the matched `over` arm (the only control variant fitted on all six models),
  `R_harm_at_post` and `R_control_harmless` reach **0.53–0.86 of the split-half
  ceiling**; on the 5 models where the `under` variant is fitted, 0.57–0.86.
  Distinguishability here is a **causal** claim, not a geometric one — the two
  directions overlap substantially and still dissociate under intervention.
  One exception worth stating rather than hiding in an absolute value: on
  **yi-6b-chat the cosine is negative** (−0.50), so harm and over-refusal control
  are anti-aligned there, not merely less aligned.
- All directions are **style-dependent**, and for harmfulness the variation is
  driven mainly by *which harmful dataset* is used — AdvBench and JBB agree
  (cos 0.92–0.96) while Sorry-Bench is far from both (−0.02 to 0.48, per-model
  maxima 0.36–0.48) against a split-half floor of ~0.99, on every model.

**The gate's verdict was FAIL until two defects were found** — a behavioural null
band of exactly zero that made G3 vacuous, and a refusal rule defeated by the
typographic apostrophe U+2019. Neither was a threshold change. Docs that still
describe the FAIL are stale, not a second opinion.

Full detail, caveats and what may not be claimed: `docs/RQ1_FINDINGS_TEMPLATE.md`
for the claim structure, and `results/rq1_status.csv` for the current numbers.

---

## Running things

**Everything goes through Slurm**, including CPU-only analysis: login nodes kill
even a few GB of cached activations with a policy `SIGKILL`.

```bash
# corpus (model-independent, no GPU)
sbatch slurm/scripts/run_cpu.sh experiments/e1_0_corpus.py

# one RQ1 stage, one model
sbatch --export=ALL,MODELS=qwen2.5-7b,BATCH_SIZE=50 \
       slurm/scripts/run_gpu.sh experiments/rq1.py --only causal

# CPU analysis off the cached activations
sbatch --export=ALL,MODELS=qwen2.5-7b slurm/scripts/run_cpu.sh \
       experiments/rq1.py --only nulls geometry projections

python experiments/rq1.py --list        # stages, dependencies, GPU needs
```

> `--export=ALL,MODELS=a,b` does **not** work — sbatch splits `--export` on
> commas, so only the first model is passed. Submit one job per model.

Each research question is **one script composed of stages**. A stage declares what
it `produces`, `requires`, and whether it `needs_gpu`; completed stages are skipped
unless `--force`, so re-running an analysis never re-runs a capture.

### Verification

```bash
sbatch slurm/scripts/run_cpu.sh tests/test_invariants.py          # 22 environment invariants
RESULTS_ROOT=./results sbatch slurm/scripts/run_cpu.sh tests/verify_rq1_run.py
```

`verify_rq1_run.py` checks what the pipeline does not check itself: no train/test
leakage, layer selection on train rather than test, every direction beating its
length-only baseline, no gate verdict from a criterion with zero tests, FDR
actually applied, the steering readout calibrated, and — with `VERIFY_AGAINST` —
two results roots compared in two tiers (label-independent concepts must reproduce
**exactly**; label-dependent ones get a tolerance, because greedy generation is
only bit-reproducible at a fixed batch size).

**A full independent reproduction is part of settling an RQ**, not an optional
extra.

---

## Repo map

```
PLAN.md ENVIRONMENT.md EXPERIMENTS.md     the three-file rule (see Start here)
core/           config, pools (corpus), positions (rendering), capture,
                extract (estimators), refusal (labelling), guard, interventions
                (steering), model_io, model_meta, stages (runner), io_utils
experiments/    e1_0_corpus.py   E1.0 corpus build + freeze
                rq1.py           all of RQ1, as stages
tests/          static checks (check_names, check_shadowing), environment
                invariants, estimator unit tests, verifier + its mutation suite,
                smoke tests for read/steer positions, roles, steered generation
tools/          analysis: rq1_status, audit_independent, gate_compare,
                gate_sensitivity, asymmetry_structure, style_decompose,
                soft_refusal_probe/_split, control_learning_curve,
                adjudicate_*/score_*, make_figures, repro_diag
slurm/          run_cpu.sh, run_gpu.sh, fetch_model.sh, blank_slate*.sh;
                logs/<campaign>/ per run, logs/adhoc/, evidence/ (kept logs),
                watch/ (chain watcher)
results/        live artifacts, one dir per experiment per model
results_verify/ independent reproduction at identical config
```

`activations.pt` is a **cache, not a result** — regenerate with
`rq1.py --only extract`. Extraction is deterministic.

Superseded result archives were deleted once their content was documented **and
they had become inert** — the last two were built on a 200/200 corpus that the
drift tool's own corpus-scale guard refuses to compare against the live 500/500.
`docs/REMOVED_ARCHIVES.md` records why each was invalid, which is what the current
checks exist to prevent. The
scientific record of what changed lives in the code comments that justify each
check, `docs/REMOVED_ARCHIVES.md`, and `slurm/evidence/README.md` (the handful of
job logs kept as primary evidence, each with what it proves).

Activation caches live outside the results tree, in a scratch directory keyed by
the absolute results root — see `Config.cache_dir`. Keying matters: `results` and
`results_verify` are independent repetitions, and a shared cache would have the
second silently load the first's activations and agree with itself.
