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
| **[results/RQ1_FINDINGS.md](results/RQ1_FINDINGS.md)** | What RQ1 actually found, including every correction made along the way | a run produces new evidence |

Read them in that order. **The only numbers in `EXPERIMENTS.md` are thresholds
fixed before a run** — a measured value appearing there is a bug.

The most important thing to understand about this project: **almost every headline
claim so far has been killed or reshaped by a control.** The controls are the main
experimental apparatus, not decoration. `RQ1_FINDINGS.md` §6 lists eight
corrections where a confident result turned out to be an artifact of the
measurement.

---

## Where RQ1 stands

Every RQ1 stage in `EXPERIMENTS.md` is settled except E1.7 Level 2. In brief:

- Three variables are **representationally distinct** — recoverable against a
  1000-draw anisotropy null, length-only baselines and cluster-bootstrapped CIs,
  with a positive control that reaches the noise floor when two directions really
  are the same variable.
- They are **asymmetrically coupled but not independently manipulable**. One
  directed relation replicates exactly: `R_harm → R_control` in 44 of 44
  qualifying cells across both models.
- Gate 1's pre-registered conjunction **fails** — its two conditions disagree.
- All directions are **style-dependent**, and for harmfulness the variation is
  driven mainly by *which harmful dataset* is used.

Full detail, caveats and what may not be claimed: [RQ1_FINDINGS.md](results/RQ1_FINDINGS.md).

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
tests/          test_invariants.py, verify_rq1_run.py, smoke_generate_steered.py
tools/          analysis and maintenance: gate_compare, asymmetry_structure,
                style_decompose, readjudicate, repro_diag, clean_slurm_logs,
                prune_caches
slurm/          run_cpu.sh, run_gpu.sh, fetch_model.sh; logs + KEPT.md
results/        live artifacts, one dir per experiment per model
results_verify/ independent reproduction (activation caches pruned)
results_llama_probe/  third-model probe: Llama-3.1-8B under-refusal check
```

`activations.pt` is a **cache, not a result** — regenerate with
`rq1.py --only extract`. Extraction is deterministic.

Superseded result archives were deleted once their content was documented;
`docs/REMOVED_ARCHIVES.md` indexes what they held and why each was invalid. The
scientific record of what changed lives in `RQ1_FINDINGS.md` §6 (eight
corrections, with before/after) and `slurm/1_run_phase/KEPT.md` (job → artifact).
`EXPERIMENTS_v1_superseded.md` is the previous playbook, kept for its retractions;
it is not a source of specifications.
