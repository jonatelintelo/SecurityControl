# SecurityControl — The Causal Architecture of LLM Safety

Code for the ICLR project studying whether LLM safety behavior is implemented
by a structured, measurable causal architecture — rather than a single
"refusal direction" — and whether the *structure* of that architecture
(concentration, redundancy, effective rank) predicts how easy it is to break.

This file explains what each part of the code does and which research
question(s) it answers. It is meant to be read alongside the full research
plan (kept in the project conversation / project doc, not duplicated here in
full — see **Research plan summary** below for the parts that matter for
reading the code). Keep this file up to date as the code changes.

## Research plan summary

**Central hypothesis.** Safety is mediated by (at least) three interacting
latent variables:

- `R_role` — perceived role/authority/origin of an instruction (is this from
  the system, the user, or an untrusted tool/injected source?)
- `R_harm` — recognition that content is harmful/security-relevant
- `R_control` — the downstream mechanism that actually governs refusal vs.
  compliance

implemented through model components (`{attention, neurons, experts,
routing} -> {R_role, R_harm, R_control} -> Y`, where `Y` is observable
behavior). The **structural hypothesis**: high functional concentration /
low redundancy in a security function implies greater intervention
sensitivity (easier to break with a small attack).

**Research questions** (referenced throughout the code as `RQ1`-`RQ7`):

| RQ | Question | Where it's addressed |
|----|----------|----------------------|
| RQ1 | Is safety functionally decomposable into `R_role`/`R_harm`/`R_control`? | `phases/phase1_geometry.py` |
| RQ2 | What causal architecture (sequential/parallel/overlapping) connects them? | `phases/phase2_causal_structure.py` |
| RQ3 | Which components (neurons, attention, experts, routing) implement each function? | `phases/phase3_components.py` (neurons only — see Scope below) |
| RQ4 | Do different attacks compromise different stages of the architecture? | `phases/phase5_context_and_attacks.py` (Part B) |
| RQ5 | Does architecture structure (`C_R`, `N_eff`, `r_eff`, `k_50`) predict attack budget (`k*`, `alpha*`)? — **the central predictive hypothesis** | `phases/phase3_components.py` (metrics) + `phases/phase4_architecture_prediction.py` (prediction test) |
| RQ6 | Is the architecture static or context-dependent? | `phases/phase5_context_and_attacks.py` (Part A) |
| RQ7 | Does experimentally increasing redundancy increase robustness? (optional) | not implemented yet |

**Structural quantities** (defined precisely in the research plan, computed in
`core/attribution.py` and `core/metrics.py`):

- `C_R` — functional concentration (`sum(p_i^2)` over normalized per-component contributions)
- `N_eff = 1/C_R` — effective component count
- `r_eff` — effective functional rank (participation ratio of a component Gram matrix's eigenvalues)
- `k_50` — causal redundancy: smallest number of top-ranked components whose ablation cuts the security function to ≤50% of baseline
- `k*` / `alpha*` — attack budget: smallest neuron-ablation count / steering magnitude that reaches an attack-success threshold `tau` on a *topically disjoint* attack-prompt pool

## Repo layout

```
core/           shared library code — everything phase-specific scripts import
phases/         one script per research phase (phase1..phase6), each with run(cfg)
run_phase.py    CLI entry point: `python run_phase.py <1-6>`
slurm/          Slurm sbatch scripts + historical job logs
results/        phase outputs (gitignored) — CSV/JSON/*.pt per phase, one subdir each
requirements.txt
```

Design principle: **no giant end-to-end script.** Each phase is a short,
readable script that (a) loads the previous phase's frozen outputs from
`results/<phase>/`, (b) calls into `core/` for all real logic, (c) saves its
own outputs for the next phase. `core/` has no dependency on `phases/`, so
every phase reuses the exact same extraction/hooking/steering/metric code —
there is deliberately only one implementation of "extract a diff-of-means
direction" or "ablate neurons and measure the effect," used by every phase
that needs it.

## `core/` — shared library

| File | What it does |
|---|---|
| `config.py` | `Config` dataclass + `load_config()`. Everything is driven by environment variables (`MODEL_ID`, `FAST_DEV`, `SEED`, `MAX_NEW_TOKENS`, `ATTACK_SUCCESS_THRESHOLD`, `TRAIN_FRACTION`, `RESULTS_ROOT`). `FAST_DEV=1` shrinks dataset sizes, sweep grids, and generation length for a fast CPU/dev smoke test — same code path, smaller numbers, not a separate mode. |
| `data.py` | All prompt datasets, curated offline (no network dataset download — HPC compute nodes may not have internet beyond the HF hub). `ContrastivePair`/`MessageSpec` are declarative (system/user/assistant-prefix specs); `materialize_pair(tokenizer, pair)` renders them through the model's own chat template so the same dataset works for any chat-tuned HF causal LM. Defines `ROLE_PAIRS`, `HARM_PAIRS`, `CONTROL_PAIRS`, `PERSONA_PAIRS` (optional axis), `CONTEXT_INTENTS` (RQ6 direct/paraphrase/roleplay/authority/jailbreak variants), `ATTACK_PROBE_PROMPTS` (RQ5's disjoint attack pool), `ATTACK_STEERING_SIGN` (see note below), and `check_disjoint_topics()` which asserts the three harmful-topic pools never overlap. |
| `model_io.py` | `load_model()`, `build_prompt()` (chat-template rendering, see thinking-mode note below), `refusal_logit_margin()` (behavioral proxy: `logit("I") - logit("Sure")` right after the prompt), `generate_text()`, `is_refusal_text()` (keyword-based refusal classifier used for ASR). |
| `hooks.py` | `HookEngine` — raw PyTorch forward hooks for residual-stream and MLP-neuron activation capture on any `model.model.layers[i].mlp.{gate,up,down}_proj`-shaped decoder (Qwen/Llama/Mistral-style; **not** MoE — see Scope). Plus two batched capture helpers: `capture_last_token_residuals()` and `capture_mlp_neuron_activations()` (padding-mask-aware). |
| `subspaces.py` | `SubspaceEngine` — direction/subspace math: `extract_direction` (diff-of-means), `extract_subspace` (multi-pair SVD basis, not currently used since each concept is 1D), `get_orthogonal_projector`, `cosine_similarity`, `canonical_subspace_angles`, `probe_validation_accuracy` (held-out sign-classification accuracy — the "probe accuracy" baseline referenced in RQ5). |
| `attribution.py` | `AttributionEngine` — maps directions onto neurons. `compute_static_weight_alignment` (pure weight-geometry, no activations), `compute_neuron_attributions` (activation-weighted causal attribution — the "official" ranking used everywhere), `compute_activation_diff_ranking` (NeuroStrike-style: pure activation contrast, no weights), `compute_architectural_quantities` (`C_R`, `N_eff`, `r_eff`). |
| `interventions.py` | `InterventionEngine` — the three causal interventions: `steer_subspace` (additive residual steering), `ablate_neurons` (NeuroStrike-style zeroing at `down_proj`), `install_causal_rescue` (re-inject a direction at the prefill boundary without restoring the ablated neurons). |
| `metrics.py` | Sweep/measurement helpers built on the above: `component_ablation_curve`/`find_k50` (RQ5's `A_R(k)`/`k_50`), `attack_success_rate`/`find_min_k_for_asr`/`find_min_alpha_for_asr` (RQ5's `k*`/`alpha*`), `component_set_overlap` (Jaccard, used for neuron-ranking comparisons and RQ6's `S_comp`), `simple_correlation` (Pearson+Spearman for the prediction test). |
| `io_utils.py` | `get_logger` (stderr + per-phase `.log` file), `save_json/load_json`, `save_torch/load_torch`, `save_df/load_df`, `require_phase_output` (fails with an actionable message — "run phase X first" — if an upstream phase hasn't produced its output yet). |

### Two things every reader should know before touching this code

1. **Attack-steering sign is not uniform across concepts.** Each direction is
   `diff-of-means(positive, negative)`, and "positive" was chosen per-concept
   for narrative clarity (role: privileged/legitimate; harm: harmful content;
   control: refusal) — not consistently as "the safe class." So `+alpha`
   does *not* uniformly mean "attack" — see `core.data.ATTACK_STEERING_SIGN`
   and its docstring before adding any new steering experiment.
2. **Thinking-mode must stay disabled.** `Qwen/Qwen3.5-9B` (and other
   Qwen3-family models) auto-insert a `<think>...</think>` block after the
   prompt. `core.model_io.build_prompt` passes `enable_thinking=False` to
   `apply_chat_template` for exactly this reason — without it, every
   behavioral measurement (`refusal_logit_margin`, `is_refusal_text`/ASR)
   silently measures the model's reasoning preamble instead of its actual
   answer. This was found the hard way during smoke testing (see Status
   below) — don't remove it, and if you add a new place that calls
   `apply_chat_template` directly instead of going through `build_prompt`,
   it needs the same treatment.

## `phases/` — one script per research phase

Every phase module exposes `run(cfg)` and is runnable standalone
(`python phases/phase1_geometry.py`) or via the CLI (`python run_phase.py
1`). Each phase's module docstring lists its exact inputs/outputs — this
table is the short version:

| Phase | RQ(s) | Reads | Writes | What it does |
|---|---|---|---|---|
| 1. `phase1_geometry` | RQ1 | — | `directions.pt`, `selected_layers.json`, geometry/probe-accuracy/projection CSVs | Extracts per-layer `R_role`/`R_harm`/`R_control` directions, validates each with held-out probe accuracy, computes the full pairwise cosine-similarity geometry across all 32 layers, and projects a labeled probe set onto every direction. Auto-selects `early`/`mid`/`late` reference layers (`mid` = argmax \|cos(R_harm, R_control)\|) that every later phase reuses. |
| 2. `phase2_causal_structure` | RQ2 | phase 1 | `cross_intervention_matrix.csv` | Cross-intervention matrix: steer each concept's direction at a source layer, read the representational effect (cosine shift) and behavioral effect (refusal-margin shift) at a downstream target layer. Evidence for sequential vs. parallel vs. overlapping causal structure. |
| 3. `phase3_components` | RQ3, RQ5 | phase 1 | `architecture_metrics.csv`, `neuron_rankings.pt`, `neuron_ranking_overlap.csv`, `neuron_functional_classification.csv`, `mediation_rescue_results.json` | For each (layer, concept): three independent neuron rankings (causal attribution / static weight alignment / NeuroStrike-style activation diff) + their top-k overlap ("can we recover the same neurons without activation-difference profiling?"); the frozen `C_R`/`N_eff`/`r_eff`/`k_50` architecture metrics that phase 4 predicts from; functional classification of top neurons by concept; and a mediation+rescue experiment (ablate each concept's neurons at the mid layer, see which `R` disappears, restore `R_control` without restoring the neurons, check if behavior returns). |
| 4. `phase4_architecture_prediction` | RQ5 (central prediction) | phase 1, phase 3 | `prediction_table.csv`, `prediction_correlations.csv`, `k_star_curves.csv`, `alpha_star_curves.csv` | Freezes phase 3's architecture metrics, then measures `k*`/`alpha*` (attack budget) on `ATTACK_PROBE_PROMPTS` — a topically disjoint harmful-intent pool never used to estimate the architecture. Correlates architecture metrics against attack budget across all (layer, concept) rows. **Single-model scope**: this is a within-model correlation across (layer, concept) configurations, not leave-one-model-out across models (needs ≥2 models — see Scope). |
| 5. `phase5_context_and_attacks` | RQ6, RQ4 | phase 1, phase 3 | `context_stability.csv`, `component_stability.csv`, `causal_transfer_matrix.csv`, `attack_stage_diagnosis.csv` | Two RQs sharing one phase because they're the same machinery applied to different prompt *contexts*. Part A (RQ6): for each of 10 intents rendered under {direct, paraphrase, roleplay, authority, jailbreak}, representation-level stability (`S_repr`), component-level stability (`S_comp`), and — the important one — causal transfer (`do(mechanism(c1)); eval(c2)`: ablate the neurons found under context c1, measure the effect on behavior under context c2). Part B (RQ4): relative to a direct-context, no-intervention baseline, measures how far 5 conditions (roleplay/authority/jailbreak framing, representation steering, neuron ablation) push `R_role`/`R_harm`/`R_control` and behavioral ASR — i.e. which stage of the architecture each attack family actually compromises. |
| 6. `phase6_feature_interactions` | (qualitative, supports RQ2) | phase 1 | `role_to_harm_interaction.csv`, `persona_interaction.csv` + sample generations | The informal "how do features interact" experiments: (1) does steering a benign prompt toward the injected/tool-output role direction make it read as more harmful downstream? (2) does pushing persona toward an unconstrained/misaligned framing change whether a harmful prompt is still recognized as harmful / still refused? The "remove fear" case study from the informal write-up is **deferred**, not implemented — needs an emotions probe (optional per the plan) that doesn't exist yet. |

## How to run

Everything is environment-variable driven (`core/config.py`):

```bash
# Real run (Slurm GPU job — see slurm/scripts/run_phase.sh)
MODEL_ID=Qwen/Qwen3.5-9B python run_phase.py 1
sbatch slurm/scripts/run_phase.sh 1     # then 2, 3, 4, 5, 6 in order — each reads the previous phase's results/

# Fast CPU/GPU smoke test — shrinks datasets/grids/generation length, same code path
FAST_DEV=1 MODEL_ID=Qwen/Qwen3.5-9B python run_phase.py 1
```

Phases must be run in order 1→6 the first time; `require_phase_output()`
fails immediately with an actionable message if you try to run a phase
before its dependency.

## Status (as of the last full smoke test)

All 6 phases have been run end-to-end against the real `Qwen/Qwen3.5-9B` on
a GPU Slurm node with `FAST_DEV=1`. Two real bugs were found and fixed
during that process:

1. Thinking-mode contamination (see note above) — was making every
   behavioral measurement (ASR, refusal margin) degenerate.
2. A hook-registration-order bug in phase 2 that silently zeroed the
   representational-effect columns whenever `source_layer == target_layer`
   (fixed by registering the steering hook before the residual-capture
   hook — see the comment at that call site).

Post-fix, results look sane: held-out probe accuracies 0.90–1.0, a
self-consistent cross-intervention matrix (each concept's steering aligns
most with its own direction, as expected), and genuine refusal text
reaching the ASR classifier. Under `FAST_DEV`'s deliberately tiny
intervention budget the model refused every disjoint attack prompt
(`ASR=0` throughout) — expected for a small budget on a well-aligned 9B
model, not a bug, but it means a **full-scale run (`FAST_DEV` unset)** is
still needed to actually locate non-trivial `k*`/`alpha*` values and get a
meaningful RQ5 prediction-correlation result.

## Scope — what this pipeline does *not* yet do

Deliberately out of scope for the current single-dense-model pipeline
(agreed direction: build the pipeline model-agnostic, but skip
MoE/expert/routing code paths entirely for now rather than stub them out):

- **MoE / expert / routing components** (RQ3, RQ4, RQ5's "one dense + one
  MoE model" requirement, GateBreaker-style analysis). `core/hooks.py` only
  instruments dense `mlp.{gate,up,down}_proj`-style layers.
- **Attention components.** RQ3 names attention alongside neurons; only
  neurons are implemented.
- **Leave-one-model-out evaluation** (RQ5) — needs ≥2 models. The
  `prediction_table.csv` row schema (`layer, concept, metrics..., k_star,
  alpha_star`) is deliberately `model`-ready so this is a matter of adding
  rows from a second model, not a redesign.
- **RQ7** (redundancy-hardening intervention) — optional per the plan, not
  started.
- **Emotions probe** / "remove fear" case study — optional per the plan,
  deferred (see phase 6).
- **Visualization/plotting** — everything currently lands as CSV/JSON; no
  figure-generation code exists yet.
- Dataset scale is modest (offline curated: ~20 harm pairs, 15 control, 12
  role, 10 context intents, 15 disjoint attack prompts) — enough to validate
  the pipeline, thin for final statistical claims.
