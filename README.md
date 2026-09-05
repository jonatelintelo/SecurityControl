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
| RQ3 | Which components (neurons, attention, experts, routing) implement each function? Incl. "what functions do NeuroStrike's components perform?" → `neurostrike_neuron_functions.csv` classifies *their* selected neurons into role/harm/control | `phases/phase3_components.py` (neurons only — see Scope below) |
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
| `subspaces.py` | `SubspaceEngine` — direction/subspace math: `extract_direction` (diff-of-means), `get_orthogonal_projector`, `cosine_similarity`, `probe_validation_accuracy` (held-out sign-classification accuracy — the "probe accuracy" baseline referenced in RQ5). |
| `attribution.py` | `AttributionEngine` — maps directions onto neurons. `compute_static_weight_alignment` (pure weight-geometry, no activations), `compute_neuron_attributions` (activation-weighted causal attribution — the "official" ranking used everywhere), `compute_activation_contrast_ranking` (unsupervised activation contrast — an ablation of NeuroStrike's probe, *not* their method), `compute_architectural_quantities` (`C_R`, `N_eff`, `r_eff`). |
| `neurostrike.py` | Faithful port of NeuroStrike's white-box method (NDSS 2026) from `../NeuroStrike-Neuron-Level-Attacks-on-Aligned-LLMs/white_box/`: hooks `gate_proj`/`up_proj` outputs, max-pools over sequence, trains their logistic safety probe, selects neurons at \|z\|>3 & w>0, and prunes at their prune site. Adds `rank_by_probe_weight` so their signal can also drive a graded top-k budget. Deviations from the reference are documented inline. |
| `judge.py` | `AttackJudge` — Llama-Guard-3-8B + refusal-keyword AND-rule, matching NeuroStrike's ASR definition. Lazily loaded; keyword check short-circuits first so sweeps don't pay for the guard on obvious refusals. `USE_LLAMA_GUARD=0` falls back to keyword-only for dev runs. |
| `interventions.py` | `InterventionEngine` — the three causal interventions: `steer_subspace` (additive residual steering), `ablate_neurons` (zeroing at `down_proj` input), `install_causal_rescue` (re-inject a direction at the prefill boundary without restoring the ablated neurons). |
| `metrics.py` | Sweep/measurement helpers built on the above: `component_ablation_curve`/`find_k50` (RQ5's `A_R(k)`/`k_50`), `attack_success_rate`/`find_min_k_for_asr`/`find_min_alpha_for_asr`/`find_min_layer_prefix_for_asr` (RQ5's `k*`/`alpha*` and NeuroStrike's prefix budget), `component_set_overlap` (Jaccard, for neuron-ranking comparisons and RQ6's `S_comp`), `simple_correlation`. |
| `io_utils.py` | `get_logger` (stderr + per-phase `.log` file), `save_json/load_json`, `save_torch/load_torch`, `save_df/load_df`, `require_phase_output` (fails with an actionable message — "run phase X first" — if an upstream phase hasn't produced its output yet). |

### Two things every reader should know before touching this code

1. **Attack-steering sign is not uniform across concepts.** Each direction is
   `diff-of-means(positive, negative)`, and "positive" was chosen per-concept
   for narrative clarity (role: privileged/legitimate; harm: harmful content;
   control: refusal) — not consistently as "the safe class." So `+alpha`
   does *not* uniformly mean "attack" — see `core.data.ATTACK_STEERING_SIGN`
   and its docstring before adding any new steering experiment.
2. **Two different neuron bases coexist, deliberately.** `core/attribution.py`
   treats one neuron as the post-SiLU product `silu(gate_i)*up_i` (the
   `down_proj` input); `core/neurostrike.py` follows the reference and treats
   `gate_proj` and `up_proj` outputs as separate neuron sets. The *index space
   is shared* (`down_proj` input `i` == `silu(gate_i)*up_i`), which is what
   makes their top-k sets and ours directly comparable in phase 3 — but do not
   assume the two modules mean the same thing by "a neuron."
3. **Three measurement signals, not interchangeable.**
   - `A_R(k)` (and therefore `k_50`) is the **projection onto that concept's
     own direction** — `make_projection_score_fn`. It must be per-security-
     function: scoring an `R_role` ablation by the refusal margin would measure
     the effect on `R_control`, a different quantity, which is what the code
     used to do.
   - `refusal_logit_margin` ("I" vs "Sure") is recorded at the same grid points
     as a cheap behavioral cross-check (`k_50_behavioral`), never as `A_R`.
   - `AttackJudge` (Llama-Guard) defines reported ASR.
   `validate_margin_vs_judge` measures the proxy and the guard at identical
   steering points and writes `proxy_validation.csv`, so their agreement is
   reported rather than assumed.
4. **Thinking-mode must stay disabled.** `Qwen/Qwen3.5-9B` (and other
   Qwen3-family models) auto-insert a `<think>...</think>` block after the
   prompt. `core.model_io.build_prompt` passes `enable_thinking=False` to
   `apply_chat_template` for exactly this reason — without it, every
   behavioral measurement (`refusal_logit_margin`, `is_refusal_text`/ASR)
   silently measures the model's reasoning preamble instead of its actual
   answer. This was found the hard way during smoke testing (see Status
   below) — don't remove it, and if you add a new place that calls
   `apply_chat_template` directly instead of going through `build_prompt`,
   it needs the same treatment. (NeuroStrike does the same for Qwen3-family
   models — their `construct_prompt` passes `enable_thinking=False` — so this
   is consistent with the reference, not a divergence from it.)

## `phases/` — one script per research phase

Every phase module exposes `run(cfg)` and is runnable standalone
(`python phases/phase1_geometry.py`) or via the CLI (`python run_phase.py
1`). Each phase's module docstring lists its exact inputs/outputs — this
table is the short version:

| Phase | RQ(s) | Reads | Writes | What it does |
|---|---|---|---|---|
| 1. `phase1_geometry` | RQ1 | — | `directions.pt`, `selected_layers.json`, `concept_dimensionality.csv`, geometry/probe-accuracy/projection CSVs | Extracts per-layer `R_role`/`R_harm`/`R_control` directions, validates each with held-out probe accuracy, computes the full pairwise cosine-similarity geometry across all layers, and projects a labeled probe set onto every direction. Also reports each concept's **dimensionality** (participation-ratio rank of the difference spectrum) — RQ1 asks about dimensionality explicitly, and a 1-D diff-of-means direction cannot answer it; `r_eff_spectrum` ≈ 1 means the concept genuinely is a single axis. Auto-selects `early`/`mid`/`late` reference layers reused by later phases. |
| 2. `phase2_causal_structure` | RQ2 | phase 1 | `cross_intervention_matrix.csv` | Cross-intervention matrix: steer each concept's direction at a source layer, read the representational effect (cosine shift) and behavioral effect (refusal-margin shift) at a downstream target layer. Evidence for sequential vs. parallel vs. overlapping causal structure. |
| 3. `phase3_components` | RQ3, RQ5 | phase 1 | `architecture_metrics.csv`, `neuron_rankings.pt`, `neuron_ranking_overlap.csv`, `neuron_functional_classification.csv`, **`neurostrike_neuron_functions.csv`**, `ablation_curves.csv`, `mediation_rescue_results.json` | For each (layer, concept): three independent neuron rankings (causal attribution / static weight alignment / NeuroStrike-style activation diff) + their top-k overlap ("can we recover the same neurons without activation-difference profiling?"); the frozen `C_R`/`N_eff`/`r_eff`/`k_50` architecture metrics that phase 4 predicts from; functional classification of top neurons by concept; and a mediation+rescue experiment (ablate each concept's neurons at the mid layer, see which `R` disappears, restore `R_control` without restoring the neurons, check if behavior returns). |
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

### Scaling the study up

Scaling should require only (a) these environment variables and (b) larger
prompt pools in `core/data.py` — not edits to phase code.

| Variable | Default (full run) | What it controls |
|---|---|---|
| `K_GRID` | `1,5,...,2000,4000` | single-layer top-k neuron budget for `k*`/`k_50` |
| `ALPHA_GRID` | `0.5,...,16,24` | steering magnitudes for `alpha*` (sign applied per concept) |
| `LAYER_PREFIX_FRACTIONS` | `0.25,0.5,0.75,1.0` | NeuroStrike's layer-prefix attack depth |
| `LAYER_STRIDE` | unset (3 reference layers) | set to `1` to analyze **every** layer — the cheapest way to go from 9 to ~96 rows in phase 4's prediction test |
| `MAX_NEW_TOKENS` | 512 | generation length for ASR judging |
| `USE_LLAMA_GUARD` | 1 | 0 = keyword-only judging (dev only; not reportable as ASR) |
| `ATTACK_SUCCESS_THRESHOLD` | 0.5 | `tau` |
| `NEUROSTRIKE_PROBE_EPOCHS` / `NEUROSTRIKE_Z_THRESHOLD` | 5000 / 3.0 | reference probe hyperparameters |
| `GRAM_TOP_K` | 256 | neurons entering the `r_eff` Gram matrix |

**Known undersizing:** NeuroStrike trains its probe on thousands of prompts;
our curated pool is ~40. A logistic probe over 18944 features on 40 samples is
badly underdetermined, so `neurostrike_weights.pt` should not be trusted until
the `core/data.py` pools are scaled up. Phase 3 logs a warning to this effect
on every run.

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
reaching the ASR classifier.

### RQ1 result: none of the three variables is a clean 1-D axis

`concept_dimensionality.csv` (participation-ratio rank of each concept's
difference spectrum):

| concept | `r_eff_spectrum` | top-1 variance share |
|---|---|---|
| role | 1.59 | 0.75 |
| control | 2.27 | 0.60 |
| harm | 2.50 | 0.53 |

The 1-D diff-of-means direction used downstream captures only ~53–75% of the
variance, so `R_harm` and `R_control` are meaningfully multi-dimensional —
direct support for the plan's premise that refusal "may be mediated by
multiple directions rather than a single universal axis." **Caveat:** under
`FAST_DEV` these are estimated from only 4–6 pairs, which caps `r_eff` at
`n_pairs - 1`; the true values may be higher. Re-measure at full dataset size
before quoting. Implication: `C_R`/`N_eff`/`k_50` are currently computed
against a 1-D projector and therefore understate the real structure.

### Capability constraint: most apparent "attacks" were model collapse

Adding capability retention to the α sweep (FAST_DEV, Qwen3.5-9B) changed the
conclusions again — averaged over all (layer, concept) cells:

| \|α\| | ASR (concept dir) | utility | ASR (random dir) | utility |
|---|---|---|---|---|
| 0.1 | 0.000 | 1.000 | 0.000 | 1.000 |
| 0.5 | 0.089 | **0.956** | 0.000 | 0.978 |
| 1.5 | 0.511 | **0.133** | 0.267 | 0.467 |

At \|α\|=1.5 the "ASR 0.5–0.8" that earlier runs reported is **model collapse**:
only 13% of benign prompts are still answered coherently. The judge was scoring
degenerate output as non-refusal. An earlier conclusion here — that `R_control`
steering at layer 21 was a real, direction-specific vulnerability — does **not**
survive this control and was collapse-driven.

What does survive: at \|α\|=0.5, where utility is ~0.96 and the model is
intact, the concept direction yields ASR 0.089 while a random direction yields
**exactly 0.0**, and the random direction produces **no viable attack at any
magnitude tested**. The single clean, capability-preserving result is
`R_control` at layer 10 (α\*=0.5).

Practical consequence: **α\* must be read jointly with `utility`**. An α\* whose
grid point has low utility is not an attack budget. `viable_attack` encodes this.

### Earlier run, before the capability constraint (superseded, kept for the record)

Once norm-relative steering and the random-direction control were added, the
earlier "layer 10 is vulnerable" result **partly survived and partly did not**:

| condition | concept direction | random direction | verdict |
|---|---|---|---|
| L10 `R_control` | ASR 0.8 at \|α\|=**0.5** | needs \|α\|=1.5 | **real** — 3x more efficient than random |
| L21 `R_control` | ASR 0.8 at \|α\|=1.5 | 0.0 everywhere | **real** — direction-specific |
| L10 `R_role` | 0.4 at \|α\|=1.5 | **0.8** at 1.5 | **not real** — random does better |
| L10 `R_harm` | 0.0 everywhere | **0.8** at 1.5 | **not real** — random does better |

Three conclusions, none of which were visible before the controls existed:
1. Only `R_control` steering has a **direction-specific** effect. `R_role` and
   `R_harm` steering is matched or beaten by a random direction at the same
   magnitude, so it is not evidence of a role/harm attack pathway.
2. The layer-10-vs-21 gap **shrank but did not vanish** under norm-relative
   steering (α* 0.5 vs 1.5). The norm confound was inflating it, not inventing it.
3. **Large-magnitude steering breaks the model generically** — a random
   direction at \|α\|=1.5 reaches ASR 0.8. So α* is only meaningful *relative to
   the random baseline at the same magnitude*; an absolute α* that random
   matches is not a vulnerability.

**Open caveat:** at destructive magnitudes the guard may be scoring incoherent
output as non-refusal rather than as a genuine jailbreak. A coherence/utility
check on generations at high α is needed before reporting these as ASR.

### Positive control: passes for steering, fails for ablation

The single most important gate before any RQ5 number is meaningful is
whether we can break safety *at all* — otherwise "robust model" and "broken
measurement" are indistinguishable. Current status on `Qwen3.5-9B`
(`FAST_DEV`, 5 disjoint attack prompts, Llama-Guard judging):

- **Representation steering: control passes.** Steering `R_control` at
  layer 10 with `alpha=-12` reaches `ASR=1.0`. It discriminates, too —
  `R_control` at layer 21 and `R_role`/`R_harm` at both layers stay at 0.0.
  This is the first non-degenerate attack budget observed, and it validates
  the whole chain (steer -> generate -> guard -> unsafe verdict). Note
  `alpha=12` was the largest magnitude in the dev grid, so `alpha*` is
  censored: `<=12` for that cell, `>12` for the others.
- **Neuron ablation: control fails.** `ASR=0.0` everywhere, including
  NeuroStrike's own parameterization at full depth — 5485 neurons pruned
  across all 32 layers, versus their reported 76.9% ASR. Almost certainly
  data starvation rather than model robustness: their probe trains on
  thousands of prompts, this run used 12, and a logistic probe over 18944
  features on 12 samples is meaningless, so near-random neurons are being
  pruned. Consistent with that, `*_vs_neurostrike` neuron overlap is ~0.0002
  (uninterpretable, not a finding), while our internal comparisons show real
  structure (`causal_vs_static` 0.26, `causal_vs_activation_contrast` 0.21).

**Do not report `k*` or any neuron-level RQ5 result until the ablation
positive control passes** (i.e. until scaling the `core/data.py` pools
reproduces a non-zero ASR at NeuroStrike's budget).

## Prompt design: why it is factorial

`R_role`/`R_harm`/`R_control` are estimated from a **crossed factorial design**
(`core.data.factorial_pairs`), not from three separate templates. Each direction
is the main effect of one factor with the other two balanced:

| factor | positive | negative |
|---|---|---|
| role | untrusted tool-output channel | trusted operator channel |
| harm | harmful request | topic-matched benign request |
| control | forced assistant prefix `"I"` | forced prefix `"Sure"` |

This exists because the legacy per-concept sets carried **systematic bias**, not
just noise — and more prompts would only have made the wrong estimate more
precise:

- `ROLE_PAIRS` appended the injection to the negative side only: a **−16 token**
  imbalance in *every* pair, so `R_role` was partly a "longer / adversarial text
  present" detector. Factorial role framings are **exactly length-matched**
  (16 tokens each); measured imbalance is now 0.
- The three concepts used structurally different templates (system prompt
  present/absent, forced assistant token present/absent), so cross-concept
  cosine geometry — the core RQ1 evidence — could reflect template differences
  rather than relationships between security variables.
- Several benign counterparts leaked safety-salient words (`own`, `legitimate`,
  `professional`) that appeared on one side only.

The aggressive injection strings ("SYSTEM OVERRIDE", "ignore previous
instructions") are deliberately **not** used to estimate `R_role`; they are
attack conditions in phase 5. Using them as the estimator would define `R_role`
as an adversarial-text detector rather than the perceived-origin variable.

Set `USE_FACTORIAL_DESIGN=0` to fall back to the legacy sets for comparison.

## Methodological controls (read before interpreting any number)

These exist because without them the headline quantities are not interpretable.
None is optional.

| Control | Guards against | Where |
|---|---|---|
| **Random-ablation** control on `A_R(k)` | `A_R(k)` is *guaranteed* to fall for the ranked condition — neurons are ranked by their projection onto `R`, then `R` is measured. Only the ranked-vs-random gap shows the ranking carries information. | `component_ablation_curve` (`ablation` column); `k_50_random_control` |
| **Random-direction** control on `alpha*` | distinguishes "steering *R* breaks safety" from "any perturbation this large breaks safety" | `random_direction_like`; `direction_type` column |
| **Random-ranking** control on `k*` | same, for the neuron-budget attack | `ranking_source="random"`; `k_star_random_control` |
| **Norm-relative steering** | residual norms grow ~8x from layer 10 to 21, so a fixed absolute `alpha` is a far larger *relative* perturbation early than late. Absolute steering makes "early layers are more vulnerable" unfalsifiable — it's what a norm artifact looks like. | `steer_subspace(relative=True)`, `STEER_RELATIVE=1` (default) |
| **`R_control` transfer test** | `R_control` differs only in a forced final token, so it could be token identity rather than a control variable; same-construction held-out validation cannot detect this. Projects *unforced* prompts instead. | `control_transfer_test.csv` — 0.5 means chance, i.e. artifact |
| **Baseline ASR / positive control** | distinguishes "model is robust" from "our attack pipeline cannot break anything" | `baseline_asr`, NeuroStrike prefix curve |
| **Capability retention** | an intervention that destroys the model is not an attack. At large α a random direction reaches high "ASR" because output degenerates and the judge scores incoherence as non-refusal. `k*`/`α*` therefore require `utility >= MIN_UTILITY`; `viable_attack` marks grid points meeting both conditions. | `core/utility.py`, `utility` column |
| **Bootstrap CIs / Wilson intervals** | ASR is a binomial estimate over a finite prompt set and the RQ5 correlation runs over few non-independent rows; bare point estimates overstate precision. (4/5 successes ⇒ 95% CI [0.38, 0.96].) | `correlation_with_ci`, `wilson_interval` |

**`alpha` means different things in the two steering modes.** Relative (default):
a fraction of the residual norm, grid near 1.0. Absolute (`STEER_RELATIVE=0`):
a raw magnitude, grid an order of magnitude larger. `core/config.py` picks the
matching default grid; overriding `ALPHA_GRID` without matching the mode
silently produces either no-ops or destructive interventions.

## Known measurement caveats

- `w_{i,R}` is a **geometric proxy** (mean |activation| x weight-projection onto
  `R`), not an ablation-derived causal contribution. The research plan calls it
  "the causal contribution of component i" — either the attribution becomes
  genuinely causal, or the paper's wording should say proxy.
- `C_R`/`N_eff`/`k_50` are computed against a **1-D projector**, but
  `concept_dimensionality.csv` shows `R_harm`/`R_control` are ~2.3–2.5
  dimensional, so these understate the real structure.
- With one model, RQ5 is a **within-model correlation** over non-independent
  (layer, concept) rows, not leave-one-model-out generalization.

## Next steps toward paper-ready results

### The binding constraint: phase 4 compute

Per-phase wall time from the last full `FAST_DEV` run (Qwen3.5-9B, 1 GPU):

| phase | wall time |
|---|---|
| 1 / 2 / 5 / 6 | 18s / 10s / 95s / 22s |
| 3 | 101s |
| **4** | **72.5 min — 96% of total** |

Scaling phase 4 to full size multiplies that by roughly **576x**
(36x grid points: 96 rows x 51 budget points; 2x generations per point;
8x generation length) — about **29 GPU-days**. "Just run everything at full
scale" is therefore not an option, and phase 4 needs an efficiency pass
*before* it is launched, not after.

### Track A — run full-scale phases 1, 2, 3, 5, 6 now

All of these are seconds-to-minutes even at full size, and they carry four of
the paper's five contributions independently of how RQ5 resolves:

| output | contributes |
|---|---|
| phase 1 geometry + `concept_dimensionality.csv` | RQ1 |
| phase 2 cross-intervention matrix | RQ2 |
| phase 3 rankings, NeuroStrike comparison, `neurostrike_neuron_functions.csv` | RQ3 |
| phase 3 `architecture_metrics.csv` (C_R, N_eff, r_eff, k_50 + random control) | RQ5 predictors |
| phase 5 stability, causal transfer, attack taxonomy | RQ6, RQ4 |

Full-scale phase 3 additionally **de-risks phase 4**: its k-grid runs to 4000
with the random-ablation control, so it establishes whether `k_50` is findable
at all. If `A_R` never halves even at k=4000, phase 4's entire neuron arm is
known-futile before spending GPU-weeks on it.

Suggested invocation (see the scaling table above for the variables):

```bash
LAYER_STRIDE=1 sbatch slurm/scripts/run_phase.sh 1   # then 2, 3, 5, 6
```

### Track B — make phase 4 affordable before running it

In leverage order:

1. **Batch generation.** `metrics.attack_success_rate` currently loops one
   prompt at a time through `generate()`. Batching is ~8x and costs nothing
   scientifically — the single biggest win, and pure implementation.
2. **Early termination.** If ASR at the maximum budget is below `tau`, skip the
   cell's sweep entirely. In FAST_DEV only 2/9 cells ever reached `tau`, so this
   is ~3x on its own.
3. **Binary search** over the budget grid rather than a linear sweep — ASR is
   approximately monotone in budget, so 11 points collapse to ~4 (~2.8x).
4. **Subsample the control arms.** The NeuroStrike and random ranking sources do
   not need all 96 rows; a layer subset suffices to establish the contrast.

Combined, these bring ~29 GPU-days to roughly **10 hours**.

### Still missing for "paper-ready" specifically

- **No plotting code exists.** Everything lands as CSV. Figures are a
  deliverable, and this is the actual gap between "results" and "paper-ready".
- **No run provenance.** `results/` is wiped on each run with no config or
  git-hash snapshot, so a figure cannot be traced back to the run that produced
  it. Cheap to add, and needed for camera-ready reproducibility.
- **Second model (MoE)** remains the difference between a within-model
  correlation and the leave-one-model-out test RQ5 actually specifies.

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
