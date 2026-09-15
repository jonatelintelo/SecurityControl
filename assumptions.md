# Assumptions and environment facts — the verification ledger

**Status:** v1, 2026-09-15. Every design assumption made in `plan.md` or `experiments.md`
lives here as a numbered entry, together with every fact about models, datasets, templates,
external code and the cluster that those documents rely on. **Nothing here is believed
until its status says so.** Entries taken from the literature or from the superseded
documents in `archive/` are recorded as claims with their origin; they are re-verified by
the procedure named in the entry, not inherited.

**How this file is used.**

- `plan.md` and `experiments.md` cite entries by ID (`L3`, `D9`, …) and assert nothing about
  models, datasets, code or the cluster themselves. If a statement in those files depends on
  something that could turn out false in practice, it must have an ID here.
- **Status values:** `unverified` · `verified` (date, how, artifact) · `refuted` (date, how,
  and what was done instead) · `superseded` (no longer needed). A verification that required
  a model forward pass names the results artifact; an offline verification names the
  script or the source read.
- **This is the only planning document that changes as verification results come in.**
  Results of *experiments* still go to `rqX_findings.md`; results of *checks* go here.
- **Origin codes:** `lit` = a cited paper; `old` = the superseded documents in `archive/`
  (facts that were verified there by code that no longer exists, so they are `unverified`
  again); `design` = our own reasoning; `published` = a config, dataset card or repository
  that can be read.
- If an entry is refuted, the *If refuted* column says what the design does instead, and
  the amendment is logged in `experiments.md` § 9 when it changes an experiment.

Groups: **L** literature claims the design depends on · **D** datasets · **M** models and
templates · **I** instruments and labels · **T** estimators · **V** intervention
primitives · **K** attacks · **N** statistics · **X** cluster and engineering · **B** compute
estimates.

---

## L — literature claims the design depends on

| ID | Claim | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| L1 | Refusal is mediated by a difference-of-means direction: ablating it bypasses refusal, adding it induces refusal | lit: Arditi et al. 2024 | § 1.3, § 1.4, E1.6 | E1.1 anchor on the AdvBench-vs-Alpaca pair; E1.6 diagonal cell for `control` | unverified | `control` is not steerable on that model; RQ2–4 use only the models where L1 holds and the failure is reported |
| L2 | Harmfulness and refusal are distinct directions and steer asymmetrically | lit: Zhao et al. 2025 | plan § 2, E1.1, E1.6 | E1.1 anchor; E1.6 G2 on the `harm`/`control` pair | unverified | the two-variable core is not replicable here; GATE 1 fails and the negative is the result |
| L3 | Harmfulness is read best at the last instruction token and refusal at the last prompt token | lit: Zhao | § 1.2, § 1.3 | E1.4 separation at both positions for both variables | unverified | positions are chosen by the E1.4 profile on train and the choice is recorded |
| L4 | Roles occupy linear directional subspaces decodable by a logistic probe with `C = 5e-3`, best at mid depth | lit: Ye et al. 2026 | § 1.3, E1.1 | E1.1 probe accuracy, C sweep, cross-corpus transfer, fidelity run at their site | unverified | probe hyperparameters from the sweep; if no linear role signal exists the role variable is not recoverable and RQ2's role edges are `undetermined` |
| L5 | Linguistic style dominates role tags in the role representation | lit: Ye et al. | plan § 2, E1.5 | E1.5 level 1 (and level 2 if run) | unverified | role interventions are built on tags only; E2.5 rungs are re-designed |
| L6 | `alpha = 1` in class-gap units is a behaviourally effective and capability-safe operating point | lit: Arditi, Zhao | § 1.4, E1.6 | E1.6 stage A profile with the capability bound | unverified | the grid is what is used; `alpha = 1` loses its special status |
| L7 | NeuroStrike's selection rule `abs(z) > 3 and w > 0` on gate/up activations, zeroed at the module output, reaches the ASR the paper reports on the roster's dense models with the utility they report | lit: Wu et al. 2026 | E3.0, E3.3, E4.3 | E3.0 reproduction | unverified | GATE 2 fails for that model; no component claim about NeuroStrike on it |
| L8 | L³ expert silencing reproduces the reported ASR on a supported MoE model | lit: te Lintelo et al. 2026 | E3.0, E3.3, E4.4 | E3.0 reproduction | unverified | as L7 |
| L9 | GateBreaker and SAFEx releases run on the roster's MoE models and produce expert sets | published: their repositories | E3.0 baselines | offline run | unverified | baselines dropped for that model; L³ vs random only |
| L10 | A released safety-head ranking (Ships / Sahara) exists and runs on at least one roster model | lit: Zhou et al. 2024 | E3.0, E3.3 | offline: locate release, run | unverified | our own head-ablation ranking, declared as ours |
| L11 | NeuroStrike's own evaluation uses Llama-Guard-3-8B and four refusal markers | published: their code | E4.0 comparability | read their evaluation script | unverified | comparability row re-specified to their actual instrument |
| L12 | Arditi's refusal-substring list has 12 markers and is published in their repository | published | § 1.5 | read the repository; commit the list with its URL | unverified | the list actually published is used |
| L13 | Attack success under greedy decoding agrees with sampled decoding within 0.05 | old (measured on one model by removed code) | E3.0 decoding check | E3.0 | unverified | every attack arm reports both decodings |
| L14 | Refusal and guard-judged harm are not complements (a large complied-but-harmless cell exists) | old (measured once) | E4.0 | contingency table per arm | unverified | if they are complements on a model, one instrument suffices there |

## D — datasets

| ID | Fact or assumption | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| D1 | AdvBench `harmful_behaviors`: ~520 harmful instructions; HF id and column names as recorded when verified | published | E1.0 | read the dataset card and a row | unverified | pool re-specified |
| D2 | JailbreakBench behaviours: 100 harmful (+100 benign) with category labels | published | E1.0 | as D1 | unverified | pool re-specified |
| D3 | SORRY-Bench: 440 base prompts (`prompt_style == "base"`) plus 20 non-base styles, each paired 1:1 with a base prompt by id; the multilingual styles are identifiable | published | E1.0, E4.2 | read card; assert pairing on load | unverified | jailbreak families re-grouped from what exists |
| D4 | SORRY-Bench base prompts are heterogeneous in severity, so that aligned models comply with a non-trivial fraction | design (from the benchmark's stated purpose) | E1.0 widening order, R1 | E1.1 stage 1: compliance rate per source on every roster model | unverified | D9's fallback |
| D5 | Alpaca standalone instructions (`input == ""`) number well over 400 after dedup | published | E1.0 | offline count | unverified | pool re-specified |
| D6 | XSTest: 250 `safe` and 200 `unsafe` items with a label column | published | E1.0 | read card | unverified | pool re-specified |
| D7 | Including XSTest-safe (benign-but-sensitive) items in the harmless pool keeps `harm` from being a topic detector | design | E1.0, plan § 8.1 | E1.2 per-source geometry: `harm` fit with vs without XSTest, and its separation on XSTest-safe vs Alpaca | unverified | topic confound reported; `harm` refit on topic-matched pairs |
| D8 | C4 `en` is accessible offline on the cluster in streaming or cached form | published / cluster | E1.0b | offline load | unverified | Dolma or Wikipedia passages |
| D9 | The harmful-and-complied cell reaches >= 50 items on the cross-model control variant, on at least four roster models | design (R1) | § 2 cell rule, E1.1 | E1.1 stage 1 2x2 on every model | unverified | ordered fallback: (1) `control_over` as the cross-model variable; (2) per-model `under` where it exists; (3) if `over` is also thin, an explicit compliance-forcing arm is *designed and pre-registered as an amendment* (e.g. benign-context prefills), reported as such and never used as the primary variable |
| D10 | StrongREJECT (~313) and HarmBench text behaviours overlap with AdvBench; after exact and near-duplicate filtering >= 150 held-out intents remain | published / design | E1.0, E2.5, E4.0 | offline filter and count | unverified | the held-out threshold in E2.5/E4.0 is lowered and the power consequence stated |
| D11 | Round-robin 400 + 400 is achievable from the pools after dedup and special-token removal | design | E1.0 | offline build | unverified | sizes reduced; the cell rule (D9) governs |
| D12 | Token-Jaccard 0.8 (dedup) and 0.5 (held-out disjointness) are strict enough that no held-out intent is a paraphrase of a fitting item | design | E1.0 | manual audit of the 20 nearest pairs across the boundary | unverified | thresholds tightened |

## M — models and templates

Roster entries M1–M6 are recorded once per roster model when verified; until then the roster
in `plan.md` § 9 is a proposal.

| ID | Fact or assumption | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| M1 | For each roster model: HF id, architecture class, layer count, `d_model`, expert count, where the shape lives in the config (nested `text_config` on some families), and the path to the decoder stack | published | everything | read config; locate the stack by search, never by assumption | unverified | — |
| M2 | Each roster model's chat template renders all four roles (system, user, assistant, tool); how a `tool` message renders (own tag vs a wrapped user turn), whether the content is quoted or JSON-wrapped | published | E1.0, E1.1, E2.5 | `apply_chat_template` offline per role | unverified | a model missing a role class is either dropped or run on three classes with the caveat in every artifact |
| M3 | Template constraints on message order (system must be first; alternation enforced) | published | E1.0 natural-slot design | offline render | unverified | design deviations recorded per model |
| M4 | Whether the template inserts a reasoning/think block before the assistant turn and how it is disabled (`enable_thinking`, harmony channels); whether `<think>` stripping is needed on outputs | published | § 1.2, § 1.5 | offline render and one generation | unverified | positions and stripping rules per model |
| M5 | MoE router semantics per model: top-k, bias correction, grouping, shared experts, and where the *actual* selection can be read | published / old | E3.5, E4.4 | read modelling code; known-answer test (a masked expert never appears in the indices on any token) | unverified | adapters written per model |
| M6 | NeuroStrike ships probe weights for the roster's dense models; where not, their `1_get_safety_neuron.py` runs with their data | published: their repository | E3.0 | list the release; run the script | unverified | model replaced or probe retrained and declared |
| M7 | L³ supports the primary MoE model natively; the second MoE model needs an adapter | published: their repository | E3.0 | read the release | unverified | as M5 |
| M8 | Template "bleed" (BPE merging the last instruction character into template markup) is at most one character and only on the `tool` role | old | § 1.2 | E1.0 rendering report | unverified | affected items excluded from exact-position analyses |
| M9 | Rendered length differs between roles by a few tokens only, so a length-only baseline is meaningful rather than degenerate | old | E1.0, E1.1 | E1.0 length report | unverified | length-matched refits become primary |
| M10 | The dtype and attention implementation used by each reference attack (bf16/fp16, eager/sdpa) are known, so equivalence tests compare like with like | published: their code | E3.0 | read their code | unverified | equivalence tolerance stated instead of bitwise |
| M11 | Each roster model, with capture hooks at every layer on the probe set, fits on the available GPU class | design | everything | smoke run | unverified | sharding or a smaller probe set |
| M12 | Multimodal-wrapper checkpoints are avoided; if one is admitted, its vision tower does not affect text-only forward passes | design | plan § 9 | offline: compare logits with and without the vision tower loaded | unverified | model dropped |

## I — instruments and labels

| ID | Fact or assumption | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| I1 | A 48-token generation budget labels refusal as well as 128 or 256 tokens (>= 95% agreement, `undetermined` < 30%) | old | § 1.5 | the rule in § 1.5 on 200 items per model | unverified | the smallest passing budget is used |
| I2 | Degeneracy thresholds (distinct-word ratio 0.35, 3-gram repeat 0.5, < 4 words) separate degenerate from genuine outputs | old | § 1.5 | calibration on 200 real generations per model, recorded | unverified | thresholds re-calibrated per model |
| I3 | The refusal-prefix rule agrees with independent model judges well enough to be the primary label (kappa >= 0.6) | old | § 1.5 | two-judge audit on a blind 300-item sheet per model | unverified | the judge becomes primary and the rule the sensitivity |
| I4 | Typographic apostrophes and Unicode variants occur in generations and defeat a naive substring rule | old | § 1.5 | count normalisation hits on the label run | unverified | normalisation dropped as unnecessary |
| I5 | Llama-Guard `unsafe` on a harmful-and-complied response corresponds to genuine compliance rather than a lecture or deflection | design | E1.1 stage 1 | manual read of 50 guard-unsafe complied items per model | unverified | guard prompt or threshold changed; audit recorded |
| I6 | Llama-Guard-3-8B runs alongside a roster model on one node within memory | cluster | E1.1, E4.0 | smoke run | unverified | guard run as a separate CPU/GPU pass over saved generations |
| I7 | The refusal-margin token pair (first refusal token vs first compliance token) is identifiable per model | design | § 1.5 | offline: inspect the first token of labelled refusals and compliances | unverified | margin readout dropped for that model |
| I8 | A frontier-model second judge is available under a fixed instruction and its id/date can be recorded | design | E4.0 | one call | unverified | agreement analysis limited to the guard vs the rule |
| I9 | Utility benchmarks (HellaSwag, RTE, WinoGrande, ARC-Challenge, OpenBookQA, CoLA, IFEval) are runnable offline with the harness we choose, and their reported numbers on unmodified roster models match published values | published | E3.0, E4.0 | one clean run per model | unverified | harness or benchmark replaced |

## T — estimators

| ID | Assumption | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| T1 | Sampling 8 content tokens per sequence for the role probe gives the same probe as 4 or 16 | design | § 1.3 | sensitivity in E1.1 | unverified | the count at which accuracy saturates |
| T2 | The residual stream is an adequate read site for the role probe (fidelity to the role paper's site) | design | § 1.3 | E1.1 fidelity check | unverified | role read at their site for validation, residual for geometry, with the caveat |
| T3 | 50 split-halves give a stable floor (its CI narrower than the smallest overlap difference of interest) | design | § 1.6 | compare 50 vs 200 halves on one model | unverified | 200 halves |
| T4 | Stratified refits (per role, per source, 50 draws) give enough directions for the spectral effective rank to be meaningful | design | E1.3 | rank stability under halving the draws | unverified | spectral half reported as descriptive only |
| T5 | Role composition is equal on both sides of the `harm` contrast by construction | design | § 1.3 | assert on the fitting set | unverified | stratified resampling |

## V — intervention primitives

| ID | Assumption | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| V1 | Registering an intervention hook after a capture hook makes the capture read the pre-intervention value | old | § 1.4 | invariant test: capture-first gives exactly zero effect | unverified | hook framework changed |
| V2 | Greedy output is bit-reproducible only at fixed batch composition, so paired contrasts must share a job | old | § 1.1 | repeat a 50-item generation at two batch sizes | unverified | composition rule relaxed |
| V3 | A harmless-prompt KL of 0.5 nats at `t_post` is inside the range where outputs stay coherent | old | § 1.6 | degeneracy rate vs KL on E1.6 stage A | unverified | headline bound moved to the largest coherent value in the swept family |
| V4 | 8 / 16 random-direction draws give a stable 95th-percentile band | design (cost) | § 1.6 | 32-draw check on one cell family | unverified | draw count raised |
| V5 | Clamp fidelity below `1e-2` of the class gap is achievable in bf16 at every layer | old | § 1.4, E2.2, E3.3 | per-arm fidelity record | unverified | tolerance stated from the measured floor, never relaxed after a verdict |
| V6 | Directional ablation leaves a residual small relative to the class gap in bf16 | old | E2.3 | per-arm residual record | unverified | reported |
| V7 | Teacher-forcing the unsteered greedy continuation is a valid reference for `t_gen+j` reads (the steered model would not have diverged far within 4 tokens) | design | § 1.2 | compare with reads on the steered model's own continuation on one arm | unverified | reads on own continuation |
| V8 | Zeroing a component's output is the attack's own suppression form (not mean-ablation) for NeuroStrike, L³ and GateBreaker | published: their code | § 1.4, E3.3 | read their code | unverified | their form used |
| V9 | Interchange-patching pairs by index do not differ systematically in length | design | E2.4 | length difference distribution of pairs | unverified | length-matched pairing |
| V10 | The steered vector's arrival at the read position is linear in `alpha`, so the D-2 correction is a subtraction | design | RQ2 | fit through origin, `R^2` reported | unverified | correction reported as a bound |

## K — attacks

| ID | Assumption | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| K1 | At least one injection rung reaches the injectable criterion on at least one roster model | design (GATE 3) | E2.5, E4.1 | E2.5 | unverified | E2.5 reports resistance; the role edge is adjudicated from E2.1/E2.1n only; RQ4 drops the family on that model |
| K2 | At least one jailbreak family lands in a common ASR band with a component attack on each model | design (GATE 3, E4.0) | E4.0 | E4.0 dose ladder | unverified | families compared on the dose-response curve, stated as unmatched |
| K3 | Encoding-style jailbreaks fail harm recognition trivially, so they serve as a calibration family | design | E4.2 | E4.5 signature | unverified | the family is a real family, not a calibration |
| K4 | NeuroStrike's layer-prefix ladder spans the ASR bands on the dense models | lit | E4.3 | E4.3 | unverified | z-threshold ladder used instead |
| K5 | SORRY-Bench non-base styles wrap the same intent (paraphrase-free), so `t_intent` is locatable inside the styled prompt | published | E4.2 | offline string containment per style; styles where it fails use `t_payload` | unverified | as stated |
| K6 | An inert-framing arm can be built for every jailbreak family with matched length | design | E4.2 | offline build | unverified | family reported without a framing control |
| K7 | GCG and PAIR are runnable within budget on one dense model | design | E4.2 optional | pilot | unverified | dropped |

## N — statistics

| ID | Assumption | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| N1 | 1000 random-direction draws give a null 95th percentile stable to within the reporting precision | design | § 1.6 | quantile vs draw count curve | unverified | draw count raised |
| N2 | Resampling by instruction accounts for the clustering of role renderings | design | § 1.6 | compare instruction-level and item-level CIs on one direction | unverified | cluster level changed |
| N3 | Wilson intervals are adequate for the injectable-criterion rates at `n >= 150` | design | E2.5 | — (standard) | verified (2026-09-15, standard result) | — |

## X — cluster and engineering (from the superseded ENVIRONMENT.md; all re-verified)

| ID | Fact | Origin | Used by | Verified by | Status | If refuted |
|---|---|---|---|---|---|---|
| X1 | Login nodes kill processes holding a few GB; all work, including CPU analysis, must run under Slurm | old | § 1.1 | observe once | unverified | — |
| X2 | `sbatch --export=ALL,VAR=a,b` splits on commas; one job per model | old | run scripts | observe once | unverified | — |
| X3 | `torch.linalg.eigvalsh` is unsupported in bf16; cast to fp32 | old | E1.2, E1.3 | one call | unverified | — |
| X4 | Left padding with an attention mask is required for position resolution in padded batches on every roster tokenizer | old | § 1.2 | assert padding side per tokenizer | unverified | right padding handled explicitly |
| X5 | CPU jobs need explicit thread caps to avoid oversubscription | old | run scripts | observe | unverified | — |
| X6 | Activation caches must be keyed by results root so two reproduction roots never share a cache | old (a real incident) | § 1.7 | code invariant | unverified | — |
| X7 | Models and datasets are available in the offline HF cache on compute nodes | cluster | everything | fetch script | unverified | — |
| X8 | Slurm GPU class, memory per node, wall-time limits | cluster | everything | read `sinfo` / policy | unverified | — |

## B — compute estimates (measured earlier by removed code; estimates only)

| ID | Estimate | Origin | Used by | Verified by | Status |
|---|---|---|---|---|---|
| B1 | ~1.5 min per generation arm at 250 items x 48 new tokens on an 8B dense model | old | planning | first RQ2 arm | unverified |
| B2 | Six multiple-choice utility benchmarks at full size: ~5 min per intervention level on an 8B model | old | E4.0 tiers | first utility run | unverified |
| B3 | Generative utility (IFEval) is the expensive tier and needs the screening subsample | old | E4.0 tiers | first run | unverified |

---

## Verification log

Chronological record of status changes: date, ID, new status, how, artifact or source.

| Date | ID | Status | How | Artifact / source |
|---|---|---|---|---|
| 2026-09-15 | N3 | verified | standard statistical result | — |
