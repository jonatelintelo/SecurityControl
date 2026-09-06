# SecurityControl — The Causal Architecture of LLM Safety

Research code for the ICLR project asking whether LLM safety is implemented by a
**structured, measurable causal architecture** — rather than a single "refusal
direction" — and whether the *structure* of that architecture predicts how
easily it can be broken.

**Thesis under test:** safety decomposes into interacting latent variables
(`R_role`, `R_harm`, `R_control`); different attacks compromise different parts
of that architecture; and its concentration/redundancy predicts the intervention
budget needed to disable it.

---

## Start here (new co-author, 10 minutes)

1. Read **Status at a glance** below — what is established, open, and retracted.
2. Read **Five things you must know before touching the code** — these are
   non-obvious invariants that have already caused wrong results when violated.
3. Skim the **Glossary** for any column name you meet in `results/`.
4. Run `python tools/report_key_numbers.py` to re-derive every quoted number
   from `results/` yourself. Do not trust numbers in prose, including this file's
   — they have gone stale repeatedly. The tool is the source of truth.

The single most important thing to understand: **almost every headline claim in
this project so far has been killed by a control or by more data.** The
controls are not decoration; they are the main experimental apparatus. See
*Revision history*.

---

## Status at a glance

Pipeline runs end-to-end on `Qwen/Qwen3.5-9B` (6/6 phases, no errors). All
current numbers are **dev-scale unless noted** (`FAST_DEV=1`: 3–4 intents,
5 attack prompts, 3 layers, truncated sweep grids).

### Established

| Claim | Evidence | Confidence |
|---|---|---|
| **Components reorganize across contexts; representations do not** | `S_repr` 0.957/0.954 vs `S_comp` 0.491/0.538 against a matched-n within-context floor of 0.750/0.757. Gaps **+0.259 [0.187, 0.322]** (harm) and **+0.219 [0.138, 0.292]** (control); both CIs exclude 0. **Full-scale run.** | **strongest result** |
| The three variables are separately decodable | held-out probe accuracy 1.00 / 0.984 / 1.00 under the factorial design | good |
| `R_control` is a control variable, not token identity | transfers to *unforced* prompts at 0.783 mean / 0.938 best vs 0.5 chance | good |
| Neuron ranking carries real information | ablating top-50 ranked leaves `A_R` at 0.89–0.95; **50 random neurons leave it at ~1.000** | good |
| `R_control` is the most read/write integrated | transformer-neuron counts: control 39.0, harm 27.3, role 14.7 (chance ≈ 1.9), monotone across 3 layers | suggestive |

### Open — no evidence either way

- **RQ5, the central hypothesis, is untested.** Every `k_50`, `k*` and
  `prediction_correlations` entry is NaN at dev scale. The sweep grids stop at
  k=50, where `A_R` has only fallen to ~0.89. Not a negative result — an
  unrun one.
- **NeuroStrike connection uninterpretable.** Overlap ~0.0007; their probe had
  12 training prompts against thousands in the reference.
- **Neuron-level attacks unvalidated.** NeuroStrike's own full-depth attack
  (5485 neurons, all 32 layers) yields ASR 0.0 against their reported 76.9%,
  almost certainly because the probe is data-starved. Until this positive
  control passes, no neuron-level RQ5 number is reportable.

### Stale — needs re-run

`results/phase4_architecture_prediction/` was produced **before** the current
phase 3, so its `k*` sweeps used superseded neuron rankings. Note the
provenance tool reports `CONSISTENT` here only because every phase ran with a
dirty working tree, where the commit hash cannot distinguish code states —
**check the timestamps, not just the commit.**

---

## Five things you must know before touching the code

1. **Attack-steering sign is design-dependent.** Always call
   `data.attack_steering_sign(concept, cfg.use_factorial_design)`; never hardcode.
   The factorial design puts *untrusted* on role's positive side while the legacy
   pairs put *trusted* there, so the attack sign flips. Getting this wrong is
   silent and makes "no attack effect" unfalsifiable rather than false — it
   already happened once.
2. **`alpha` means different things in the two steering modes.** Relative
   (default): a *fraction of the residual norm*, grid near 1.0. Absolute: a raw
   magnitude, grid an order of magnitude larger. Residual norms grow ~8x with
   depth, so absolute steering confounds depth with perturbation size.
   `core/config.py` picks the matching grid; overriding `ALPHA_GRID` without
   matching the mode silently produces no-ops or destructive interventions.
3. **Three measurement signals, not interchangeable.** `A_R`/`k_50` use the
   class **separation** (`make_separation_score_fn`) — offset-free, positive at
   baseline by construction. `refusal_logit_margin` is a cheap behavioral
   cross-check recorded at the same grid points, never `A_R` itself.
   `AttackJudge` (Llama-Guard) defines reported ASR.
4. **An attack that destroys the model is not an attack.** `k*`/`alpha*` require
   `utility >= MIN_UTILITY`; `viable_attack` marks grid points meeting both
   conditions. At |α|=1.5 a *random* direction reaches high ASR simply because
   output degenerates and the judge scores incoherence as non-refusal.
5. **Thinking-mode must stay disabled.** Qwen3-family models emit `<think>`;
   with short generations the judge scores the reasoning preamble instead of the
   answer. `build_prompt` passes `enable_thinking=False`. NeuroStrike does the
   same for Qwen3, so this matches the reference rather than diverging from it.

---

## How to run

Everything is environment-variable driven (`core/config.py`); phases must run in
order 1→6 the first time, and `require_phase_output()` fails with an actionable
message otherwise.

```bash
# Full scale (Slurm GPU)
sbatch slurm/scripts/run_phase.sh 1     # then 2, 3, 4, 5, 6

# Fast smoke test — same code path, smaller data/grids
FAST_DEV=1 python run_phase.py 1

# Re-derive every number quoted in this file
python tools/report_key_numbers.py
```

### Scaling knobs

Scaling should require only these variables plus larger prompt pools in
`core/data.py` — never edits to phase code.

| Variable | Default | Controls |
|---|---|---|
| `LAYER_STRIDE` | unset (3 layers) | `1` analyzes **every** layer — cheapest way from 9 to ~96 rows for RQ5 |
| `K_GRID` | `1,5,…,2000,4000` | neuron-ablation budget for `k*`/`k_50` |
| `ALPHA_GRID` | `0.05…2.0` (relative) | steering magnitudes; **must match `STEER_RELATIVE`** |
| `LAYER_PREFIX_FRACTIONS` | `0.25,0.5,0.75,1.0` | NeuroStrike's layer-prefix attack depth |
| `MAX_NEW_TOKENS` | 512 | generation length for judging |
| `USE_LLAMA_GUARD` | 1 | 0 = keyword-only (dev only; not reportable as ASR) |
| `MIN_UTILITY` / `ATTACK_SUCCESS_THRESHOLD` | 0.5 / 0.5 | capability floor and `tau` |

---

## Repo map

```
core/           shared library — all real logic lives here
phases/         one script per phase, each exposing run(cfg)
tools/          report_key_numbers.py — re-derives every quoted number
run_phase.py    CLI: python run_phase.py <1-6>
results/        per-phase outputs + run_manifest.json (gitignored)
```

**Design principle: no giant end-to-end script.** Each phase loads the previous
phase's frozen outputs, calls into `core/`, and saves its own. `core/` never
imports `phases/`, so there is exactly one implementation of "extract a
direction" or "ablate and measure," shared by everything.

| `core/` file | Role |
|---|---|
| `config.py` | All tunables, from env vars. `FAST_DEV=1` shrinks data/grids — same code path, not a separate mode. |
| `data.py` | Prompt datasets, curated offline. **Factorial design** (`factorial_pairs`) is the default estimator; legacy per-concept pairs kept for comparison. Also `check_disjoint_topics()`, which enforces that architecture-measurement and attack pools never share a topic. |
| `model_io.py` | Model loading, chat templating, refusal-margin proxy, generation. |
| `hooks.py` | `HookEngine` — residual/MLP capture for Llama-style decoders (**not** MoE). |
| `subspaces.py` | Direction math, probe validation, dimensionality (`spectrum_effective_rank`). |
| `attribution.py` | Neuron attribution. Output side (`down_proj` cols) **and** input side (`gate`/`up` rows), plus reader/writer/transformer classification. |
| `neurostrike.py` | Faithful port of NeuroStrike's white-box probe (NDSS 2026), from the sibling repo. Deviations documented inline. |
| `judge.py` | Llama-Guard-3-8B + refusal-keyword AND-rule, matching their ASR definition. |
| `interventions.py` | Steering (absolute/relative), neuron ablation, causal rescue. |
| `utility.py` | Capability retention — the "did I break the model?" check. |
| `metrics.py` | `A_R`/`k_50`, ASR sweeps, budget search, stability metrics, bootstrap CIs. |
| `io_utils.py` | Logging, IO, `write_run_manifest` (provenance). |

---

## Research questions → phases

| RQ | Question | Phase | Status |
|---|---|---|---|
| RQ1 | Is safety functionally decomposable? | 1 | established (dev-scale) |
| RQ2 | What causal architecture connects the variables? | 2 | measured; no activation patching yet |
| RQ3 | Which components implement each function? | 3 | neurons only (read+write); no attention/MoE |
| RQ4 | Do different attacks hit different stages? | 5B | measured |
| RQ5 | **Does structure predict vulnerability?** | 3 + 4 | **untested** |
| RQ6 | Is the architecture context-dependent? | 5A | **established, full scale** |
| RQ7 | Does added redundancy increase robustness? | — | not implemented (optional) |

The pipeline's spine, and why phases are ordered this way:

| Step | Question | Outputs |
|---|---|---|
| 1 | Do the variables exist and separate? | `probe_accuracy`, `concept_dimensionality`, `geometry_cosine`, `control_transfer_test` |
| 2 | How do they causally connect? | `cross_intervention_matrix` |
| 3 | Which components implement them? | neuron rankings, `neurostrike_neuron_functions`, `neuron_read_write_roles`, mediation/rescue |
| 4 | How is each function structured? | `C_R`, `N_eff`, `r_eff`, `k_50` — **RQ5 predictors** |
| 5 | How hard is it actually to break? | `k*`, `alpha*` — **RQ5 outcome** |
| 6 | Does 4 predict 5? | `prediction_correlations` — **the thesis** |
| 7 | Does it change with context? | `context_stability`, `component_stability`, `causal_transfer_matrix` |
| 8 | Do different attacks hit different stages? | `attack_stage_diagnosis` |

Steps 4 and 5 use **deliberately disjoint prompt pools** — architecture measured
and frozen on one set, attack budget on topically disjoint intents. That
separation is what makes step 6 a *prediction* rather than a retrospective
correlation, and is why `check_disjoint_topics()` exists.

---

## Controls, and the question each answers

Without these the headline quantities are uninterpretable. None is optional.

| Control | Question it answers | Current answer |
|---|---|---|
| Random ablation | *Would ablating **any** k neurons do that?* | **No** — ranked leaves `A_R` at 0.89–0.95 at k=50, random at ~1.000 |
| Random direction | *Would **any** perturbation this large break safety?* | At \|α\|=1.5 **yes** (so those aren't attacks); at 0.5 no |
| Random ranking | Same, for the neuron-budget attack | Unanswerable — all `k*` NaN at dev grid |
| Norm-relative steering | *Is "early layers are vulnerable" a norm artifact?* | Partly — the L10/L21 gap shrank but survived |
| `R_control` transfer test | *Is `R_control` real, or just the token "I"?* | **Real** — 0.783/0.938 vs 0.5 chance |
| Capability retention | *Did safety fail, or did I destroy the model?* | Decisive — utility 0.911 at α=0.5 vs **0.178** at α=1.5 |
| Within-context split-half | *Do components reorganize, or is the ranking just noisy?* | **Reorganize** — floor 0.75, across-context 0.49–0.54, CIs exclude 0 |
| Matched-n comparison | *Is the floor measured with the same data as the contrast?* | Now yes; before, harm's gap was inflated 6x by the mismatch |
| Bootstrap / Wilson CIs | *How much is noise?* | Often most of it — ASR 4/5 has 95% CI [0.38, 0.96] |

---

## Prompt design: why factorial

`R_role`/`R_harm`/`R_control` come from a **crossed factorial design**
(`data.factorial_pairs`), each direction a main effect with the other two
factors balanced. The legacy per-concept sets carried *systematic bias*, which
more data would only have made more precise:

- `ROLE_PAIRS` appended the injection to the negative side only — a **−16 token**
  imbalance in every pair, so `R_role` was partly a "long/adversarial text"
  detector. Factorial framings are **exactly length-matched** (16 tokens each).
- The three concepts used structurally different templates, so cross-concept
  geometry could reflect template differences rather than real relationships.
- Several benign counterparts leaked safety-salient words (`own`, `legitimate`).

Aggressive injection strings are deliberately **not** used to estimate `R_role`;
they are attack conditions in phase 5. Using them as the estimator would define
`R_role` as an adversarial-text detector rather than a perceived-origin variable.
`USE_FACTORIAL_DESIGN=0` restores the legacy sets for comparison.

---

## Glossary

`k_50` — **redundancy.** Rank neurons by contribution to `R`, ablate the top-k,
measure `A_R(k)` = fraction of the security function remaining. `k_50` is the
smallest k where `A_R ≤ 0.5`. Small = fragile/concentrated; large = redundant.

`C_R` = `Σ pᵢ²` — **concentration** (uniform over ~19k neurons would be 5.3e-5;
ours is 2e-4…3e-3). `N_eff = 1/C_R` — effective component count. `r_eff` —
participation ratio of the neuron Gram eigenvalues: how many *independent*
mechanisms versus how many components merely fire.

`k*` / `alpha*` — smallest ablation budget / steering magnitude reaching
ASR ≥ τ **with capability intact**.

`S_repr` / `S_comp` — representation- and component-level stability across
contexts. `reorganization_gap` = within-context floor − across-context, both at
matched n.

**Two different `r_eff`s exist.** Phase 3's `effective_functional_rank_r_eff` is
component-level (neuron Gram matrix). Phase 1's `r_eff_spectrum` is
representation-level (dimensionality of the direction itself). Same formula,
different matrix.

**`r_eff_spectrum` is capped at `n_pairs − 1`.** With 4/4/9 training pairs the
ceilings are 3/3/8, so the current values (role 2.03, harm 1.76, control 3.03)
are **not comparable across concepts** — control simply had more headroom. Do
not quote the ordering until sample sizes are equal or the ceiling is slack.

---

## Known limitations

- `w_{i,R}` is a **geometric proxy** (activation × weight-projection), not an
  ablation-derived causal contribution. The plan calls it "the causal
  contribution of component i" — either the attribution becomes causal, or the
  paper says proxy.
- `C_R`/`N_eff`/`k_50` are computed against a **1-D projector**, while
  `concept_dimensionality.csv` suggests the variables are not 1-D.
- With one model, RQ5 is a **within-model** correlation over non-independent
  (layer, concept) rows — not the leave-one-model-out test the plan specifies.
- Effect sizes have **shrunk with every measurement fix** (the single-neuron
  ablation effect went 11.5% → 6.3% → ~2%). The *direction* of the effect has
  been robust; the magnitude was inflated by artifacts. Treat unreplicated
  magnitudes with suspicion.

## Not implemented

- **MoE / experts / routing** — `hooks.py` only instruments dense MLPs. Needed
  for GateBreaker-style analysis and the plan's "one dense + one MoE" baseline.
- **Attention components** — RQ3 names them; only neurons are done.
- **Activation patching** — RQ2 names it; we do additive steering only. Lives in
  phase 2 (~10 s to re-run), so cheap to add later.
- **Leave-one-model-out** (RQ5) — needs ≥2 models. The prediction-table schema is
  already `model`-ready.
- **RQ7** redundancy hardening — optional in the plan.
- **Plotting** — everything is CSV; no figure code exists. This is the real gap
  between "results" and "paper-ready."

---

## Next steps

**The binding constraint is phase 4.** Per-phase dev-scale wall time: phases 1/2/5/6
are 10–95 s, phase 3 ~3 min, **phase 4 ~72 min (96% of total)**. Scaling phase 4
naively is ~576× that — roughly **29 GPU-days**.

**Track A — run full-scale phases 1, 2, 3, 5, 6 now.** Minutes each, and they
carry four of five contributions. Full-scale phase 3 also *de-risks* phase 4: its
k-grid runs to 4000, so it establishes whether `k_50` is findable at all before
GPU-weeks are spent.

**Track B — make phase 4 affordable first.** In leverage order: batch generation
(`attack_success_rate` currently loops one prompt at a time; ~8×), early
termination when ASR at max budget < τ (~3×), binary search over the budget grid
(~2.8×), and subsampling the control arms. Together: ~29 days → ~10 hours.

---

## Revision history — conclusions the controls overturned

Retractions, not results. Do not quote the superseded numbers. **Six of eight
headline claims died to a control or to more data** — the strongest argument for
keeping the controls in the loop.

| # | Claim made | What overturned it |
|---|---|---|
| 1 | ASR 1.0 everywhere, even at k=1 | **Thinking-mode contamination** — generations never reached the answer |
| 2 | Steering at `source == target` layer has exactly zero effect | **Hook-ordering bug** — capture registered before steering |
| 3 | "Layer 10 is uniquely vulnerable" | **Norm confound** — norms grow ~8× with depth; gap shrank but survived |
| 4 | `R_role` steering is not an attack pathway | **Sign bug** — we had been steering the *defensive* direction |
| 5 | `R_control` @ L21 is a real vulnerability | **Capability retention** — utility 0.133; that was model destruction |
| 6 | `R_harm` is the most multi-dimensional variable | **Ceiling artifact** — `r_eff` capped at `n_pairs−1` (3/3/8) |
| 7 | "`R_control` reorganizes, `R_harm` does not" | **Full-scale run** — harm's gap moved 0.041 → 0.259 (6×); both reorganize equally |
| 8 | `k_50` = 50 and 1 for `R_role` | **Offset bug** — `A_R` divided by a near-zero baseline (+0.078), exploding the ratio. Now uses class separation; all `k_50` correctly NaN at k≤50 |
