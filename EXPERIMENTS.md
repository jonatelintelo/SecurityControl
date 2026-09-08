# Experiment Register — execution playbook

**Restarted 2026-09-07 from a clean slate.** All prior results are invalid and
quarantined in `results_ARCHIVED_v1_invalid/`.

## The research plan

There is **one plan**. It is the scoped plan, with the method detail from the
earlier long-form draft merged in wherever the scoped version compressed it away
— representation-extraction specifications, the attribution formalism, the
functional-role taxonomy, the models/evaluation/scope-control sections. Where the
two disagreed, the scoped version won. The keys below cite sections of that single
merged plan and are used throughout, so every experiment traces back to it.

| Key | Plan section |
|---|---|
| **PLAN-RQ*n*** | Research questions 1–7 |
| **PLAN-PEP-*n*** | *Prioritized Experimental Plan*, items 1–10 |
| **PLAN-EXTRACT** | *Representation Extraction* — per-variable extraction specs |
| **PLAN-GEOM** | *Representational Geometry* |
| **PLAN-XINT** | *Cross-Intervention Causal Matrix* |
| **PLAN-ATTR** | *Component-Level Attribution* — the `C_{i,l}^R` formalism |
| **PLAN-ROLES** | *Writers, Readers, Detectors, and Transformers* |
| **PLAN-MED** | *Causal Mediation and Rescue* |
| **PLAN-MOE** | *Dense versus MoE Security Circuits* |
| **PLAN-TAX** | *Mechanistic Attack Taxonomy* |
| **PLAN-CONC** | *Security Concentration and Functional Redundancy* |
| **PLAN-HOPE** | *Relation to Functional Operator Analysis* |
| **PLAN-RED** | *Proof-of-Concept Security Redundancy* |
| **PLAN-STRUCT** | *Structural Characterization* — `C_R`, `N_eff`, `r_eff`, `k_50`, `S_R` |
| **PLAN-CPE** | *Central Prediction Experiment* |
| **PLAN-CDRE** | *Context-Dependent Reorganization Experiment* |
| **PLAN-INF** | informal *Experiments* list — Passive / Active |
| **PLAN-SCOPE** | *Models*, *Evaluation*, *Scope Control* |

The plan does not map its experiments onto its RQs. That mapping is ours.

**Naming.** The plan's third variable appears as both `R_refusal` and
`R_control`. We use **`R_control`** throughout.

## How we work

1. One RQ at a time; one experiment at a time within it.
2. Design each experiment against its cited literature, deriving criteria and
   controls at design time.
3. Implement, run, check against its own criteria.
4. **Do not start the next until the current is settled.** Settled means
   trustworthy, not favourable — a well-powered null is a finished experiment.

Order: **RQ1 → RQ2 → RQ3 → RQ4**, then RQ5–7 and framing.

**Minimum viable core (PLAN-PEP).** The plan declares the first four stages the
minimum viable paper: recover the latent variables, build the cross-intervention
matrix, connect NeuroStrike's neurons to the representations, and perform causal
rescue. That is RQ1 + RQ2 + the NeuroStrike half of RQ3. Everything beyond is
upside, and the concentration/prediction work is named the strongest route to a
high-impact contribution.

Status: `not started` · `designing` · `implementing` · `running` · `settled` · `blocked`

---

# Run environment

| | |
|---|---|
| Dense | **both** `Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen3.5-9B`, bf16 |
| MoE | `Qwen/Qwen3.5-35B-A3B` — deferred until dense RQ1–4 settle |
| Chat template | ChatML; `enable_thinking=False` (recorded deviation; why the CoT role class is dropped) |
| Python | `/home/b6aj/jtelintelo.b6aj/miniforge3/envs/venv_causal_safety/bin/python` |
| Results | `results/<experiment>/<model_slug>/` |
| Compute | **Slurm only** — `sbatch slurm/scripts/run_experiment.sh experiments/<x>.py` |
| Seed | `SEED=0` |
| Scope | everything in this playbook is in scope; no deadline-driven cuts |

PLAN-SCOPE targets 5–7 models across ≥4 families; the prioritized plan narrows the
first pass. We run **both dense models from the start**, so every RQ1–3 result has
a second model immediately, and the MoE arm follows.

### Consequences of running two models — these are build requirements

**Model is a loop dimension, not a config constant.** Every experiment runs per
model and writes to `results/<experiment>/<model_slug>/`. Nothing may assume a
single model's shape.

**Depth is not comparable by index.** Qwen2.5-7B has **28** layers, Qwen3.5-9B
has **32**. Cross-model curves are plotted against **relative depth**
`layer / (n_layers − 1)` ∈ [0,1]; absolute layer indices are reported only
within a model.

**Directions are never compared across models.** `d = 3584` vs `4096`, different
bases — a cross-model cosine is meaningless. Cross-model comparison is of
*quantities* (AUC-vs-depth, `r_eff`, concentration, asymmetry patterns), never of
vectors.

**The two models answer different questions and both are load-bearing.**
Qwen2.5-7B is text-only and is where NeuroStrike demonstrably lands (ASR 0.727 vs
0.287), so the MVP's third stage has a model where the signal exists. Qwen3.5-9B
is the multimodal-wrapper model and the same family as the MoE arm. Agreement
between them is the answer to O-6: if results hold on the text-only model too, the
vision tower is not driving them.

---

# Standing constraints

- **Capability retention is part of any attack claim.** Every ASR is reported with
  a utility number at the same intervention.
- **Every ranking or direction needs a matched random control**, at matched size
  and matched utility.
- **Nothing is measured on the prompts used to fit it**, verified programmatically.
- **Interval estimates, not point estimates**, on anything reaching the paper.
- **Position indexing must be per-example.** `t_inst` is not index `-1`; with
  left-padding it differs per row. Every extraction records the index it used.
- **Functional roles are established causally.** PLAN-ROLES requires detector /
  transformer / writer / reader be identified "through intervention, activation
  patching, and directional contribution analyses **rather than through
  correlation alone**".
- **Provenance.** Every run writes a manifest with commit, dirty flag, config and
  Slurm job id. A number without a manifest does not exist.

### Evaluation dimensions (PLAN-SCOPE)

Every claim is reported along the four dimensions the plan names: **behavioral
security** (refusal/compliance, ASR, safety benchmarks), **utility** (standard
utility and instruction-following evaluations), **mechanistic consistency** (did
the intervention produce the predicted change in the target representation or
component?), and **cross-model generalization**.

---

# Analysis protocol

Cross-cutting decisions that apply to every experiment. Fixed here, in advance,
because each is a route to a result that looks real and is not.

### Position comparability

The three variables are read at **different token positions** — `R_harm` at
`t_inst`, `R_control` at `t_post-inst`, `R_role` over content tokens. Residual
stream is now common to all three (see E1.1), but *position* is not, and a cosine
between a direction estimated at one position and one estimated at another does
not support the claim the geometry is meant to make ("at layer 27, harmfulness is
anti-aligned with the system role" implies the two coexist at a point in the
computation).

**Protocol.** Every direction is additionally evaluated at a **common position,
`t_post-inst`**, and E1.2/E1.3 geometry is reported on that common-position set.
Paper-faithful positions are kept for validation and for anything compared to
Zhao. The role probe is token-level, so evaluating it at `t_inst` and
`t_post-inst` is free; `R_harm` at `t_post-inst` is exactly `R_control_pos`, which
we compute anyway. **Cross-position comparisons are never claimed.**

### Layer selection and multiple comparisons

With 32 layers × 3 variables × 6 concept pairs, "harmfulness is anti-aligned with
system role at layer 27" is one of ~200 comparisons. Reporting the largest is a
garden of forking paths and a reviewer will say so.

**Protocol.**
- **Report full layer curves, not selected layers.** The primary figures are
  quantity-vs-layer, so no layer is privileged by selection.
- Where a single layer must be named (steering site, attribution site), it is
  chosen on the **train split only**, by a rule fixed in advance and recorded in
  the manifest, then applied unchanged to the test split.
- Any claim about a specific layer carries **Benjamini–Hochberg FDR correction**
  across the layers tested, and the family size is stated.
- **Layer names must be ordered.** A previous run selected `mid = 23` by argmax
  while `late` was fixed at `2N/3 = 21`, so "late" preceded "mid" and a third of
  the cross-intervention matrix read *upstream* of its own intervention, producing
  exact `0.0` rows. Assert `early < mid < late` at write time.

### Replicates

`SEED=0` fixes sampling, but a single fit gives no stability estimate. Every
direction is additionally estimated **split-half** on the train split; the
split-half cosine is the noise floor against which any cross-concept cosine is
read. Bootstrap CIs elsewhere.

### Refusal labelling — three instruments, three jobs

`R_control` is the `refused vs complied` contrast, so the label is load-bearing
from E1.1 onward. The labeller is **accepted without hand-validation**, on the
condition that degenerate generations cannot enter the contrast. Each instrument
is used only for the question it can actually answer:

| Instrument | Question it answers | Used for |
|---|---|---|
| **Refusal-prefix rule** | *did the model decline?* | the `refused` / `complied` label |
| **Degeneracy gate** | *is this a real generation at all?* | exclusion |
| **Llama-Guard** | *is this response actually harmful?* | validating the harmful-and-complied cell |

**Llama-Guard is not a refusal detector.** It scores response harmfulness, which
is a different question: a model can comply with a harmless request (not a
refusal, not harmful) or refuse a harmless one (a refusal, still not harmful).
Using it as the refusal label would silently relabel both cases. It is used below
only where response harmfulness is genuinely what we need to know.

**Labels are three-way: `refused` / `complied` / `undetermined`.** The third class
is excluded from the contrast rather than folded into `complied`, because both of
the failure modes we care about collapse into a binary:

- a **degenerate** output containing refusal-like fragments reads as a *false
  refusal*;
- a **degenerate or truncated** output with no refusal marker reads as a *false
  compliance* — and this is the documented failure mode already recorded in the
  implementation notes ("a judge scores incoherence as non-refusal").

**Degeneracy gate.** After stripping `<think>` blocks, an output is
`undetermined` if it is empty, shorter than 4 words, has a distinct-token ratio
below 0.35, or repeats any 3-gram across more than half its length. These
thresholds are a starting specification, not a validated instrument: they are
**calibrated once against a sample of real generations** at the start of E1.1
(inspect the outputs each rule excludes, confirm they are genuinely degenerate and
that coherent refusals and coherent compliances both survive), and the calibration
is recorded. Adjust before the run, not after seeing the contrast.

**Truncation is not compliance.** Refusal labelling generates only ~48 tokens, so
many outputs stop mid-sentence. An output that neither declines nor visibly
answers within the budget is `undetermined`. Defaulting these to `complied` would
bias `R_control` systematically toward whatever the model happens to say first.

**Llama-Guard cross-check on the cell that matters.** For harmful prompts labelled
`complied`, Llama-Guard should judge the response unsafe. If it judges it safe,
the "compliance" is likely an evasive non-answer rather than genuine compliance,
and the item becomes `undetermined`. This protects exactly the
harmful-and-complied cell that `R_control`'s identifiability rests on.

**Reported as an artifact:** counts per label, the degeneracy-exclusion rate, and
the Llama-Guard disagreement rate on harmful-and-complied. A high exclusion rate
is itself a finding about the generation budget, not something to hide.

---

# Variable definitions

| | `R_role` | `R_harm` | `R_control` |
|---|---|---|---|
| **Definition** | "the perceived role, authority, or origin of an instruction" | "recognition of security-relevant or harmful content" | "the downstream **mechanism governing** refusal versus compliance" |
| **PLAN-EXTRACT spec** | matched examples, identical or similar content under **system, user, assistant, tool, untrusted external**; **both linear probes and direct activation-space contrasts** | matched harmful/benign **while controlling for refusal behaviour** | a general **k-dimensional subspace** `[r_1 … r_k]`, not assumed 1-D |
| **Literature estimator** | multiclass role probe | Zhao harmfulness direction | Zhao refusal direction |

### What Zhao et al. already established — verified from the paper

- Harmfulness and refusal are **encoded separately**, as distinct directions.
- Steering **harmfulness** makes harmless instructions read as harmful.
- Steering **refusal** elicits refusal *without reversing the harmfulness
  judgment* — an **asymmetry**, not a symmetric dissociation.
- **Certain jailbreaks reduce refusal signals without reversing the model's
  internal belief of harmfulness.**

Their asymmetric steering test is the method E1.6 needs and gives E1.1 a concrete
replication target. But for the `{R_harm, R_control}` pair, RQ1's separability is
**already published**, and the last bullet is already an RQ4-style stage diagnosis
for one attack family. Our contribution on that pair must come from adding
`R_role`, from the component-level mapping (RQ3), and from attack families Zhao
did not test. State this in related work rather than let a reviewer find it.

---

# RQ1 — Is LLM safety functionally decomposable?

> Are role perception, harmfulness recognition, and refusal/compliance control
> represented as **causally distinguishable** latent variables? We investigate
> their **geometry**, **dimensionality**, **emergence across layers**, and
> **causal interactions** without assuming a fixed sequential architecture.

PLAN-GEOM declares this stage **descriptive and explicitly not the main
contribution** — its purpose is to establish the latent variables the causal
analyses need. Budget accordingly.

| # | Experiment | Source | Status |
|---|---|---|---|
| **E1.1** | Recover and validate the three variables at every layer | PLAN-PEP-1, PLAN-EXTRACT | `designing` |
| **E1.2** | Pairwise geometry: **principal angles, projection measures, canonical correlations**; cosine only as the 1-D special case | PLAN-GEOM, PLAN-INF | `not started` |
| **E1.3** | Projections across prompt sets + correlations between projections | PLAN-INF | `not started` |
| **E1.4** | Dimensionality, and the **smallest dimensionality that explains the behavioural interventions** | PLAN-RQ1, PLAN-EXTRACT | `not started` |
| **E1.5** | Emergence across layers, **and persistence in subsequent layers** | PLAN-RQ1, PLAN-GEOM | `not started` |
| **E1.6** | Causal distinguishability — **go/no-go gate 1** | PLAN-RQ1 | `not started` |
| **E1.7** | Variation across **prompt categories**; role **metadata versus style**; later across model families and dense-vs-MoE | PLAN-GEOM, PLAN-EXTRACT | `not started` |
| **E1.8** | *Optional:* persona-trait and emotion directions | PLAN-INF optional | `deferred` |

**Method constraint:** *"without assuming a fixed sequential architecture."* E1.6
must not presuppose `R_role → R_harm → R_control` — that is RQ2's job.

---

## E1.1 — Recover and validate the three latent variables

**Status:** `designing` · **Source:** PLAN-PEP-1, PLAN-EXTRACT

### Read site: residual stream for all three, including role

PLAN-ATTR defines component attribution as

```
C_{i,l}^R(h) = || P_R f_{i,l}(h) ||_2 ,   f_{i,l}(h) = a_{i,l}(h) · w_{i,l}^out
P_R = R (R^T R)^{-1} R^T
```

`w^out` is a neuron's **output direction**, which lives in the **residual
stream**. `P_R` — and therefore every RQ3 attribution, including role — is only
defined if `R` is a residual-stream subspace. PLAN-XINT likewise steers
`h_l' = h_l + α r_role^(l)`, a residual-stream operation, and PLAN-GEOM computes
principal angles *between* the three representations, which requires one common
space.

**So `R_role` is estimated on the residual stream, exactly like `R_harm` and
`R_control`.** The role paper's demo hooks `post_attention_layernorm` because that
suits *their* goal of reading role identity per token; that site is a
per-token-rescaled view of the residual and is not interchangeable with it for
cosine geometry or for `P_R`. We keep it only as a **fidelity check** that we
reproduce their probe. This resolves what was previously an unexplained asymmetry
in our design.

### Two estimators for role, not one

PLAN-EXTRACT: *"The analysis will consider **both linear probes and direct
activation-space contrasts**."* So role gets both:

- **probe** — multiclass logistic over role classes (the role paper's estimator)
- **contrast** — diff-of-means between role pairs, the same estimator used for
  `R_harm` and `R_control`

The probe's `w_i − w_j` is the softmax decision boundary and should agree with the
diff-of-means contrast. Agreement means the role signal is not an artefact of the
estimator; disagreement is itself a finding.

### Role classes

PLAN-EXTRACT names **system, user, assistant, tool, untrusted external content**.
We use `system / user / tool / assistant`; CoT is dropped (`enable_thinking=False`)
and "untrusted external" is realised as the `tool` class, the injection-relevant
one.

### Harm must be separable from refusal *by construction*

PLAN-EXTRACT: *"Particular attention will be paid to examples in which a model
**recognizes harmfulness but nevertheless complies**, allowing harmfulness
representation to be separated from refusal behavior."*

This is a **data requirement**, not an analysis choice, and it is what makes
`R_harm` and `R_control` genuinely distinct rather than two views of one axis.
The crossed corpus must contain a usable number of *harmful-and-complied* items.
We label refusal behaviourally anyway, so this cell is measurable — and if it is
near-empty, `R_control` is not identifiable and the corpus must be widened until
it is populated. **The harmful × refused 2×2 is a first-class artifact, reported
before any direction is trusted.**

**Widening must not leak into RQ4.** The obvious way to populate the
harmful-and-complied cell is to add jailbreak framings — but jailbreaks are an
RQ4 attack family (E4.2), and fitting `R_control` on them would make RQ4's
"jailbreaks compromise stage X" partly circular, and would violate PLAN-CPE's
"completely disjoint attack intents". Widen in this order instead:

1. **Role framings** — already in the design; `tool`/`assistant`-framed harmful
   instructions attract lower refusal than `user`-framed ones at no cost.
2. **Milder harmful sources** — Sorry-Bench spans a wide severity range; sample
   toward the compliant end rather than adding attacks.
3. **Only if still empty:** a jailbreak family that is then **excluded from RQ4**
   and declared as such.

### `R_control` as a k-dimensional subspace

Refusal is *not* assumed one-dimensional: `R_control^(l) = [r_1 … r_k]`, with the
target being *"the smallest dimensionality required to explain the observed
behavioural interventions"*. E1.1 produces the rank-1 diff-of-means as the first
basis vector; E1.4 determines `k` behaviourally.

### What the source papers specify

| | `R_harm`, `R_control` | `R_role` |
|---|---|---|
| Source | Zhao et al. [2507.11878](https://arxiv.org/pdf/2507.11878) | Ye, Cui & Hadfield-Menell [2603.12277](https://arxiv.org/abs/2603.12277), + [role-probe demo](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/master/demo/role-probe-demo.ipynb) |
| Estimator | difference-of-means | multiclass logistic, L2, `C=5e-3`, `max_iter=2000` |
| Their read site | residual stream | `post_attention_layernorm` — **we use residual, see above** |
| Position | harm `t_inst`, refusal `t_post-inst` | every content token, tags filtered out |
| Contrast | harmful vs harmless; **refuse vs accept** | role class |
| Data | AdvBench/JBB/Sorry-Bench vs Alpaca/XSTest | C4 + Dolma3 in role tags |
| Scale | 100 per side | 150 base × 5 roles = 750 |

**Zhao's equations, verbatim:**

```
v^l_harmful = mu^l,t_inst_harmful     - mu^l,t_inst_harmless
v^l_refuse  = mu^l,t_post-inst_refuse - mu^l,t_post-inst_accept
```

**Token positions, verified against ChatML:** `t_inst` = last content token of the
instruction-bearing turn (that turn's `<|im_end|>` − 1), whatever role tag it
carries; `t_post-inst` = last token of the templated prompt.

### Our design: their estimators, our substrate

Running Zhao on instruction data and the role probe on webtext would leave every
cross-concept cosine confounded by that distribution gap — and PLAN-GEOM's
principal angles need one common substrate. So:

```
instructions × {harmful, harmless} × {system, user, tool, assistant}
```

| Variable | Estimator | Position | Contrast (other factor balanced) |
|---|---|---|---|
| `R_harm` | diff-of-means | `t_inst` | harmful vs harmless, pooled over role |
| `R_role` | multiclass probe **+** diff-of-means | content tokens | role class |
| `R_control` | diff-of-means | `t_post-inst` | **refused vs complied**, pooled over role |
| `R_control_pos` | diff-of-means | `t_post-inst` | harmful vs harmless — *control condition* |

**On behavioural labelling of `R_control`.** Not circular: it is measured at
`t_post-inst`, before any token is generated, so "the state already encodes whether
the model will refuse" is a substantive predictive claim. The narrower hazard is
that a behaviourally-fitted `R_control` is optimally predictive of `Y` **by
construction**, so it must never be compared against `R_harm` on how well it
predicts `Y`; every `R_control → Y` claim rests on **steering**.

`R_control_pos` exists because `R_harm` is the harmful/harmless contrast at
`t_inst`, so the same contrast at `t_post-inst` may be nothing but `R_harm`
transported downstream. `cos(R_harm, R_control_pos)` quantifies that.

### Paper-faithful runs, as validation

1. **Zhao replication** — the asymmetry, on their data and contrast.
2. **Role probe reproduction** — at `post_attention_layernorm`, their site.
3. **Cross-corpus transfer** — does our crossed-corpus role probe classify roles on
   C4 constant-content data, and vice versa? If ours fails, it is reading
   "instruction-ness" and must be rebuilt.

### Scale, splits, and cost

- 200 harmful + 200 harmless × 4 roles = **1,600 sequences**.
- **75/25 split by instruction, not by sequence** — one instruction's four role
  renderings must never straddle the split.
- Test split ≈ 100 instructions × 4 roles = **400 items**, ample for AUC with
  bootstrap CIs. `R_control`'s effective `n` is the smaller of the
  refused/complied cells, which is why the ≥ 50 threshold above exists.

**Layer coverage is not uniform, and the asymmetry is deliberate.** Single-
position capture is cheap (1,600 × 4,096 × 32 layers ≈ 0.8 GB fp32), so `R_harm`,
`R_control` and the contrasts are computed at **every layer**. Token-level capture
for the role probe is the memory-dominant path — every content token × every layer
— so it is bounded by sampling content tokens per sequence
(`MAX_CONTENT_TOKENS`, default 8), keeping a full-layer run near 3.5 GB. **The
role probe still covers every layer**; `LAYER_STRIDE` exists only as an escape
hatch and, if used, must be recorded in the manifest, because a strided role probe
is not comparable with full-layer curves for the other two variables in E1.5.

### Criteria

Thresholds are stated now, before the run, so they cannot be chosen to fit the
result. They are for E1.1 only — later experiments get their own at design time.

| Check | Threshold | If it fails |
|---|---|---|
| Held-out separation, best layer, each concept | **AUC ≥ 0.75 with CI excluding the matched random direction** | the variable is not recoverable; fix extraction before proceeding |
| Layer-0 separation | **AUC < 0.90** | the classifier is reading token identity, not a representation — rebuild the contrast |
| Harmful-and-complied cell | **≥ 50 items** (≥ 30 after the split), counted **after** degeneracy exclusion and the Llama-Guard cross-check | `R_control` is not identifiable; widen the corpus in the order above |
| `undetermined` rate | **< 30%** of generations | the refusal budget is too short or the labeller too brittle; raise `REFUSAL_MAX_NEW_TOKENS` before touching the contrast |
| Role probe, held-out, 4 classes | **accuracy CI excludes chance (0.25)** | role is not decodable on our substrate |
| Cross-corpus transfer (crossed → C4) | **above chance** | the probe reads "instruction-ness"; rebuild |
| Role probe vs role contrast | **cos(w_i − w_j, diff-of-means) > split-half floor** | the role signal is estimator-dependent; report and investigate |
| `cos(R_harm, R_control_pos)` | *diagnostic, no threshold* | if near 1, the late-position harm contrast is just `R_harm` transported; report it and rely on the behavioural `R_control` |

`R_control`'s AUC is reported on a **balanced** test subset, since the
refused/complied split is not 50/50 by construction and AUC on a heavily skewed
set is easy to misread.

### Produces

`results/e1_1/` — `directions.pt`, `role_probes.pt`, `direction_validation.csv`,
`role_probe_accuracy.csv`, `role_probe_transfer.csv`, `refusal_labels.csv`,
`harm_refusal_2x2.csv`, `corpus.json`, `null_band.json`, `run_manifest.json`

---

## E1.2–E1.5 — methods

| # | Method | Criterion |
|---|---|---|
| **E1.2** | **Principal angles** between subspaces, projection measures, canonical correlations; cosine as the 1-D special case | Null band: cosines between random unit vectors concentrate at 0 with sd ≈ `1/sqrt(d)`. Subspace measures get a matched random-subspace null |
| **E1.3** | Project held-out labelled prompts onto each direction; per-layer distributions per class; Pearson correlations between projections | CIs; null is the correlation between projections onto random directions |
| **E1.4** | **Stratified directions, then their spectrum**: estimate diff-of-means within each stratum (per role, per source, bootstrap), stack the unit directions, take `r_eff` of *that*. Then the behavioural criterion: **the smallest `k` whose subspace reproduces the intervention effect of the full subspace** | `r_eff ≈ 1` ⇒ one axis; `≫ 1` ⇒ subspace. The behavioural `k` is the number the paper reports |
| **E1.5** | Held-out separation and probe accuracy as a function of layer; **persistence** — how far a layer-`l` direction still separates at layers `> l` | Where each concept emerges, and whether the three emerge at different depths |

**Do not compute `r_eff` on raw activations** (that is the width of the
representation space) nor on the between-class scatter (rank-1 with two classes).
Both were tried and are wrong.

### Two circular dependencies, and how they are broken

These would block implementation if left implicit.

**E1.2 needs subspaces; subspace dimensionality comes from E1.4.** Principal
angles and canonical correlations are defined between *subspaces*, but until E1.4
returns `k` we only have rank-1 directions. **Resolution: E1.2 runs in two
passes.** Pass 1 (rank-1 cosine geometry, with the null band) runs immediately
after E1.1 and is sufficient for PLAN-INF's stated question. Pass 2 (principal
angles, canonical correlations) runs after E1.4 fixes `k`. The paper reports
pass 2; pass 1 is a checkpoint.

**E1.4's behavioural `k` needs steering; steering is built in E1.6.** "The
smallest `k` whose subspace reproduces the intervention effect of the full
subspace" is an intervention criterion. **Resolution: E1.4 also splits.** The
spectral half (stratified directions → `r_eff`) runs before E1.6 and is passive.
The behavioural half runs after E1.6's steering harness exists and reuses it.

**Revised RQ1 order:**
`E1.1 → E1.2(pass 1) → E1.3 → E1.4(spectral) → E1.5 → E1.6 → E1.4(behavioural) → E1.2(pass 2) → E1.7`

### E1.7 — metadata versus style

PLAN-EXTRACT: *"We will evaluate whether explicit role metadata and linguistic
style produce compatible or conflicting representations."* This connects directly
to the role paper's central finding that **style dominates tags**. Cross the role
*tag* with the role *style* and ask whether the role direction tracks the tag, the
style, or a mixture.

**Two levels, because the full version needs a corpus we do not have.**

*Level 1 — observed style, available now.* Each instruction carries the native
style of its source: AdvBench/JBB/Sorry-Bench are imperative harmful requests,
Alpaca is task instructions, XSTest is questions, C4 is prose. Recording `source`
alongside `role` in E1.1's corpus makes a first pass possible at zero cost:
does the role direction shift with source-style at fixed tag?

*Level 2 — controlled style, a separate corpus build.* The real test needs the
*same* content rewritten to sound like a system prompt, a user request, and tool
output, crossed with the tag. That requires a generation step (rewrite each
instruction in each style, verify the rewrite preserves content) and is its own
corpus, built when E1.7 runs.

**E1.1's obligation is only Level 1: record `source` as a factor.** Stating it as
"record style as a factor" made E1.1 depend on a corpus that does not exist.

---

## E1.6 — Causal distinguishability (go/no-go gate 1)

**Method.** Zhao's asymmetric steering, extended to 3×3. For each ordered pair
`(A, B)`: steer along `A`, read `B`'s projection downstream and record behaviour.
The asymmetries are the evidence, and they demonstrate distinguishability without
assuming any ordering.

**Controls:** random directions at matched norm; steering inside the
capability-preserving range; both signs of `α`.

**Gate, stated precisely.** The plan's criterion is that `R_role`, `R_harm` and
`R_control` *are causally distinguishable* — which requires each variable to be
distinguishable from **each** of the others, not merely that some pair separates.
Concretely, for every ordered pair `(A, B)`:

- steering `A` moves `A`'s own readout more than it moves `B`'s (a diagonal
  dominance condition), with CIs excluding the random-direction null; **and**
- at least one pair shows a genuine **asymmetry**, i.e. effect(A→B) ≠ effect(B→A)
  with non-overlapping CIs. Symmetric coupling everywhere would be consistent with
  one variable measured three ways.

Family size for the CIs is 9 ordered pairs × the layers tested; FDR-corrected per
the analysis protocol.

**Failing the gate is a result, not a failure of the project** — it says safety is
not decomposable into these three variables in this model, which contradicts the
central hypothesis and is publishable as such. What it forecloses is RQ2–RQ4 as
written.

---

# RQ2 — What causal architecture connects these variables?

> Activation patching, cross-intervention, mediation, and rescue distinguish
> sequential, parallel, and partially overlapping causal structures.

PLAN-XINT specifies three intervention *types* per variable — `I_role`, `I_harm`,
`I_control`, which **strengthen, suppress, or restore**. The matrix is therefore
3 variables × 3 intervention types × 3 readouts, plus behaviour.

| # | Experiment | Source | Status |
|---|---|---|---|
| **E2.1** | Cross-intervention matrix, measuring effects **on the other representations and on final behaviour** | PLAN-PEP-2, PLAN-XINT | `not started` |
| **E2.2** | **Injection repair** — under a *successful prompt injection*, repair only role (`h_l' = h_l + α r_role^(l)`) and test whether harm recognition, refusal representation, and safe behaviour return | PLAN-XINT | `not started` |
| **E2.3** | **Converse of E2.2** — strengthen refusal while leaving role corrupted. The difference between E2.2 and E2.3 is the evidence about causal ordering | PLAN-XINT | `not started` |
| **E2.4** | **Activation patching** | PLAN-RQ2 | `not started` |
| **E2.5** | **Mediation** — suppress *NeuroStrike's own* neurons; determine which latent variable disappears | PLAN-PEP-3, PLAN-MED | `not started` |
| **E2.6** | **Rescue** — restore only the missing subspace component, `h_l' = h_l + P_{R_control} Δh_l`, or inject an independently estimated representation; does safety return with the neurons still suppressed? | PLAN-MED | `not started` |
| **E2.7** | Adjudicate sequential / parallel / partially overlapping; test the candidate chain as a hypothesis | PLAN-PEP-2 | `not started` |
| **E2.8** | Feature-interaction: steer one vector **or toggle neurons** early, measure downstream **concept projections and neurons**, at later layers **and later tokens** | PLAN-INF | `not started` |
| **E2.8a** | Steer a benign user instruction toward the **tool-output role** — more harmful downstream? | PLAN-INF | `not started` |
| **E2.8b/c** | Remove fear / switch persona to misaligned | PLAN-INF | `blocked` (E1.8) |

**E2.2 and E2.3 are the sharpest experiments in the plan.** They convert "cross-
intervention matrix" from a table of effects into a directional argument:
repairing role restores everything downstream ⇒ role is upstream; strengthening
refusal restores behaviour but not harm recognition ⇒ refusal is downstream of
harm. PLAN-MED extends the same logic to components:
`N↓ ⇒ R_control↓`, then restore `R_control` alone and see whether `Y` returns,
evidencing `N → R_control → Y`.

**E2.7's adjudication rule must be fixed before the matrix is inspected.**

### E2.0 — E2.2/E2.3 need a working prompt injection first

E2.2 is defined "under a *successful* prompt injection", so it presupposes an
injection setup that actually flips the model. Building and validating that setup
is E4.1's job, which the dependency order places *after* RQ2. **This is a real
ordering conflict, not a technicality** — without it, E2.2 and E2.3 cannot run.

**Resolution.** Pull a minimal slice of E4.1 forward as **E2.0**: construct an
injection condition and verify it achieves a non-trivial success rate on held-out
intents. It does not need the full attack taxonomy, only one working injection.
E4.1 later extends it. E2.0 blocks E2.2/E2.3 and nothing else.

---

# RQ3 — Which components implement each security function?

> Neurons, attention components, MoE experts and routing decisions that
> **causally** detect, transform, write, or read individual security
> representations — and what NeuroStrike's and GateBreaker's components do.

### The four functional roles (PLAN-ROLES)

The plan's stated goal here is to avoid treating all "safety neurons" as
performing the same function. The distinction is **what a component consumes and
what it produces**:

| Role | Mapping | Operationalisation |
|---|---|---|
| **Detector** | `x → R_harm` | input features → a security variable. Reads *raw input*, writes a variable |
| **Transformer** | `R_harm → R_control` | one latent security variable → another. Reads *a variable*, writes *another* |
| **Writer** | `f_i(h) → R_control` | output contributes directly to the target subspace — high `C_{i,l}^R` |
| **Reader** | `R_control → Y` | consumes a representation, turns it into logits/behaviour |

A **detector** and a **transformer** differ by what is on the input side: a
detector's input alignment is with *input features*, a transformer's with *another
security variable*. Our removed code had no detector class because it only
measured alignment with the *same* concept on both sides.

### Attribution score (PLAN-ATTR)

```
C_{i,l}^R(h) = || P_R f_{i,l}(h) ||_2 ,  f_{i,l}(h) = a_{i,l}(h) · w_{i,l}^out
P_R = R (R^T R)^{-1} R^T      aggregated as  C_{i,l}^R = E_{h~D}[ C_{i,l}^R(h) ]
```

This is a **writer** score by construction, and it requires `R` to be a
residual-stream subspace — the reason role is extracted there.

| # | Experiment | Source | Status |
|---|---|---|---|
| **E3.1** | Compute `C_{i,l}^R` for all neurons, all three variables; identify role-writing, harm-writing, control-writing, and multi-representation components | PLAN-ATTR | `not started` |
| **E3.2** | Classify detector / transformer / writer / reader **causally** — intervention and patching, not correlation | PLAN-ROLES | `not started` |
| **E3.3** | Same for **attention components** | PLAN-RQ3 | `not started` |
| **E3.4** | What functions do **NeuroStrike's** components perform? *(MVP item 3)* | PLAN-RQ3, PLAN-ATTR | `not started` |
| **E3.5** | Do neurons' **input/output directions** align with our concept vectors? | PLAN-INF | `not started` |
| **E3.6** | Recover the same safety neurons by alignment alone, without activation-difference profiling | PLAN-INF | `not started` |
| **E3.7** | **Functional overlap** — components contributing to multiple security representations (Desired Result 2, "partially overlapping pathways") | PLAN-PEP-5, PLAN-ATTR | `not started` |
| **E3.8** | **Functional vs parameter sparsity** — complement `C_{i,l}^R` with function-space similarity between candidate components. NeuroStrike's neurons may correspond to *fewer* underlying functional roles than neurons | PLAN-HOPE | `not started` |

**E3.8** uses the functional-operator perspective conceptually only; the plan
explicitly does **not** require implementing the full framework. The usable idea:
two neurons can be parameter-different but function-similar, so counting neurons
overstates the number of mechanisms. This bears directly on our open puzzle that
`N_eff` runs into the thousands while Apple report a single behaviourally decisive
neuron.

**Known gap:** attention appears nowhere in the codebase (O-2).

---

# RQ4 — Do different attacks compromise different stages?

| # | Attack family | Source | Status |
|---|---|---|---|
| **E4.1** | **Prompt injection**, direct and indirect (distinct from jailbreaks) | PLAN-RQ4, PLAN-TAX | `not started` |
| **E4.2** | Adversarial **jailbreaks** | PLAN-RQ4, PLAN-TAX | `not started` |
| **E4.3** | **Representation steering** | PLAN-CPE | `not started` |
| **E4.4** | **NeuroStrike-style** neuron suppression | PLAN-RQ4, PLAN-TAX | `not started` |
| **E4.5** | GateBreaker MoE: expert silencing, routing manipulation | PLAN-RQ4, PLAN-TAX | deferred with MoE |
| **E4.6** | *Optional:* small safety-removing adaptation | PLAN-CPE | `deferred` |
| **E4.7** | **Diagnose** each failure as role corruption / failed harm recognition / failed control / routing bypass **or a combination**, plus the changes in the responsible **components** | PLAN-PEP-9, PLAN-TAX | `not started` |

Output is the taxonomy `attack → compromised security stage → behavioral
failure`, **inferred experimentally rather than assumed in advance** (PLAN-TAX).

### E4.0 — Behavioural matching (design requirement)

The plan states this three times: *"behaviorally similar safety failures"*,
*"attacks producing similar unsafe behavior"*, *"the same observable failure —
unsafe compliance"*. **Attack families must be compared at matched behavioural
outcome**, or a difference in projection signature could be attack *strength*
rather than attack *stage*. Depends on **O-1** (validated judge) and **O-4**
(defensible utility measure).

---

# Sufficiency audit

| RQ | Claim we would be entitled to | Sufficient? |
|---|---|---|
| **RQ1** | Three distinct variables with geometry, dimensionality, layer profile, causal distinguishability | **Yes.** E1.6 discharges gate 1; a well-powered null also answers RQ1 and halts the study |
| **RQ2** | The structure is sequential / parallel / partially overlapping, with mediation and rescue | **Yes** — E2.2/E2.3 make the ordering argument directional rather than correlational |
| **RQ3** | Which neurons *and attention components* causally detect/transform/write/read each function | **Conditionally.** Needs E3.3 (attention, never implemented) and causal E3.2. Otherwise MLP-only, and must be stated as such |
| **RQ4** | Behaviourally matched failures compromise different identifiable stages | **Yes, but strictly downstream** — stage labels are meaningless unless RQ1 and RQ2 succeeded |

**Dependency order**, with the two circular dependencies broken and E2.0 pulled
forward:

```
E1.1 → E1.2(pass1) → E1.3 → E1.4(spectral) → E1.5
     → E1.6 [GATE] → E1.4(behavioural) → E1.2(pass2) → E1.7
     → E2.0 (one working injection) → E2.1–E2.8
     → E3.1–E3.8
     → O-1, O-4 → E4.0 → E4.1–E4.7
```

Two things gate rather than merely precede: **E1.6** (the plan's go/no-go) and
**E2.0** (E2.2/E2.3 are undefined without a working injection).

**Llama-Guard is needed from E1.1**, earlier than O-1 implies — but only for the
harmful-and-complied cross-check, where response harmfulness is genuinely the
question being asked. O-1's validation concern applies to its use as the ASR
instrument from RQ4, not to that check.

---

# Open issues

### O-1 — The judge is an unvalidated instrument
Every ASR depends on Llama-Guard-3-8B AND a refusal-keyword rule, never checked
against human labels. Blocks E4.0. **Needed before RQ4.**

### O-2 — Attention (E3.3) is one table row and a large project
Per-head attribution across all layers, plus a role taxonomy that does not
transfer from MLP neurons — attention moves information *between positions*.
**Needed before RQ3 can be answered as specified.**

### O-3 — RESOLVED — "detector" is now defined
PLAN-ROLES defines all four roles by what a component consumes and produces; see
RQ3.

### O-4 — Utility is a heuristic, and E4.0 makes it load-bearing
A coherence heuristic, where PLAN-SCOPE requires "standard utility and
instruction-following evaluations". **Needed before RQ4.**

### O-5 — CLOSED — scope versus deadline
Decided: implement everything in this playbook; no deadline-driven cuts. Attention
attribution (O-2) therefore stays in scope for RQ3.

### O-6 — CLOSED — run both dense models
PLAN-SCOPE lists **multimodal safety** as out of scope, and both Qwen3.5
checkpoints are `*ForConditionalGeneration` with a `vision_config`, whereas Zhao's
and NeuroStrike's results are on text-only models.

Decided: **run both `Qwen2.5-7B-Instruct` (text-only) and `Qwen3.5-9B`**, rather
than choosing. This converts the concern into a control — agreement across the two
shows the vision tower is not driving the result, and disagreement localises the
effect. See *Run environment* for the build requirements this imposes (model as a
loop dimension, relative depth, no cross-model direction comparison).

### O-7 — RESOLVED by decision — refusal labelling accepted, degeneracy gated
The refusal labeller is accepted without hand-validation. The condition is that
**degenerate generations cannot enter the `refused vs complied` contrast**, since
they produce false refusals (degenerate text containing refusal-like fragments)
and false compliances (incoherent or truncated text with no refusal marker). The
protocol — three-way labelling with an `undetermined` class, a degeneracy gate,
and a Llama-Guard cross-check on the harmful-and-complied cell — is in
*Analysis protocol → Refusal labelling*. It no longer gates RQ1.

Residual risk, accepted and to be stated in the paper: the refusal-prefix rule is
unvalidated against human labels, so `R_control` inherits whatever that rule keys
on. The degeneracy gate bounds the damage but does not eliminate it.

---

# Undefined in the plan

| Quantity | Named in | Problem |
|---|---|---|
| **component diversity** | PLAN-RQ5 | Named, never defined |
| **pathway diversity** | PLAN-RQ5 | Same |
| `sim(·,·)` in `S_R` | PLAN-STRUCT | Similarity measure unspecified at both levels |
| `τ` | PLAN-CPE | "a predefined attack-success threshold" — value never fixed |

**`functional overlap` is defined** by PLAN-ATTR's "components contributing to
multiple security representations", i.e. high `C_{i,l}^R` for more than one `R`.

**Two concentration measures exist and are not the same.** PLAN-CONC defines
top-`k` concentration `κ_R(k) = Σ_{TopK} C_i^R / Σ_i C_i^R`; PLAN-STRUCT defines
`C_R = Σ_i p_i^2` with `N_eff = 1/C_R`. Report both; `κ` is the one the plan ties
directly to NeuroStrike and GateBreaker vulnerability.

---

# Deferred until RQ1–4 settle

### RQ5 — Does structure predict vulnerability?

| # | Element | Source |
|---|---|---|
| E5.1 | Concentration, effective dimensionality, causal redundancy, component diversity, pathway diversity | PLAN-RQ5 |
| E5.2 | `C_R`, `N_eff`, `r_eff`, `k_50` | PLAN-STRUCT |
| E5.3 | `κ_R(k)`, and the specific tests `κ_control ↑ ⇒ NeuroStrike vulnerability ↑` and `κ_expert ↑ ⇒ GateBreaker vulnerability ↑` | PLAN-CONC |
| E5.4 | **Freeze** the architecture; estimate only on representation/causal datasets | PLAN-CPE |
| E5.5 | Evaluate on **completely disjoint attack intents** | PLAN-CPE |
| E5.6 | `k*`, `α*`, and `{C_R, N_eff, r_eff, k_50} → {k*, α*}` | PLAN-CPE |
| E5.7 | **Leave-one-model-out** evaluation | PLAN-CPE |
| E5.8 | Compare against **simpler predictors**: parameter count, baseline refusal rate, probe accuracy, model family, architecture type, utility | PLAN-CPE |

The plan separates **parameter redundancy** from **security-function redundancy**:
a billion-parameter model may hold substantial generic redundancy while relying on
few components for a critical safety function. Note also that the role paper
already publishes a predict-before-attack result (role confusion predicts ASR
pre-generation; destyling drops ASR 61% → 10%), adjacent to RQ5.

### RQ6 — Static or context-dependent? (PLAN-CDRE)

| # | Element |
|---|---|
| E6.1 | Variants per intent: **direct, paraphrase, role, authority, jailbreak** |
| E6.2 | Per context estimate `R_harm(c)`, `R_control(c)`, `M_harm(c)`, `M_control(c)` |
| E6.3–E6.4 | `S_repr(c1,c2)`, `S_comp(c1,c2)` |
| E6.5 | **Causal transfer** `Effect(do(M_R(c1)); c2)` — the plan calls this the most important |
| E6.6 | Distinguish stable/stable vs stable/dynamic vs full reorganization |

### MoE (PLAN-MOE) — deferred with the MoE arm

Decompose `h → Router → E_j → R_security → Y` and intervene independently on
router logits, expert selection, whole expert outputs, neurons within selected
experts, attention, and the representation. Distinguishes: router selects a
safety-specific pathway; router selects topic experts with safety emerging
downstream; particular experts write refusal; experts encode harm while attention
implements refusal; several experts contribute redundant pathways.

### RQ7 — Redundancy hardening (PLAN-PEP-10, PLAN-RED)

Directional adapter `ΔW = R_control A` with `R_control` fixed and only `A`
learned, so the adapter learns *when* to activate a known subspace rather than an
unrestricted output direction. Explicitly **not** proposed as a new PEFT method —
a controlled test of `more redundancy ⇒ more robustness`.

---

# Out of scope (PLAN-SCOPE, verbatim)

Full HOPE adaptation · general-purpose PEFT framework · **multimodal safety** ·
multilingual safety · RAG-specific mechanisms · agent memory security · backdoor
attacks · secure code generation · complete theory of optimal security redundancy.

MoE experts/routing and cross-model expansion are **deferred, not excluded** — the
prioritized plan sequences dense first.

---

# Implementation notes

Claims inherited from the invalid previous attempt, **re-verified rather than
trusted**: `tests/test_invariants.py` — **17/17 confirmed, 2026-09-08**.

### Model shapes — verified from published configs

| Model | Architecture | Layers | `d` | Experts |
|---|---|---|---|---|
| `Qwen/Qwen2.5-7B-Instruct` | `Qwen2ForCausalLM` | **28** | 3584 | — |
| `Qwen/Qwen3.5-9B` | `Qwen3_5ForConditionalGeneration` | **32** | 4096 | — |
| `Qwen/Qwen3.5-35B-A3B` | `Qwen3_5MoeForConditionalGeneration` | **40** | 2048 | **256** |

**Qwen3.5 is multimodal and its config is nested.** `config.num_hidden_layers` and
`config.hidden_size` are **`None`**; the real values live under
`config.text_config`. Code doing `model.config.hidden_size` silently receives
`None`. **Requirement: resolve shape through `config.text_config` and locate the
decoder stack by search, raising if it cannot be found** — never assume
`model.model.layers`. A hook on the wrong module yields plausible-looking numbers,
the worst failure mode available. See also **O-6**.

### Traps that silently produce wrong numbers

- **Register the steering hook BEFORE the capture hook.** Hooks fire in
  registration order; capture-first reads the pre-steering value and every delta
  is exactly `0.0`. Verified.
- **`eigvalsh` crashes on bf16** — cast to float32. Verified:
  `"linalg_eigh_cpu" not implemented for 'BFloat16'`.
- **Residual norm grows with depth**, so absolute steering confounds depth with
  perturbation size; steering is **norm-relative**. *Magnitude not re-verified —
  measure on the first GPU pass.*
- **Left padding** is required for `t_post-inst`; `t_inst` still needs per-example
  indices.
- **`apply_chat_template(..., return_tensors="pt")` returns a `BatchEncoding`**,
  not a tensor. Guard with `if not torch.is_tensor(x): x = x["input_ids"]`.
- **`enable_thinking=False`** must be passed. *Why:* Qwen3 emits `<think>`, and a
  short generation would be judged on the reasoning preamble. NeuroStrike does the
  same for Qwen3.
- **Readouts must be offset-free class separation, not raw projections.** A
  `+0.078` baseline offset once made two conditions diverge for reasons unrelated
  to the intervention. AUC is rank-based and immune.
- **Keep the three measurement signals distinct**: class separation
  (representational), refusal logit margin (cheap behavioural cross-check), the
  judge (ASR only).
- **Direction signs are asserted, never inferred.** A wrong sign produces "no
  effect" — unfalsifiable rather than false. It flipped once already.
- **A judge scores incoherence as non-refusal**, which is why α=1.5 collapse shows
  high ASR even for random directions. Utility gating separates an attack from a
  broken model.
- **Thread caps must be set by the entry point**, literally, above every project
  import. Exposing a `cap_cpu_threads()` helper from a module that imports torch is
  self-defeating — the import initialises torch first. This bug was created by a
  previous version of this very file.

### Everything runs on Slurm, including smoke tests

A `FAST_DEV` run of a **0.5B** model was `SIGKILL`ed on the login node seconds
after weight loading with **119 GB free** — a node policy kill, not OOM.

### Calibration (previous attempt, not re-verified)

Relative steering: utility ≈ 0.96 at `α = 0.5`; collapse to utility 0.0 at
`α = 1.5`, where random directions also reach ASR 1.0. Treat as a starting range.

---

# Build order

**Stage 0 — corpus and positions.** No model, no GPU, fully checkable offline:
composition, source balance, disjointness, the split, and token-position
resolution across all four role tags. Everything downstream inherits these, so
they are verified first and independently.

**Stage 1 — refusal labelling, on one model, before any extraction.** Generate,
calibrate the degeneracy thresholds against real outputs, apply the Llama-Guard
cross-check, and emit the `harm × refused` 2×2 and the `undetermined` rate.

**This is a deliberate gate, not a warm-up.** Two E1.1 criteria — a
harmful-and-complied cell of ≥ 50 and an `undetermined` rate under 30% — are
knowable from this stage alone, and both can require a corpus rebuild or a longer
generation budget. Discovering either *after* a full-layer, two-model extraction
wastes the expensive part of the run.

**Stage 2 — capture and estimators.** Residual-stream capture at per-example
positions, `pre_mlp` for the fidelity check, then diff-of-means, the role probe,
AUC with bootstrap CIs, and split-half stability.

**Stage 3 — E1.1 end to end, per model**, emitting the full artifact set.

*Memory note:* Llama-Guard-3-8B is a second 8B model. Label refusals with the
target model, release it, then run the cross-check — or place them on separate
devices. Do not assume both fit alongside a 9B target.

---

# What E1.1 must do

Requirements, derived from the spec above. **This is what to build to**, not a
description of anything that exists — the implementation is written against this
list, and any code predating it is evidence at best, never a specification.

**Corpus**
- [ ] Crossed corpus `instructions × {harmful, harmless} × {system, user, tool, assistant}`, one instruction under every role
- [ ] Source-balanced sampling across all harmful and all harmless sources
- [ ] Attack intents held out and disjoint by construction, exact and near-duplicate
- [ ] **`source` recorded per instruction**, separately from the role tag — Level 1 of the metadata-vs-style test (Level 2 is a separate corpus, built at E1.7)
- [ ] 75/25 split **by instruction**, so one instruction's four renderings never straddle it
- [ ] Corpus widened, in the non-leaking order specified, if harmful-and-complied is thin

**Refusal labelling**
- [ ] Three-way `refused` / `complied` / `undetermined`; `undetermined` excluded from the contrast, never folded into `complied`
- [ ] Degeneracy gate, with its thresholds calibrated against real generations before the run and the calibration recorded
- [ ] Truncated-without-signal treated as `undetermined`
- [ ] Llama-Guard cross-check on harmful-and-complied; disagreements become `undetermined`
- [ ] Label counts, exclusion rate and disagreement rate emitted as artifacts

**Extraction**
- [ ] All directions on the **residual stream**, every layer
- [ ] Per-example `t_inst` and `t_post-inst` indices, recorded, never index `-1`
- [ ] `R_harm` @ `t_inst`, `R_control` @ `t_post-inst` (refused vs complied), `R_control_pos` @ `t_post-inst` (harmful vs harmless), `R_harm_user` (user role only)
- [ ] Role by **both** estimators — multiclass probe and diff-of-means contrast
- [ ] Every direction **also** evaluated at the common position, for position-comparable geometry
- [ ] Each direction records which class is positive
- [ ] Role probe covers every layer; any stride recorded in the manifest

**Validation**
- [ ] Held-out AUC + bootstrap CI per layer per concept, against a matched random direction
- [ ] `R_control` AUC on a **balanced** subset
- [ ] Split-half stability per direction — the noise floor for every later cosine
- [ ] Layer-0 check
- [ ] Role probe vs role contrast agreement
- [ ] Cross-corpus transfer, both directions
- [ ] Zhao replication on their own contrast, corpus and scale (100/side) — **extraction and separation only**; their *asymmetry* result needs steering and is E1.6's job, not E1.1's
- [ ] `post_attention_layernorm` fidelity check for the role probe
- [ ] `harm × refused` 2×2 emitted before any direction is trusted
- [ ] `early < mid < late` asserted wherever named layers are used

**Provenance**
- [ ] Manifest with commit, dirty flag, config, Slurm job id, model shape, corpus composition, and any source that failed to load

Later experiments will need a safety judge, a utility measure, and NeuroStrike
neuron selection. Each is written against its own spec when its experiment is
designed. Code removed from this repo is recoverable from git history as
*evidence about what happened before* — it is not a source of specifications.

---

# Data status

Availability and schemas verified against the Hub, 2026-09-08.

| Pool | Purpose | State |
|---|---|---|
| AdvBench | harmful | **OK** — 520, cols `prompt`, `target` |
| JBB-Behaviors | harmful | **OK** — config `behaviors`; columns are **capitalised** (`Goal`, not `goal`), so case-sensitive matching drops the source silently |
| Sorry-Bench | harmful | **OK** — config `default`, 9,240 |
| Alpaca | harmless | **OK** — 52,002; only `input == ""` rows, so each is standalone |
| XSTest | benign-but-**sensitive** | **OK via mirror** `natolambert/xstest-v2-copy`, split `prompts`, 450. Official `walledai/XSTest` is **gated**. Safe subset = `type` not prefixed `contrast_` |
| StrongREJECT | held-out attack eval | **OK** — 313 |
| C4 | role constant-content | **OK** — config `en` (passing `en` as a *split* is the error that made it look unavailable), streamed |
| Dolma3 | second role source | not checked; C4 suffices to start |
| persona / emotion | optional (E1.8) | not built |

### Two sampling traps, both found by running it

**Source imbalance destroys the hard negatives.** Pooling all sources and
shuffling is proportional-to-size, so a 200-prompt harmless pool came out **197
Alpaca / 3 XSTest**. XSTest is the benign-but-sensitive negative — the one thing
preventing `R_harm` collapsing into a "sensitive topic" detector.
**Requirement: sample round-robin across sources, not proportionally.** A balanced
draw gives 67/67/66 harmful and 100/100 harmless at n=200.

**A different dataset is not a disjoint dataset.** StrongREJECT overlaps AdvBench:
**3 exact + 4 near-duplicate** collisions against a 400-instruction fitting pool.
**Requirement: enforce disjointness by construction**, filtering the attack pool
against the fitting pool on exact match plus token-Jaccard ≥ 0.8, rather than
checking for overlap after the fact. Verified 0/0 collisions after filtering.
