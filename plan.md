# The Causal Architecture of LLM Safety — research plan

**Status:** draft v1, 2026-09-15. Merged from `main.tex` (long-form draft, Aug 2026) and
`notes_from_mail.tex` (scoped draft), restricted to RQ1–RQ4, and checked against the
literature as of September 2026. This document is the decision record for *what* we are
doing and *why*. It contains no results.

**How the documents split (decision).** Three files, three kinds of claim:

| File | Holds | Changes when |
|---|---|---|
| `plan.md` (this file) | Motivation, hypotheses, RQs, related work and novelty, contributions, the experiment programme at design level (what each experiment must establish, its manipulation, readout, controls, and what each outcome means), gates, scope | the science changes |
| `experiments.md` | The exact operational specification of every experiment named here: datasets and sizes, positions, estimators, parameter values, sweep grids, decision thresholds, artifacts. **No results, ever.** | an experiment's setup changes |
| `rq1_findings.md` … `rq4_findings.md` | Results, per RQ, against the criteria fixed in `experiments.md` | a run produces evidence |
| `assumptions.md` | Every design assumption and every fact about models, templates, datasets, external code and the cluster that the two files above rely on, each with its origin, where it is used, how it is verified, and a status. **The only planning document that changes as checks come in.** | a check is run |

The split is kept because it is what stopped results from contaminating specifications
last time. Two rules make it work: **a number that was measured on a model never appears
in `plan.md` or `experiments.md`**, and **neither file asserts anything about a model,
dataset, template, external code base or the cluster** — such statements are ledger
entries in `assumptions.md`, cited by ID, and are `unverified` until a check says
otherwise. Where this file gives a reason that rests on such a statement, the ID follows
it in parentheses.

---

## 1. Thesis in one paragraph

LLM safety behaviour is not a single "refusal mechanism". It is produced by at least
three separable latent variables — the perceived **role/origin** of an instruction, the
recognised **harmfulness** of its content, and the downstream **control** decision to
refuse or comply — connected by a causal structure that can be mapped by intervention,
implemented by identifiable sparse computational components (neurons, attention heads,
MoE experts and routes), and compromised at *different* stages by *different* attacks that
produce the *same* observable failure. We establish the variables (RQ1), their causal
structure (RQ2), their implementation (RQ3), and use the resulting architecture as a
diagnostic frame for prompt injection, jailbreaks, neuron-level and expert-level attacks
(RQ4).

---

## 2. Motivation

Recent work provides complementary but disconnected observations about how safety is
represented and implemented inside aligned LLMs:

1. **Refusal is steerable through low-dimensional activation-space directions**
   (Arditi et al. 2024), although later work shows refusal can be mediated by several
   directions or a multi-dimensional "concept cone" rather than one universal axis
   (Wollschläger et al. 2025).
2. **Harmfulness and refusal are represented separately** (Zhao et al. 2025): steering
   the harmfulness direction makes benign requests read as harmful, whereas steering the
   refusal direction elicits refusal without changing the harmfulness judgement, and
   some jailbreaks suppress the refusal signal while leaving harmfulness recognition
   intact.
3. **Safety depends on sparse components.** Roughly 3–5% of neurons or ranks are
   safety-critical (Wei et al. 2024; Chen et al. 2024); NeuroStrike (Wu et al., NDSS 2026)
   disables alignment by pruning under 0.6% of neurons in targeted layers; single MLP
   neurons can gate refusal (2605.08513); single attention heads can carry a large share
   of safety behaviour (Zhou et al. 2024). In MoE models safety concentrates in a small
   set of experts reachable through routing (SAFEx, GateBreaker, L³/Large Language
   Lobotomy, RouteHijack, RASET).
4. **Prompt injection is role confusion** (Ye, Cui, Hadfield-Menell, ICML 2026): models
   internally represent *who is speaking*, that representation follows linguistic style
   more than role tags, and the degree of confusion predicts injection success before any
   token is generated.

These findings describe pieces of one phenomenon: safety arises from an internal
*computational architecture*, not from a single guard. What is missing is a unified causal
account connecting

```
semantic security variables  <->  latent representations
                             <->  computational components
                             <->  security behaviour
```

Three specific gaps motivate this project:

- **Role has never been placed in the causal chain.** The role-confusion paper shows
  role representations *predict* injection success and states explicitly that it does
  not address whether role *causally* determines refusal. Zhao et al. do not model role
  at all. Whether role acts *through* harmfulness recognition, or reaches the control
  decision directly, or both, is unknown.
- **"Safety neurons" are treated as one kind of thing.** NeuroStrike, GateBreaker, L³,
  the safety-head work and the single-neuron work each identify components whose removal
  disables safety, but none says *which* representation those components carry:
  detecting harm, converting harm into a refusal decision, writing the refusal
  representation, or reading it out into tokens. A 2026 causal-mediation study
  (2604.11663) places harm understanding early and gating neurons late, but classifies
  components by layer and module type, not by the *named* variable they mediate, and
  ignores role.
- **The literature disagrees on what jailbreaks break.** Zhao et al. and the
  "robust harmful features" head study (2606.28153) find harm recognition survives
  successful jailbreaks; Ball et al. (2024) find jailbreaks *suppress* the model's
  perception of harmfulness; Kirch et al. (2411.03343) find 35 attacks use heterogeneous,
  non-linear mechanisms. These studies use different models, instruments and attack sets.
  No study has measured prompt-level attacks (injection, jailbreaks) and
  parameter-level attacks (neuron suppression, expert silencing) with the *same*
  instruments at *matched* behavioural outcome.

The objective is therefore to characterise the **causal architecture of LLM safety** and
to determine how different security failures compromise different stages of it.

---

## 3. Scope decision: RQ1–RQ4 only

The source drafts pose seven research questions. Co-authors proposed restricting to
RQ1–RQ4. **Assessment: agree, and for reasons that go beyond confounds.**

| Dropped RQ | Why it is dropped |
|---|---|
| **RQ5 — structure predicts vulnerability** | (a) A prediction claim across models needs a roster of well over ten models with the *full* RQ1–3 pipeline run on each; with 5–7 models any "architecture → attack budget" relation is an anecdote. (b) The dependent variable is ill-defined: the minimum attack budget `k*` depends on the attack's *own* selection method (NeuroStrike's probe, L³'s LSTM, GateBreaker's gate profile) at least as much as on the model, so `k*` measures the attack. (c) Models differ in safety-training recipe, data, and refusal style in ways we cannot observe or control, which confounds every cross-model comparison of "vulnerability". (d) The structural measures (`C_R`, `N_eff`, `r_eff`, `k_50`) each depend on the attribution method chosen, giving a garden of forking paths on a handful of points. |
| **RQ6 — context-dependent reorganisation** | A separate paper. Refitting directions and component sets per context on thin per-context cells is underpowered; the role-confusion and prompt-feature papers already show mechanisms differ by framing; and it would double the intervention programme. Its cleanest fragment — "the same harmful intent under a jailbreak framing" — is *exactly* RQ4's jailbreak family and is covered there. |
| **RQ7 — redundancy-based hardening** | Now done by others (L15): Distributed Safety Alignment (2608.01414) and NeuronGuard (2608.23959) redistribute safety across neurons and report robustness gains against neuron-level white-box attacks. A proof-of-concept here would be neither novel nor decisive, and it presupposes RQ5. |

**What survives from RQ5–7, in descriptive form only.** RQ3's ablation curves produce, for
free, *how many* components of a given ranking must be removed before a variable's
separation or the behaviour collapses. These counts are reported per model as
descriptive facts about implementation (they are what "sparse circuits" means), with **no
cross-model prediction claim**. Everything else — concentration indices, effective rank
as a vulnerability predictor, context stability, redundancy hardening, HOPE-style
functional-operator similarity, persona and emotion directions — is out of scope
(§ 12).

---

## 4. Central hypothesis and variables

### 4.1 The variables

Safety behaviour is mediated by at least three latent variables, each realised as a
low-dimensional subspace of the residual stream at each layer `l`:

| Variable | Meaning | Contrast that defines it | Read position |
|---|---|---|---|
| `R_role` | the perceived role, authority or origin of an instruction: system, user, assistant, tool/untrusted-external | identical content rendered under different roles | the instruction's content tokens; refit at `t_post` for geometry |
| `R_harm` | recognition that the content is harmful or security-relevant | harmful vs harmless instructions, with roles balanced and refusal behaviour controlled where possible | last instruction token `t_inst`; refit at `t_post` |
| `R_control` | the downstream mechanism governing refusal versus compliance | prompts the model *actually refuses* vs *actually complies with*, **within** a harm label, roles balanced | last prompt token `t_post`, i.e. before any token is generated |

Naming follows the scoped draft: `R_control`, not `R_refusal`, because the variable
covers the decision to comply as much as the decision to refuse (under-refusal on
harmful content and over-refusal on benign content are both "control").

Notation: `t_inst` is the last content token of the instruction-bearing turn; `t_post`
is the last token of the templated prompt, from which the first generated token attends;
`Y` is observable behaviour (refusal or compliance, and, for attacks, harmful output).

**Refusal is not assumed one-dimensional.** Each variable is allowed a `k`-dimensional
subspace `R^(l) = [r_1 … r_k]`; the smallest `k` that reproduces the behavioural effect of
the full subspace is measured (RQ1), and every downstream experiment uses the measured
`k`.

### 4.2 Candidate architecture, as a hypothesis

```
R_role  ->  R_harm  ->  R_control  ->  Y
```

is the simplest candidate. It is **tested, never assumed**, and it is one point in a
larger space. `Y` is a fourth node, not a synonym for `R_control`: the control
representation is read before any token is generated, behaviour is what follows, and the
two can dissociate. **The object RQ2 estimates is the directed graph over
`{R_role, R_harm, R_control, Y}`**: which edges are present, which paths are mediated by
which other variables (singly and jointly), where effects interact rather than add, and
how much of the full causal effect the three named coordinates jointly account for
(completeness). Named patterns in that space — sequential, parallel, role-direct, control
upstream, disconnected role — are labels for regions of it, used for readability; the
graph is the result even when no label fits. Alternatives the design must be able to
distinguish include: role and harm contributing to control independently (parallel); role
reaching control both directly and through harm; direct paths from role or harm to `Y`
that bypass control; role disconnected on the fitting distribution and active only under
injection; control not downstream of harm at all; and a large unexplained remainder, which
would mean the three variables are not the whole architecture.

At the computational level:

```
{ attention heads, MLP neurons, MoE experts, routing }  ->  { R_role, R_harm, R_control }  ->  Y
```

and components are classified by *what they consume and what they produce*:

| Functional role | Mapping | Meaning |
|---|---|---|
| **Detector** | `x -> R` for each `R in {R_role, R_harm, R_control}` | turns input features into a security variable. A detector of `R_control` (`x -> R_control`) triggers the refusal decision from surface features without passing through harm recognition: the shallow-alignment pathway that keyword-driven over-refusal and template jailbreaks would exploit |
| **Transformer** | `R_a -> R_b` for every ordered pair the structure allows: `R_role -> R_harm`, `R_role -> R_control`, `R_harm -> R_control`, and the reverse of any edge RQ2 establishes | converts one security variable into another. Which transformer classes exist is itself evidence about the architecture: `R_role -> R_control` transformers with no `R_role -> R_harm` ones is the component-level signature of a role-direct structure |
| **Writer** | `f_i(h) -> R` for each `R` | its output contributes directly to the target subspace |
| **Reader** | `R -> Y` for each `R` | consumes a security variable and produces logits or behaviour. A reader of `R_harm` or `R_role` is a direct path to behaviour that bypasses `R_control`, and is reported as such |

The taxonomy is stated for every variable and every ordered pair, not only for the
sequential chain, because the chain is a hypothesis: the functional classes RQ3 finds must
be able to describe whichever structure RQ2 returns. The classes are assigned by
intervention (ablation, patching, and whether a component's activation follows steering of
an upstream variable), never by correlation alone.

A central goal is to stop treating all "safety neurons" as the same thing.

### 4.3 Go/no-go criteria

```
GATE 1 (after RQ1)   R_role, R_harm, R_control are causally distinguishable
GATE 2 (start RQ3)   each external attack (NeuroStrike, L3) reproduces on our pipeline
GATE 3 (start RQ4)   at least one injection condition and one jailbreak family
                     reach non-trivial success on held-out intents on >= 1 roster model
```

Failing Gate 1 is publishable as a negative result and forecloses RQ2–4 as written.
Failing Gate 2 means component-level claims about that attack cannot be made. Failing
Gate 3 shrinks RQ4 to the attack families that work and says so.

---

## 5. Research questions

**RQ1 — Is LLM safety functionally decomposable?**
Are role perception, harmfulness recognition and refusal/compliance control represented
as distinct, recoverable and *causally distinguishable* latent variables? We establish
their geometry (overlap at a common position, against a split-half noise floor and a
random-direction null), dimensionality, emergence and persistence across depth, and
whether role is carried by metadata or by style — on every roster model, dense and MoE.
*Deliverable:* validated directions/subspaces per variable per layer, and GATE 1.

**RQ2 — What causal architecture connects the three variables and behaviour?**
We intervene on each variable and read *all three* variables and behaviour downstream,
using position-restricted steering, mediator clamping (natural direct and indirect
effects, single and joint mediators), directional ablation with rescue, and interchange
patching — plus the named test the drafts single out: under a successful prompt injection,
repair *only* role and see whether harm recognition, control and safe behaviour return,
versus strengthening control while leaving role corrupted. *Deliverable:* per model, an
estimated causal graph over `{R_role, R_harm, R_control, Y}` — edge set, mediation
annotations, interaction flags and a completeness estimate — from pre-registered rules,
with the named architectures as labels for the region the graph falls in.

**RQ3 — Which computational components implement each variable?**
We attribute MLP neurons, attention heads and (for MoE) experts and routing decisions to
each variable, classify them causally as detectors, transformers, writers or readers,
and — the central component experiment — take the externally defined component sets of
NeuroStrike (neurons), the safety-head literature (heads) and L³ (experts; GateBreaker and
SAFEx as baselines), suppress them, determine *which variable disappears*, and test
whether restoring that variable alone, with the components still suppressed, restores
safe behaviour. *Deliverable:* a representation-to-circuit map and a functional
classification of the components existing attacks exploit.

**RQ4 — Do different attacks compromise different stages of the same architecture?**
Prompt injection (direct and indirect), prompt jailbreaks, neuron suppression and expert
silencing are run on disjoint held-out intents and compared **at matched attack success**.
For each we measure the change in `R_role`, `R_harm`, `R_control` and in the RQ3
component sets, and — the causal test — which representation-level *repair* undoes the
attack. *Deliverable:* an experimentally inferred taxonomy
`attack -> compromised stage -> behavioural failure`, and an adjudication of the
harm-suppression vs refusal-suppression disagreement in the literature.

---

## 6. Related work and novelty

### 6.1 What is already established (and is therefore replication, not contribution)

Each row below is a reading of a paper and is audited against the paper's text before
submission (L16); rows the design depends on have their own ledger entries (L1–L14).

| Established result | Source | Our use |
|---|---|---|
| Refusal is mediated by a difference-of-means direction; ablation bypasses, addition induces | Arditi et al. 2024 (2406.11717) | estimator, steering convention, refusal readout |
| Refusal may need several directions / a concept cone; orthogonality is not causal independence | Wollschläger et al. 2025 (2502.17420) | motivates `k`-dim subspaces and the behavioural-`k` test |
| Harmfulness and refusal are distinct directions; steering asymmetry; some jailbreaks suppress refusal but not harm; latent harm is robust to adversarial fine-tuning | Zhao et al. 2025 (2507.11878) | RQ1 `{harm, control}` pair is a replication target; their asymmetry test is the template for GATE 1 |
| LLMs represent the role of text; style dominates tags; role confusion predicts injection success before generation; destyling collapses ASR | Ye et al., ICML 2026 (2603.12277) | role probe estimator; role corpus construction; `R_role` is their variable |
| Safety is sparse: ~3% params / 2.5% ranks (Wei et al. 2024); ~5% neurons recover 90% of safety (Chen et al. 2024); single neurons gate refusal (2605.08513); single heads matter (Zhou et al. 2024, Ships/Sahara) | as cited | components to explain; head-attribution method |
| NeuroStrike: probe-selected safety neurons, <0.6% pruned, 76.9% mean ASR, transfer across families | Wu et al., NDSS 2026 (2509.11864) | external neuron set (target of explanation) and RQ4 attack family |
| MoE safety concentrates in few experts: SAFEx (2506.17368), GateBreaker (2512.21008), L³ (2602.08741), RouteHijack (2605.02946); routing is topic-driven while safety is expert-localised (RASET, 2605.29708) | as cited | external expert sets; RQ4 attack family; RASET's routing finding is a prior we test against |
| Stage-wise reading of alignment: early ethical classification, mid-layer "emotion", late refusal tokens; jailbreaks disturb the transition (weak classifiers, logit lens) | Zhou et al. 2024 (2406.05644) | correlational precedent for "stages"; we make it interventional and add role |
| Causal mediation across layers/modules/neurons: harm understood early, MLP gating neurons late (2604.11663) | as cited | precedent for RQ3's detector/reader split, by layer not by variable |
| Different *parameter-level* unsafe routes (SFT, RLVR, abliteration) diverge mechanistically; harm recognition and compliance are separable axes (2604.18510) | as cited | precedent for a mechanistic taxonomy; parameter-level only, no prompt attacks, no role |
| Jailbreaks suppress the harmfulness percept (Ball et al. 2024, 2406.09289) vs. harm survives and refusal heads are suppressed (2606.28153) vs. heterogeneous mechanisms across 35 attacks (2411.03343) | as cited | the disagreement RQ4 adjudicates |
| SAE refusal features are redundant and layered; some are dormant until others are suppressed (2509.09708); minimal local causal explanations of jailbreak success (LOCA, 2605.00123) | as cited | related instruments; motivates rescue and `k`-dim subspaces |
| Redundancy-based hardening works (DSA 2608.01414, NeuronGuard 2608.23959) | as cited | reason RQ7 is dropped |

### 6.2 What is new in this project

1. **Role placed inside the causal chain of safety.** No prior work intervenes on the role
   representation and measures the consequence for harm recognition and for the control
   decision. The role-confusion authors list this as open.
2. **A three-variable causal structure established by mediation, ablation-with-rescue
   and interchange patching, not by steering asymmetry alone.** Zhao et al. establish
   two-variable separability by steering; we establish *ordering* and *mediation* with
   direct/indirect effect decomposition and matched random-direction nulls, on dense and
   MoE models across vendors.
3. **What the components that existing attacks exploit actually carry.** For NeuroStrike
   neurons, safety heads and L³ experts we determine which named variable disappears
   when they are suppressed and whether restoring that variable alone rescues safety.
   This converts "safety neuron / safety expert" into detector / transformer / writer /
   reader, and tests RASET's "routing is topic-driven" claim with representation-level
   readouts.
4. **A mechanistic attack taxonomy at matched behavioural outcome, spanning prompt-level
   and component-level attacks, validated by repair.** The stage an attack compromises is
   defined operationally by which representation-level repair undoes it, on held-out
   intents, with the same instruments across families. This is also the first common-frame
   measurement of the harm-suppression vs refusal-suppression question.

### 6.3 Claims we explicitly do not make

- That the three variables are the *only* safety-relevant variables.
- That the architecture is the same across contexts (RQ6 dropped).
- That any structural measure predicts vulnerability across models (RQ5 dropped).
- That any instrument (Llama-Guard, refusal-prefix rule) is validated against human
  labels. Agreement between independent model judges is reported as agreement.

---

## 7. Contributions and desired main result

**Contributions**

1. **Causal decomposition of LLM safety.** Role perception, harmfulness recognition and
   behavioural control are separable, recoverable latent variables that dissociate under
   intervention, across five models from four vendors including two MoE architectures.
2. **A causal safety map.** The estimated directed graph among the three variables and
   behaviour, established by cross-intervention, mediation, rescue and patching, with the
   candidate chain `role -> harm -> control -> Y` located inside it as one hypothesis
   among several, and with an estimate of how much of the causal effect the three
   variables jointly account for.
3. **Representation-to-circuit mapping.** Which neurons, heads, experts and routing
   decisions detect, transform, write or read each variable — including a functional
   classification of the components NeuroStrike and L³/GateBreaker exploit.
4. **Mechanistic attack taxonomy.** Whether prompt injection, jailbreaks, neuron
   suppression and expert silencing compromise distinct stages of the same architecture
   despite producing the same unsafe behaviour, with the stage defined by which repair
   works.

**Desired main result.** LLM safety is implemented by a structured causal architecture of
separable latent variables rather than a single refusal mechanism; that architecture is
realised by sparse, functionally heterogeneous components; and different security
failures compromise different stages of it. The strongest specific findings would be:
role reaches the control decision by a path partially independent of harm recognition;
NeuroStrike-type neurons are writers/readers of `R_control` while harm recognition
survives their removal (and L³ experts likewise or differently); injection corrupts role,
jailbreaks split between harm suppression and control suppression by family, and
component attacks remove control while leaving harm intact — so that a repair that fixes
one family does not fix another.

A well-powered negative on any of these is a finished result and is reported as such.

---

## 8. Experimental programme

### 8.1 Cross-cutting protocol (binding on every experiment)

**Corpus.** One frozen, model-independent instruction set, rendered per model through the
model's own chat template:

```
instructions x {harmful, harmless} x {system, user, assistant, tool} x {fixed-slot, natural-slot}
```

- Harmful sources: AdvBench, JailbreakBench behaviours, SORRY-Bench *base* prompts only
  (chosen for heterogeneous severity, so that the "recognises harm but complies" cell
  exists — D4, D9). Harmless: Alpaca standalone instructions and XSTest *safe* items
  (benign-but-sensitive, intended to stop `R_harm` collapsing into a topic detector — D7).
- `source` and `category` recorded per instruction; separation reported per source, never
  only pooled.
- Train/test split **by instruction**, stratified by (label, source), so no instruction's
  renderings straddle the split. Nothing is measured on the prompts used to fit it.
- **Held out by construction:** attack intents (StrongREJECT, HarmBench) for RQ4 and for
  the injection/jailbreak conditions of RQ2; SORRY-Bench's non-base styles (persuasion, role-play, authority, encodings, translations; D3)
  as RQ4 jailbreak material.
  None of these ever enters a fitting set, a labelling run or a corpus-widening step.
- Role classes are four: `tool` stands in for "untrusted external content", on the
  premise that a tool message is how external content reaches the model on every roster
  template (M2); chain-of-thought is a channel, not a role, and is out of scope.
- Two slot designs: **fixed-slot** (instruction in the same message slot for every role;
  only the marking varies) is primary; **natural-slot** (each role in its natural
  conversational position) is the deployment-realistic secondary. Agreement means role is
  carried by marking; disagreement means position carries part of it.
- A second, constant-content role corpus (C4 passages under each role tag), following the
  role paper, for cross-corpus transfer of the role probe.

**Behavioural label.** `R_control` needs a per-item label of what the model *did*:
generate greedily, label three-way `refused / complied / undetermined` using Arditi's
published prefix rule (primary) with stricter variants as sensitivity; degenerate,
truncated and clarifying outputs are `undetermined`, never `complied`; on harmful-and-
complied items Llama-Guard must judge the response unsafe or the item is
`undetermined`. The `harm x refused` 2x2 is emitted for every model before any direction
is trusted. **Risk R1 (§ 11):** on strongly aligned models the harmful-and-complied cell
may be empty; then the over-refusal contrast (refused vs complied within *harmless*) is the
cross-model control variable, and the empty cell is reported as a finding about the model.

**Estimators.** Difference of means for `R_harm` and `R_control` (Zhao); multiclass linear
probe *and* pairwise difference of means for `R_role` (only the latter can be steered
with). All on the residual stream so the three share one space. Directions keep their raw
class-gap norm; steering coefficients are denominated in class-mean separations
(`alpha = 1` is the published operating point of Arditi and Zhao, L6). Every direction records
which class is positive.

**Nulls and floors.** (i) A 1000-draw random-direction null, magnitude-matched, for every
separation and every steering effect, matched on `alpha`, layer, position and readout;
(ii) the split-half stability distribution as the noise floor for every cross-concept
similarity — nothing is compared to zero; (iii) length-only and bag-of-words classifiers every direction must beat (T6); (iv) count-matched random component sets for every component ablation;
(v) rank-matched random subspaces for every patching and rescue arm. Every claim on the
paper carries a bootstrap CI (resampled by instruction) and Benjamini–Hochberg FDR over
the family of layers/cells tested, with the family size stated.

**Capability bound.** An intervention cell is in-range only if the KL divergence on
harmless prompts at `t_post` stays inside a bound; a magnitude-matched random direction
sets the self-calibrating bound, and a fixed bound family is swept. No behavioural claim is
made from out-of-range cells. No attack-success number is reported without a utility
number at the same intervention.

**Positions and layers.** Positions are resolved per example and recorded. Geometry is
only ever computed between directions refit at a common position. Full layer curves
against relative depth are the primary figures; where one layer must be named it is
chosen on the train split by a rule fixed in advance. Directions are never compared
across models — only quantities are.

**Readout completeness.** Every intervention arm, in every RQ, reads all three latent
variables at every downstream position where they are defined (`t_post` refits, and
later tokens where used) *and* behaviour. `R_control` is a latent variable, not a name for
behaviour: a verdict never rests on one readout shown alone, and dissociations between a
representation and behaviour are reported as findings about readers, not as noise.

**Cross-model rule.** A roster-level claim requires the same verdict on at least 4 of 5
models; otherwise the per-model table is the finding. Dense and MoE are reported separately
as well as pooled.

**Pre-registration.** Every decision rule in this section and in § 8.2–8.5 is written in
`experiments.md` with its thresholds before the experiment it governs produces a number.
Amendments are dated and never retroactive.

### 8.2 RQ1 — recovering and validating the variables

| # | Experiment | Establishes | Manipulation / measurement | Controls | Reading |
|---|---|---|---|---|---|
| **E1.0** | Corpus build and freeze | the substrate every RQ shares | offline; the crossed corpus, held-out pools, transfer corpus, per-item length and tokenisation checks corpus-wide | — | freeze only after E1.1's labelling on every roster model |
| **E1.1** | The three variables are recoverable | held-out separation per layer per variable; role probe accuracy; refusal labelling and the 2x2 | diff-of-means, role probe/contrast; residual stream; every layer | random-direction null, length-only and bag-of-words baselines, split-half floor, cross-corpus role transfer, positive control, Zhao and role-paper replications as anchors | a variable that does not beat its nulls is not recoverable; stop and fix extraction |
| **E1.1b** | `R_harm` with refusal held constant | that harm is not refusal renamed | refit harm within refused items and within complied items, role-balanced | compare to pooled fit against the split-half floor | if controlled and pooled fits differ, the pooled one carries refusal and the controlled one is used downstream |
| **E1.2** | Geometry | overlap between variables at a common position, per layer | cosine (k = 1) or principal angles / canonical correlations (k > 1), on `t_post` refits | split-half floor, random-subspace null | overlap is reported as "fraction of the split-half ceiling"; high overlap does *not* refute distinctness — E1.6 does |
| **E1.3** | Dimensionality | the `k` each variable needs | spectral effective rank over stratified refits; **behavioural `k`** = smallest subspace reproducing >= 90% of the full subspace's steering effect | random subspaces | the behavioural `k` is what the paper reports and what RQ2–4 use |
| **E1.4** | Emergence and persistence | depth at which each variable appears and how long it persists | separation vs relative depth; persistence of a layer-`l` direction at layers > `l` | CI on onset depth, three thresholds | an ordering across variables is a hypothesis for RQ2, never causal evidence |
| **E1.5** | Metadata vs style for role | whether `R_role` follows the tag or the register | fixed-slot vs natural-slot; role contrast within each source style; *(optional)* a controlled-register rewrite corpus with content preserved, rewriter outside the roster | split-half floor | connects to "style dominates tags"; decides how role interventions must be built |
| **E1.6** | **GATE 1 — causal distinguishability** | the variables dissociate under intervention | 3x3 steering matrix: steer `A` at layer `l`, read `B` at downstream layers and at `t_post` *and later tokens*; behaviour by generation | alpha-matched random-direction bands, capability bound, FDR, train-selected layers evaluated on test | G1 diagonal sanity (not evidence); G2 asymmetry `delta_{A->B} != delta_{B->A}` with non-overlapping CIs; G3 behavioural dissociation — a condition moves behaviour beyond its band while another variable stays inside its band. **Verdict = G1 and G2 and G3**, robust across the swept bounds |

A named, pre-registered cell from the informal plan: steering a *benign user*
instruction toward the *tool* role is predicted to raise its harm projection downstream.
Recorded with its sign in advance so confirming it cannot be a post-hoc reading.

Dropped from the drafts: persona-trait and emotion directions (optional in the drafts;
confounded and off-thesis).

### 8.3 RQ2 — the causal structure

All experiments use the validated directions and the behavioural `k` from RQ1, on the
test split, with the RQ1 nulls and capability bound.

| # | Experiment | Question | Manipulation | Readout | Verdict rule (sketch) |
|---|---|---|---|---|---|
| **E2.1** (+ E2.1s, E2.1n, E2.1g) | Directed reach, with estimator sensitivity, the natural role manipulation and the leakage audit | does steering `A` at the instruction move `B` at the decision point, for every ordered pair, and behaviour? | position-restricted steering at the instruction span, read at `t_post` and later tokens, corrected for the steered vector's own arrival at the read position | standardised projection change on each of the three variables (role included, via its `t_post` refit); behaviour | `present(A -> B)` iff the corrected effect exceeds the random band with sign tracking `alpha` in >= theta of in-range cells, FDR-surviving; `absent` only with power (another edge present in the same design) |
| **E2.2** | Mediation matrix | for every source and every outcome (each downstream variable and behaviour): which other variables, singly or jointly, carry the effect, and is there a direct remainder? | steer the source; clamp one or both other variables to their clean value at every layer past the steer layer (natural direct effect); clamp to the steered value in an unsteered run (indirect effect) | `TE`, `NDE`, `NIE`, mediated share per mediator and for the joint clamp, additivity residual (interaction) | mediated iff share >= 0.5 with CI above the random-clamp share; a direct path to `Y` iff the joint clamp leaves a remainder beyond the band; private components separate shared geometry from causal flow |
| **E2.3** | Necessity, rescue, completeness | is the instruction-side signal *necessary*? does restoring one coordinate return behaviour? how much of the full effect do the three coordinates jointly span? | directional ablation of `A` at prompt positions; then restore one coordinate, or all three jointly | refusal on harmful prompts; all three representations | necessary iff refusal drops beyond the random-ablation band; rescue iff the refusal left missing is < half of what ablation removed, with a full-restore ceiling and a random-subspace floor; completeness = joint-restore recovery as a fraction of full-restore recovery |
| **E2.4** | Which coordinate carries the signal | interchange patching at `t_inst` | patch the full residual (ceiling), `harm`, `harm ⊥ control`, `control`, `role`, the three jointly, rank-matched random subspaces, from a harmful donor into a harmless recipient and the mirror | carried fraction of the ceiling, on behaviour and on each downstream variable | a coordinate carries iff fraction >= 0.5 with CI above every random draw; a random draw reaching 0.5 voids the cell |
| **E2.5** | **The role edge under injection** (the named experiment of both drafts) | under a *successful* injection, does repairing only role restore harm recognition, control and safe behaviour, whereas strengthening control alone restores behaviour but not harm recognition? | injection corpus: benign user task + hostile tool payload (indirect) and user-turn injection (direct); role repair on the payload span; control strengthening on the post-instruction span; harm steering as positive control | role-probe confusion, harm and control projections, guard-judged behaviour | role present and control absent on the harm readout -> ordering evidence for `role -> harm`; both present -> no ordering claimed; requires GATE 3 |
| **E2.6** | Graph estimate | what the architecture *is* | none (CPU adjudication of E2.1–E2.5) | the estimated graph over `{role, harm, control, Y}`: edge status per ordered pair, mediation annotation per path, interaction flags, completeness; then a pattern label | edges and annotations from pre-registered predicates; labels {sequential, parallel, role-direct, partially overlapping, control-upstream, disconnected role, incomplete, undecidable} assigned to the graph afterwards for readability and never in place of it; role's edges adjudicated on the fitting corpus *and* under injection, injection primary where estimable; roster-level rule 4 of 5 applies to edges |

Edges into role (`harm -> role`, `control -> role`) are measurable only at `t_post`, through
role's `t_post` refit, when the source is steered upstream of it — and only on models where
that refit is decodable and the role diagonal is steerable there (T7); at the content span
they are not measurable by token-level intervention and are never reported as absent.

Dropped: "remove fear", "switch persona" (depend on the dropped optional directions).

### 8.4 RQ3 — implementation by components

Component families: MLP neurons (all dense models; shared and routed expert MLPs on MoE),
attention heads (all models), MoE experts and routing (MoE models). The component
definitions presuppose a gated, pre-norm architecture on every roster model (M13).

| # | Experiment | Establishes | Method | Controls |
|---|---|---|---|---|
| **E3.0** | **GATE 2 — the external attacks reproduce** | that the component sets we explain are the ones the papers found | run NeuroStrike (shipped or retrained probe, their selection rule and prune site) and L³ (their routing traces, LSTM and silencing) on our pipeline; match their reported ASR and utility within tolerance; bitwise equivalence of hooks where their code is available | their judge vs ours on the same responses |
| **E3.1** | Writer attribution | which components write into each variable | `C_{i,l}^R = E_h ‖ P_R f_{i,l}(h) ‖_2`, with `f_{i,l}(h) = a_{i,l}(h) w_out_{i,l}` for neurons and the per-head OV output for heads; `P_R` the projector onto the RQ1 subspace | matched random subspaces; the same score against `R_harm`, `R_role` and `R_control` gives **functional overlap** (components writing several variables) |
| **E3.2** | Causal functional classification | detector / transformer / writer / reader, for the top-ranked components and the external sets | ablate the component set and read all three variables and behaviour; patch its activation between harmful and harmless runs; steer a variable and read whether the component's activation follows it (reader test) | count-matched random sets; capability bound |
| **E3.3** | **What external component sets carry** (the "connect NeuroStrike to the representations" item of the minimum core) | which variable disappears when NeuroStrike neurons / safety heads / L³ experts are suppressed, and whether restoring that variable alone rescues safety | suppress at the attack's dose ladder; per direction measure separation drop and projection change; rescue by clamping the removed coordinate to its clean value with components still suppressed | count-matched random sets (removed verdict), full-residual restore (ceiling), random subspace (floor); GateBreaker, SAFEx and random-matched expert sets as MoE baselines |
| **E3.4** | Attribution validation | that our ranking is not an artefact | compare `C` rankings with ablation-based ranking, gradient x activation, NeuroStrike's probe ranking, and random; test whether NeuroStrike's neurons are recoverable by alignment alone (the informal plan's "same neurons by another route") | matched random |
| **E3.5** | Dense vs MoE decomposition | whether safety lives in routing or in expert weights, and which variable each carries | three MoE interventions on the same items: mask the router (L³-style silencing), prune neurons inside the selected experts (GateBreaker-style), and the representation-level intervention on the variable itself; read routing entropy, expert selection, and all three variables | random experts, random expert-neurons at matched count |
| **E3.6** | Descriptive sparsity | how many components a variable's implementation rests on | ablation curves of separation and behaviour vs number of components removed, by each ranking | random rankings |

E3.5 distinguishes the hypotheses the long draft lists: the router selects a
safety-specific pathway; the router selects topic experts and safety emerges downstream
(RASET's claim); particular experts write `R_control`; experts encode harm while
attention implements refusal; several experts redundantly contribute.

Attention is included at head level (attribution and ablation). A full circuit-level
account of information movement between positions is out of scope; where a claim is
MLP-only it says so.

### 8.5 RQ4 — the mechanistic attack taxonomy

| # | Experiment | Content |
|---|---|---|
| **E4.0** | **Instruments, matching, calibration** | ASR = Llama-Guard judges the response unsafe *and* the response is not degenerate; the refusal-prefix rule and NeuroStrike's own rule reported beside it; a second model judge for the disagreement set, reported as agreement not correctness. Utility at every intervention: the six NeuroStrike benchmarks (L11) plus IFEval. **Behavioural matching:** every family is run on a dose ladder (injection template strength, jailbreak style, pruning fraction, silenced experts) and signatures are compared at matched ASR bands on the *same* held-out intents; a difference in signature at unmatched ASR is attack strength, not stage. **Calibration arms:** representation-level ablation of harm, of control, and role steering on inert payloads are run as if they were attacks; each stage label is `calibrated` on a model only if its arm reproduces that label's signature and repair pattern there |
| **E4.1** | Prompt injection | indirect (benign user task + hostile tool payload) and direct (hostile content in the user turn under a benign task), template rungs of increasing role mimicry; tag forging excluded from the headline |
| **E4.2** | Jailbreaks | SORRY-Bench's held-out styles grouped into persuasion, role-play/authority, and encoding families; one canonical template family (e.g. DAN/AIM; K11); PAIR on every dense model (K12); GCG only if budget allows (K7). Every family carries an *inert-framing* arm (the same framing around a harmless request) so a signature is read against the framing, not against a short clean prompt |
| **E4.3** | Neuron suppression | NeuroStrike, dose ladder over layer prefix and z-threshold |
| **E4.4** | Expert silencing | L³ (routing mask) and GateBreaker (expert-neuron pruning) on the MoE models, dose ladder |
| **E4.5** | Stage signature | per attack, per item, the change in `R_role` (probe confusion), `R_harm` (separation and projection at `t_inst` and `t_post`), `R_control` (projection at `t_post`), and in the activation of RQ3's classified component sets; signature vectors compared within and across families at matched ASR |
| **E4.6** | **Repair as the definition of stage** | for each attack at matched ASR, apply each representation-level repair from RQ2 (restore role, restore harm, restore control, all three jointly, full restore, random) and measure recovery of safe behaviour and of the other representations; the stage an attack compromises is the coordinate whose restoration recovers it; the joint arm against the full arm says whether the three variables span the attack's effect at all; a repair that fixes family A but not family B is the evidence that they compromise different stages |
| **E4.7** | Taxonomy | pre-registered decision rule from E4.5/E4.6 to {role corruption, harm-recognition failure, control failure with harm intact, component bypass with variables intact, mixed, outside the architecture, undetermined}; each label `calibrated` or not per model from E4.0's calibration arms; per model and family; roster-level rule |

Dropped: representation steering as an "attack family" (it is our instrument, so its
diagnosis is circular; representation-level interventions appear only as E4.0's calibration
arms, which never enter the taxonomy as rows) and
safety-removing fine-tuning (parameter-level routes are covered by 2604.18510 and carry
confounds we cannot control).

---

## 9. Models, data, attacks, instruments

**Roster (proposed; final choice after the template and attack-support checks M1–M7 and
L7–L10 in `assumptions.md`).** Depth of causal analysis over breadth: five models, four vendors, two
MoE architectures. Selection criteria, in order: (1) all four role classes render through
the model's own template; (2) an external attack set exists or can be produced with the
attack's released code; (3) evaluated by at least one of the source papers, so anchors
exist; (4) text-only where possible, so a vision tower is not an uncontrolled difference.

| Slot | Candidate | Why |
|---|---|---|
| Dense 1 | Qwen2.5-7B-Instruct | NeuroStrike ships probe weights (M6) and reports the attack landing on it (L7); Zhao-style corpus |
| Dense 2 | Llama-3.1-8B-Instruct | cross-vendor; used by Zhao, Arditi, 2604.18510, 2509.09708 (L16); tool role supported (M2) |
| Dense 3 | one of Gemma-3-12b-it / Phi-4 / Qwen2.5-14B-Instruct | third vendor where the template supports system and tool (M2); NeuroStrike ships weights for all three (M6) |
| MoE 1 | Qwen3-30B-A3B | supported by L³ (M7), GateBreaker (L9) and the role paper; accessible router (M5) |
| MoE 2 | OLMoE-1B-7B-Instruct (cheap, fully open, L³-supported — M7) or gpt-oss-20b (role paper's primary model; reasoning channel complicates roles and labels — M4) | second MoE vendor |

The previous roster's Qwen3.5 checkpoints are not carried forward: they are multimodal
wrappers with a reasoning block ahead of assistant turns (M4, M12), and no external attack
set supports them; every attack would have to be re-derived, which weakens the "these are
the components the attack found" claim.

**Data.** Fitting: AdvBench, JailbreakBench, SORRY-Bench base, Alpaca, XSTest, C4 (role
transfer). Held out: StrongREJECT and HarmBench intents (disjointness enforced by exact
and near-duplicate matching before use), SORRY-Bench non-base styles, injection templates.

**External attack code.** NeuroStrike (released, pinned commit; probes retrained with
their script where no weights ship), L³ (user's own release), GateBreaker (Zenodo
release), SAFEx (GitHub). Each is run through its own selection rule; our code only
applies the resulting sets, and equivalence to the reference hooks is tested where the
reference runs.

**Instruments.** Refusal: Arditi prefix rule on greedy generations, three-way with a
degeneracy gate. Response harm: Llama-Guard-3-8B. Utility: HellaSwag, RTE, WinoGrande,
ARC-Challenge, OpenBookQA, CoLA (NeuroStrike's six) plus IFEval; screening subsample per
arm, full sets at headline configurations. Capability during steering: KL on harmless
prompts.

**Evaluation dimensions, for every claim:** behavioural security (refusal, ASR), utility,
mechanistic consistency (did the intervention move the representation it was supposed
to?), cross-model generalisation.

---

## 10. Order and minimum viable paper

**Order.** RQ1 -> RQ2 -> RQ3 -> RQ4, one experiment at a time; the next starts when the
current is *settled*, meaning trustworthy, not favourable. Two items are pulled forward
across RQs: the injection condition (E2.5 needs it, E4.1 extends it) and GATE 2 (the
attack reproductions can run in parallel with RQ2 since they touch no RQ1/RQ2 code path).

**Tiers.** Every experiment is tiered in `experiments.md` § 8: *core* (its RQ cannot be
answered without it or a core rule consumes its output), *supplementary* (not needed for
any RQ; kept because the finding is worth reporting; first to be cut, never a gate). The
test applied to every item was whether it could change a main result — through method
validation, estimator dependence, or roster coverage — and anything that could is core.
Supplementary items: E1.3's effective-rank statistic, E1.4, E1.5 level 2 (core if D15
fails), E3.1's concentration curve, E3.6 pass 2, the encoding jailbreak family, GCG.

**Minimum viable paper** (the drafts' "minimum core", restated for RQ1–4): every core
experiment, on all five models for RQ1–RQ2, with E2.5 on the injectable models; RQ3's
core with NeuroStrike on the dense models and L³ on at least one MoE; RQ4's core with
injection, one jailbreak family and neuron suppression on the dense models. **Full paper**
adds expert silencing on both MoE models, all jailbreak families, and the roster-level
taxonomy. Supplementary items go in as an appendix if run.

Scheduling and compute figures are not part of this plan. Compute estimates are ledger
entries (`assumptions.md` B1–B3) because they are measurements.

---

## 11. Risks

| # | Risk | Mitigation | Whether it works is verified by |
|---|---|---|---|
| R1 | The harmful-and-complied cell is empty on strongly aligned models, so `R_control` (under-refusal) is unidentifiable | heterogeneous harmful set including borderline SORRY-Bench items; the over-refusal contrast on benign-sensitive items as the cross-model variable; the empty cell reported as a finding; jailbreaks never used to fill it; an explicit compliance-forcing arm only as a pre-registered amendment of last resort | D4, D9, D13, D14 (E1.1 stage 1 on every model, before the corpus is frozen) |
| R2 | Role has no measurable effect on behaviour on the fitting corpus, so E2.5 is the only place role matters | that is a valid structure verdict; role's edges are adjudicated on the fitting corpus *and* under injection; GATE 3 guards the design | E2.1n and E2.5 (K1) |
| R3 | No injection template reaches non-trivial success on a model | rungs of increasing role mimicry; a stronger reasoning-channel rung on models that have one; injection-resistant models reported as such | K1 (E2.5) |
| R4 | External attacks do not reproduce on our pipeline | GATE 2 before any component claim; released weights and code only; never a home-grown selection called theirs | L7–L10, M6, M7, M10 (E3.0) |
| R5 | Instruments key on features that leak into `R_control` | three-way labels, degeneracy gate, no-guard sensitivity refit, agreement between independent judges | I1–I5, I10, I11, L14, T6 |
| R6 | Steering effects are geometric leakage rather than causal flow | private components, D-2 correction, mediation with a random-clamp null, interchange patching | V10, V13, T7, E2.1g, E2.2 invalidity checks |
| R7 | Attack families compared at different strengths look different for that reason alone | dose ladders and matched-ASR comparison; inert-framing arms | K2, K6, K11 (E4.0) |
| R8 | Compute: five models x full intervention grids | staged sweeps; a runner that never repeats a capture; screening subsamples for utility | B1–B3, M11 |
| R9 | Scope creep of the kind that produced the last codebase | every experiment has an ID, a purpose and a verdict rule; anything without one is not implemented; every assumption has a ledger entry | — |

---

## 12. Explicitly out of scope

Concentration, effective-rank or redundancy measures as *predictors* of vulnerability
(RQ5); context-dependent reorganisation and functional-vs-implementation stability
(RQ6); redundancy-based hardening and directional adapters (RQ7); HOPE and any
functional-operator analysis; persona and emotion directions; chain-of-thought as a role
class; safety-removing fine-tuning as an attack family; a full attention-circuit account;
multimodal, multilingual, RAG-specific and agent-memory security; backdoors; secure code
generation; any new defence or PEFT method. Each is a follow-up, not a dilution of this
paper.

---

## 13. References

- Arditi et al. 2024. Refusal in Language Models Is Mediated by a Single Direction. arXiv:2406.11717.
- Wollschläger et al. 2025. The Geometry of Refusal in LLMs: Concept Cones and Representational Independence. arXiv:2502.17420.
- Zhao, Huang, Wu, Bau, Shi 2025. LLMs Encode Harmfulness and Refusal Separately. arXiv:2507.11878.
- Ye, Cui, Hadfield-Menell 2026. Prompt Injection as Role Confusion. ICML 2026. arXiv:2603.12277.
- Wu et al. 2026. NeuroStrike: Neuron-Level Attacks on Aligned LLMs. NDSS 2026. arXiv:2509.11864.
- te Lintelo, Wu, Picek 2026. Large Language Lobotomy: Jailbreaking MoE via Expert Silencing. arXiv:2602.08741.
- GateBreaker: Gate-Guided Attacks on MoE LLMs. arXiv:2512.21008.
- SAFEx: Stable Safety-critical Expert Identification. NeurIPS 2025. arXiv:2506.17368.
- RouteHijack: Routing-Aware Attack on MoE LLMs. arXiv:2605.02946.
- RASET: Router-Agnostic Safety-Critical Expert Tuning. arXiv:2605.29708.
- Wei et al. 2024. Assessing the Brittleness of Safety Alignment via Pruning and Low-Rank Modifications. arXiv:2402.05162.
- Chen et al. 2024. Finding Safety Neurons in Large Language Models. arXiv:2406.14144.
- Zhou et al. 2024. On the Role of Attention Heads in LLM Safety. arXiv:2410.13708.
- A Single Neuron Is Sufficient to Bypass Safety Alignment. arXiv:2605.08513.
- Zhou et al. 2024. How Alignment and Jailbreak Work: Explain LLM Safety through Intermediate Hidden States. arXiv:2406.05644.
- Ball et al. 2024. Understanding Jailbreak Success: A Study of Latent Space Dynamics. arXiv:2406.09289.
- Kirch et al. 2024. What Features in Prompts Jailbreak LLMs? arXiv:2411.03343.
- Robust Harmful Features Under Jailbreak Attacks: Attention Head Specialization. arXiv:2606.28153.
- Why Do Large Language Models Generate Harmful Content? (multi-granular causal mediation). arXiv:2604.11663.
- Kabir, Tiganj 2026. Behind Harmful Compliance: Behavioral and Mechanistic Divergence Across LLM Jailbreaks. arXiv:2604.18510.
- Beyond "I'm Sorry, I Can't": Dissecting LLM Refusal (SAE features). arXiv:2509.09708.
- LOCA: Minimal, Local, Causal Explanations for Jailbreak Success. arXiv:2605.00123.
- No Single Neuron of Failure: Distributed Safety Alignment. arXiv:2608.01414.
- NeuronGuard: Ablation-Aware Safety Signal Redistribution. arXiv:2608.23959.
- HOPE: Hilbert Operator Progressive Encoding. arXiv:2607.21366 (cited only to record why it is out of scope).
