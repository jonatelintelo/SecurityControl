# Research Plan — The Causal Architecture of LLM Safety

**This file is an extract, not a decision record.** It merges the two source
documents into one plan and records where they disagreed. Nothing here is ours:
every claim traces to `notes_from_mail.tex` or `main.tex`. Design decisions,
criteria and parameters live in `EXPERIMENTS.md`; verified facts about models,
templates and data live in `ENVIRONMENT.md`.

Edit this file only when the source documents change.

## Sources and precedence

| Source | Role |
|---|---|
| `notes_from_mail.tex` — *The Causal Architecture of LLM Safety* | **The scoped plan. Wins on every conflict.** |
| `main.tex` — *Mapping the Security Control Architecture of Language Models* | The earlier long-form draft. Supplies method detail the scoped version compressed away |

The scoped version is stronger on framing (structural prediction, context
dependence) and thinner on method. The older draft carries the representation
extraction specs, the attribution formalism, the functional-role taxonomy, the
MoE decomposition, and the models/evaluation/scope-control sections. Both are
needed; neither alone is sufficient.

### Where the two disagree, and the resolution

| Point | `main.tex` | `notes_from_mail.tex` | Resolution |
|---|---|---|---|
| Third variable | `R_refusal` — *"the downstream mechanism responsible for **executing refusal** behavior"* | `R_control` — *"the downstream mechanism **governing refusal versus compliance**"* | **`R_control`**, with the scoped definition. Not a pure rename: the scoped variable also covers the decision to comply |
| Number of RQs | 6 | 7 | **7.** `main.tex` RQ6 (redundancy) becomes scoped **RQ7**; scoped RQ6 (context dependence) is new and has no counterpart |
| Model roster | 5–7 models, ≥4 families | *"one representative dense model and one representative MoE model"* initially | **`main.tex`.** The scoped narrowing was a risk-minimisation device that turned out to cost more than it saved: adding a model is two jobs, not a re-run, because the corpus stores instructions rather than renderings. RQ1 therefore ran six models across four vendors — inside `main.tex`'s target — rather than deferring the expansion to PEP-7 |
| Concentration measure | `κ_R(k)`, top-`k` mass | `C_R = Σ p²`, `N_eff`, `r_eff`, `k_50` | **Both.** They are different quantities, not variants. `κ` is the one tied directly to NeuroStrike/GateBreaker vulnerability |
| Order of taxonomy vs prediction | taxonomy (6) before concentration (7) | prediction test (6) before taxonomy (9) | **Scoped.** The prediction test is named *"the central predictive hypothesis of the paper"*; the taxonomy is one contribution of five |
| Redundancy hardening | contribution #7, a full experiment | *"treated only as an optional causal validation and is not required as a primary contribution"* | **Scoped.** Optional |
| Refusal dimensionality | *"a general `k`-dimensional refusal subspace"*, smallest `k` that explains the interventions | not restated | **`main.tex`** — the scoped version compressed it away rather than retracting it |

---

## Motivation

Existing work gives disconnected pieces of one phenomenon. Refusal can be
manipulated through low-dimensional activation-space interventions, though it may
be mediated by several directions or a higher-dimensional subspace rather than a
single universal axis. Harmfulness and refusal are represented separately —
recognising a harmful request and deciding how to respond are distinct
computations. NeuroStrike and GateBreaker show alignment can depend on
surprisingly sparse sets of neurons and, in MoE models, on a limited number of
experts, neurons and routing decisions. Prompt-injection work shows the perceived
role or authority of an instruction is itself internally represented and
manipulable. HOPE gives a functional reading of neurons as low-rank operators, and
shows parameter redundancy and functional redundancy need not coincide.

What is missing is a unified causal account connecting

```
semantic security variables <-> latent representations
                            <-> computational components
                            <-> security behavior
```

and, more importantly, whether the **structure** of that architecture determines
its security. Existing work identifies vulnerable neurons, directions, experts or
routes only *after* an attack. The scoped plan asks whether measurable structural
properties predict, *before* the attack, how easily alignment can be disabled.

---

## Central hypothesis

Safety behaviour is mediated by multiple interacting latent security variables,
including at least

```
R_role      perceived role, authority, or origin of an instruction
R_harm      recognition of security-relevant or harmful content
R_control   the downstream mechanism governing refusal versus compliance
```

A simple candidate architecture is

```
R_role -> R_harm -> R_control -> Y
```

where `Y` is observable behaviour. **This ordering is a hypothesis to be tested,
not an assumed architecture.** Role and harm may contribute independently to
downstream control; several pathways may operate in parallel; the structure may
change with context.

At the computational level:

```
{attention, neurons, experts, routing} -> {R_role, R_harm, R_control} -> Y
```

The stronger hypothesis:

```
high functional concentration / low redundancy  ==>  greater intervention sensitivity
```

A model may hold billions of redundant parameters while relying on a very small
effective structure for an individual safety-critical function.

The architecture need not be static. The same behavioural safety function may be
realised through different computational pathways under different roles, jailbreak
contexts, prompt formulations or architectures. Hence the distinction between
**functional stability** and **implementation stability**.

---

## Research questions

**RQ1 — Is LLM safety functionally decomposable?**
Are role perception, harmfulness recognition, and refusal/compliance control
represented as causally distinguishable latent variables? Investigate their
geometry, dimensionality, emergence across layers, and causal interactions
**without assuming a fixed sequential architecture**.

**RQ2 — What causal architecture connects these security variables?**
Independently manipulate each representation; measure effects on the remaining
representations and on final behaviour. Activation patching, cross-intervention,
mediation and rescue distinguish sequential, parallel and partially overlapping
causal structures.

**RQ3 — Which computational components implement each security function?**
Identify neurons, attention components, MoE experts and routing decisions that
causally **detect, transform, write or read** individual security
representations. Specifically determine what functions are performed by
components previously identified by NeuroStrike and GateBreaker.

**RQ4 — Do different attacks compromise different stages of the same safety
architecture?**
Compare prompt injection, jailbreaks, representation-level interventions,
NeuroStrike-style neuron interventions and GateBreaker-style MoE interventions.
Determine whether behaviourally similar failures arise through role corruption,
failed harm recognition, failed behavioural control, routing bypass, or
combinations.

**RQ5 — Does the structure of the safety architecture predict vulnerability?**
Quantify functional concentration, effective dimensionality, causal redundancy,
component diversity and pathway diversity per security function. Test whether they
predict the minimum intervention required to compromise safety:

```
architecture measured before attack  ->  attack budget required for safety failure
```

*The central predictive hypothesis of the paper.*

**RQ6 — Is the safety architecture static or context-dependent?**
Examine the same harmful intents under direct requests, paraphrases, role-play,
authority manipulation and jailbreaks. Distinguish `stable representation + stable
implementation`, `stable representation + dynamic implementation`, and
`context-dependent functional reorganization`.

**RQ7 — Does experimentally increasing redundancy increase robustness?**
Optional causal validation of RQ5. Redistribute safety-writing capacity across
additional components or constrained low-rank adapters; test whether the
intervention budget required to disable safety increases. **Not a new defense
method** — a controlled test of the architecture–robustness relationship.

---

## Plan sections and their keys

Every experiment in `EXPERIMENTS.md` cites one of these.

| Key | Section | Source |
|---|---|---|
| **PLAN-RQ*n*** | Research questions 1–7 | scoped |
| **PLAN-PEP-*n*** | Prioritized Experimental Plan, items 1–10 | scoped |
| **PLAN-EXTRACT** | Representation Extraction — per-variable specs | main |
| **PLAN-GEOM** | Experiment 1: Representational Geometry | main |
| **PLAN-XINT** | Experiment 2: Cross-Intervention Causal Matrix | main |
| **PLAN-ATTR** | Experiment 3: Component-Level Attribution | main |
| **PLAN-ROLES** | Writers, Readers, Detectors, and Transformers | main |
| **PLAN-MED** | Experiment 4: Causal Mediation and Rescue | main |
| **PLAN-MOE** | Experiment 5: Dense versus MoE Security Circuits | main |
| **PLAN-TAX** | Experiment 6: Mechanistic Attack Taxonomy | main |
| **PLAN-CONC** | Experiment 7: Security Concentration and Functional Redundancy | main |
| **PLAN-HOPE** | Relation to Functional Operator Analysis | main |
| **PLAN-RED** | Experiment 8: Proof-of-Concept Security Redundancy | main |
| **PLAN-STRUCT** | Structural Characterization of Safety Functions | scoped |
| **PLAN-CPE** | Central Prediction Experiment | scoped |
| **PLAN-CDRE** | Context-Dependent Reorganization Experiment | scoped |
| **PLAN-INF** | The informal *Experiments* list — Passive / Active | scoped |
| **PLAN-SCOPE** | Models, Evaluation, Scope Control | main |

**The plan does not map its experiments onto its RQs. That mapping is ours** and
lives in `EXPERIMENTS.md`.

---

## PLAN-EXTRACT — Representation extraction

Per model and layer `l`, estimate representations for role, harmfulness and
refusal.

**Role.** Matched examples in which identical or semantically similar content
appears under different roles — **system, user, assistant, tool, untrusted
external content**. Estimate a layer-wise direction or subspace `R_role^(l)`.
Consider **both linear probes and direct activation-space contrasts**. Evaluate
whether explicit role metadata and linguistic style produce compatible or
conflicting representations.

**Harmfulness.** Matched harmful and benign prompts, **controlling for refusal
behaviour where possible**. *"Particular attention will be paid to examples in
which a model recognizes harmfulness but nevertheless complies, allowing
harmfulness representation to be separated from refusal behavior."*

**Refusal.** Not assumed one-dimensional. A general `k`-dimensional subspace
`R_refusal^(l) = [r_1 … r_k]`. Compare single- and multi-direction
representations; determine **the smallest dimensionality required to explain the
observed behavioural interventions**.

## PLAN-GEOM — Representational geometry

Layer-wise overlap per pair of representations via **principal angles, projection
measures and canonical correlations**; `cos(r_a, r_b)` as the 1-D special case,
with subspace-level measures as the primary analysis. Study: dimensionality of
each representation; layer at which each emerges; persistence across subsequent
layers; overlap between the three; variation across prompt categories; variation
across model families; differences between dense and MoE.

*"This experiment is descriptive and is therefore not intended to constitute the
main contribution. Its purpose is to establish the latent variables needed for the
subsequent causal analyses."*

## PLAN-XINT — Cross-intervention causal matrix

Interventions `I_role`, `I_harm`, `I_refusal` that **strengthen, suppress, or
restore** the corresponding representation. Build

```
             R_role  R_harm  R_refusal
 I_role         *       ?        ?
 I_harm         ?       *        ?
 I_refusal      ?       ?        *
```

measuring both the resulting internal representations and final behaviour.

Named example: under a successful prompt injection, repair **only** the role
representation, `h_l' = h_l + α r_role^(l)`, and test whether harm recognition,
refusal representation and safe behaviour return. Conversely, strengthen refusal
while leaving the corrupted role unchanged. *"Differences between these
interventions provide evidence about causal ordering."*

## PLAN-ATTR — Component-level attribution

For MLP neuron `i` at layer `l`, `f_{i,l}(h) = a_{i,l}(h) · w_{i,l}^out`. For a
target security subspace `R`:

```
C_{i,l}^R(h) = || P_R f_{i,l}(h) ||_2      P_R = R (R^T R)^{-1} R^T
C_{i,l}^R    = E_{h~D} [ C_{i,l}^R(h) ]
```

Characterises components as role-writing, harm-writing, control-writing, or
**contributing to multiple security representations**. *"We compare these rankings
with NeuroStrike neuron rankings **and other attribution baselines**."*

## PLAN-ROLES — Detectors, transformers, writers, readers

*"A central goal is to avoid treating all 'safety neurons' as performing the same
function."*

| Role | Mapping |
|---|---|
| **Detector** | `x -> R_harm` — input features into a security variable |
| **Transformer** | `R_harm -> R_control` — one latent security variable into another |
| **Writer** | `f_i(h) -> R_control` — output contributes directly to the subspace |
| **Reader** | `R_control -> Y` — consumes a representation, produces logits/behaviour |

*"We identify these roles through intervention, activation patching, and
directional contribution analyses **rather than through correlation alone**."*

## PLAN-MED — Causal mediation and rescue

*"The most important component-level experiment."* If ablating component set `N`
decreases refusal, first measure `N↓ ⇒ R_control↓`. Then restore only the missing
component, `h_l' = h_l + P_{R_control} Δh_l`, or inject an independently estimated
representation. If safe behaviour returns while the neurons remain suppressed,
this evidences `N -> R_control -> Y`. **"Analogous experiments will be conducted
for role and harmfulness representations where applicable."**

## PLAN-MOE — Dense versus MoE security circuits

Decompose `h -> Router -> E_j -> R_security -> Y`. Intervene independently on:
router logits; expert selection; entire expert outputs; neurons within selected
experts; attention components; the corresponding security representation.

Distinguishes: the router selects a safety-specific pathway; the router selects
topic-specialised experts with safety emerging downstream; particular experts
write the refusal representation; experts encode harm while attention implements
refusal; several experts contribute redundant safety pathways.

*"The analysis will directly connect to GateBreaker by determining which
representation is disrupted when safety-relevant experts or expert-localized
neurons are suppressed."*

## PLAN-TAX — Mechanistic attack taxonomy

Attack set: indirect or direct prompt injection; adversarial jailbreaks;
NeuroStrike-style neuron suppression; GateBreaker-style expert/neuron
interventions. Per attack, measure changes in `R_role`, `R_harm`, `R_refusal` and
in the responsible components.

Objective: whether attacks producing the same observable failure — unsafe
compliance — do so through fundamentally different internal mechanisms. Output:

```
attack -> compromised security stage -> behavioral failure
```

*"Importantly, this taxonomy will be inferred experimentally rather than assumed
in advance."*

## PLAN-CONC — Security concentration

```
κ_R(k) = Σ_{i in TopK(C^R)} C_i^R / Σ_i C_i^R
```

Central prediction `κ_R(k) ↑ ⇒ sparse-attack vulnerability ↑`, tested as
`κ_refusal ↑ ⇒ NeuroStrike vulnerability ↑` for dense models and
`κ_expert ↑ ⇒ GateBreaker vulnerability ↑` for MoE. Separates **parameter
redundancy** from **security-function redundancy**.

## PLAN-HOPE — Functional operator analysis

Neurons as functional operators rather than independent parameter vectors: two
neurons may be parameter-different but function-similar, or parameter-similar and
contribute to distinct security behaviours. Complement `C_{i,l}^R` with
**function-space similarity** between candidate security-critical components, to
distinguish **parameter sparsity** from **functional sparsity**. NeuroStrike may
identify a small number of neurons that correspond to an even smaller number of
functional roles.

*"We use the functional-operator perspective conceptually and methodologically,
but do not require a direct implementation of the complete HOPE framework."*

## PLAN-STRUCT — Structural characterization

For each representation `R`, let `w_{i,R}` be the **causal contribution** of
component `i`, normalised `p_{i,R} = |w_{i,R}| / Σ_j |w_{j,R}|`.

```
Functional concentration   C_R      = Σ_i p_{i,R}²        N_eff,R = 1 / C_R
Effective functional rank  r_eff,R  = (Σ λ_i)² / Σ λ_i²
Causal redundancy          k_50^R   = min { k : A_R(k) <= 0.5 }
Context stability          S_R(c1,c2) = sim( M_R(c1), M_R(c2) )
```

`A_R(k)` is the fraction of the original security function remaining after
ablating the top-`k` components, **removed in order of causal importance**.
`S_R` is evaluated at representation and component levels; *"most importantly,
mechanisms identified under c1 are causally intervened upon under c2."*

## PLAN-CPE — Central prediction experiment

Architectural quantities are estimated **exclusively on representation and
causal-analysis datasets**. The architecture is then **frozen**. On **completely
disjoint attack intents**, apply: prompt-based jailbreaks and prompt injection;
representation steering; neuron-level suppression; expert silencing; routing
manipulation; optionally, small safety-removing adaptation.

```
k*     = min { k    : ASR(k) >= τ }
α*     = min { |α|  : ASR(α) >= τ }
```

Central prediction: `{C_R, N_eff, r_eff, k_50} -> {k*, α*}`.

*"The important question is not whether architecture and vulnerability correlate
retrospectively, but whether the former can predict the latter on unseen attacks,
intents, and eventually unseen models."* Leave-one-model-out where the roster
permits. Compare against simpler predictors: parameter count, baseline refusal
rate, probe accuracy, model family, architecture type, general utility.

## PLAN-CDRE — Context-dependent reorganization

Per intent, controlled variants `x^direct, x^paraphrase, x^role, x^authority,
x^jailbreak`. Per context estimate `R_harm(c)`, `R_control(c)` and their causally
important components `M_harm(c)`, `M_control(c)`. Three levels of stability:
representation `S_repr(c1,c2)`, component `S_comp(c1,c2)`, and **causal transfer**
`Effect( do(M_R(c1)) ; c2 )`.

*"A particularly important possible result is that the same high-level safety
representation remains stable while the computational components implementing it
change with context."*

## PLAN-RED — Proof-of-concept redundancy

Directional adapter `ΔW = R_refusal A`, `R_refusal` fixed and only `A` learned, so
the adapter learns **when** to activate a known behavioural subspace rather than an
unrestricted output direction. Explicitly not proposed as a new PEFT method.
Evaluate whether distributing refusal-writing capacity across layers or components
raises the minimum number of neurons or experts that must be disabled before
safety collapses.

## PLAN-INF — The informal experiment list

**Directions to extract**

| Concept | Paper |
|---|---|
| Harmfulness + refusal | Zhao et al., *LLMs Encode Harmfulness and Refusal Separately* — arXiv 2507.11878 |
| Role | *Prompt Injection as Role Confusion* — arXiv 2603.12277 |
| *Optional:* persona traits | Chen et al., *Persona Vectors* — arXiv 2507.21509 |
| *Optional:* emotions | Transformer Circuits, *Emotions* |

**Passive — geometry**
- Cosine similarity across all found vectors, all layers, all pairs. Looking for
  structure such as: at some layer, harmfulness is anti-aligned with the
  system-role direction.
- Projections onto these vectors across relevant prompt sets; plots plus
  correlations between projections.

**Passive — NeuroStrike / GateBreaker connection**
- Do the input/output directions of the relevant neurons align with any of the
  above vectors?
- Can the same safety neurons be recovered by measuring that alignment alone,
  without activation-difference profiling?

*Ideal results:* in specific layers, specific safety-relevant features align well;
we recover exactly the NeuroStrike neurons by a different route and classify which
concept each belongs to; expect this to differ per model.

**Active — how the features interact**
Fix the prompt. Steer with one vector (or toggle neurons) at an **early
layer/token**, then measure the effect on projections onto other concepts
downstream, **at later layers and later tokens**.
- Steer a benign user instruction toward the tool-output role. Does it become more
  harmful in later layers?
- Remove fear from a harmful prompt. Does harmfulness or refusal change?
- Switch the persona to a misaligned one. Is the harmful prompt still represented
  as harmful? Does refusal still fire?

## PLAN-SCOPE — Models, evaluation, scope control

**Models.** Depth of causal analysis over breadth. Initial target ~5–7 models
across ≥4 families: dense representatives from at least three families (e.g.
Llama, Qwen, Gemma) and at least two substantially different MoE architectures
with accessible routing and expert activations. Multiple sizes within one family
where feasible. *The scoped plan narrows the first pass to one dense + one MoE.*

**Evaluation, four dimensions.**
- **Behavioral security** — refusal/compliance, attack success rate, safety benchmarks
- **Utility** — standard utility and instruction-following evaluations, to verify
  interventions do not trivially destroy capability
- **Mechanistic consistency** — did the intervention produce the predicted change
  in the target representation or component?
- **Cross-model generalization** — do the causal and concentration–risk
  relationships hold across families and architectures?

**Explicitly out of scope.** Full HOPE adaptation; a general-purpose PEFT
framework; multimodal safety; multilingual safety; RAG-specific mechanisms; agent
memory security; backdoor attacks; secure code generation; a complete theory of
optimal security redundancy.

---

## Prioritized experimental plan (scoped)

Initially one representative dense model and one representative MoE model.

1. **Recover the latent variables** — identify and validate `R_role`, `R_harm`, `R_control`
2. **Establish causal separability** — the full cross-intervention matrix; hierarchy, parallel pathways, or more complex
3. **Mediation and rescue** — suppress known security-critical neurons/experts, determine which latent variable disappears, restore it without restoring the component, test whether safety returns
4. **Map functions onto computational components** — detectors, transformers, writers, readers at neuron and expert/routing levels
5. **Quantify architecture** — concentration, effective component count, effective functional rank, redundancy, **functional overlap**
6. **Run the critical prediction test** — does architecture measured before attack predict neuron-, expert-, routing- and representation-level intervention sensitivity?
7. **Expand across models** — only after the architecture–vulnerability relationship is established, extend to additional families and sizes
8. **Test context-dependent reorganization**
9. **Construct the mechanistic attack taxonomy**
10. **Optional redundancy intervention**

`main.tex` additionally declares its **first four stages the minimum viable core**
— recover the latent variables, cross-intervention matrix, connect NeuroStrike's
neurons to the representations, causal rescue — and notes that concentration and
cross-model prediction *"provide the strongest route toward a high-impact ICLR
contribution, while the redundancy-based defense should be included only if it
follows cleanly from the mechanistic findings."*

## Go/no-go criteria

```
GATE 1   R_role, R_harm, R_control are causally distinguishable
GATE 2   structural concentration/redundancy predicts intervention sensitivity
```

---

## Desired main result

Not merely that role, harmfulness and refusal correspond to different internal
representations, but that **LLM safety exhibits a measurable causal architecture
whose structural organization explains security vulnerability**. Concretely:

1. safety behaviour decomposes into causally distinguishable internal functions;
2. these are implemented through identifiable but **partially overlapping**
   computational pathways;
3. different attacks compromise different functions or pathways despite producing
   similar external unsafe behaviour;
4. these functions differ substantially in concentration, redundancy, effective
   dimensionality and context stability;
5. these structural properties predict how much intervention is required to
   disable safety.

> LLM safety is implemented through a structured causal architecture rather than a
> single refusal mechanism. Different security failures compromise different parts
> of this architecture, while its concentration and redundancy predict the ease
> with which alignment can be disabled.

*"An even stronger result would show that high-level safety functions remain stable
while their underlying computational implementation changes across contexts. This
would suggest that safety is a dynamically instantiated computational function
rather than a fixed set of 'safety neurons' or experts."*

---

## Undefined in the plan

Quantities the plan names but never defines. Each needs an operationalisation
decided in `EXPERIMENTS.md` and pre-registered before use.

| Quantity | Named in | Problem |
|---|---|---|
| **component diversity** | PLAN-RQ5 | Named, never defined |
| **pathway diversity** | PLAN-RQ5 | Named, never defined |
| `sim(·,·)` in `S_R` | PLAN-STRUCT | Similarity measure unspecified at both representation and component level |
| `A_R(k)` | PLAN-STRUCT | *"Fraction of the original security function remaining"* — the function's measure is unspecified (probe separation? projection magnitude? behaviour?), and `k_50` changes with the choice |
| `τ` | PLAN-CPE | *"A predefined attack-success threshold"* — value never fixed |
| *causal* `w_{i,R}` | PLAN-STRUCT | Declared causal, but the only component score the plan supplies (PLAN-ATTR's `C_{i,l}^R`) is a projection. The substitution is not licensed by the plan |

**Two concentration measures exist and are not the same.** PLAN-CONC's `κ_R(k)` is
top-`k` mass; PLAN-STRUCT's `C_R = Σ p²` is a Herfindahl index with
`N_eff = 1/C_R`. Report both. `κ` is the one the plan ties directly to NeuroStrike
and GateBreaker vulnerability.

**`functional overlap` is defined**, by PLAN-ATTR's *"components contributing to
multiple security representations"* — high `C_{i,l}^R` for more than one `R`.
