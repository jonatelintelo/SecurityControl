# Methods

Draft of the Methods section. Prose is written to be lifted into LaTeX; bracketed
`[TODO]` markers flag values that must be filled from a final run rather than
quoted from the current dev-scale results.

Everything below is implemented in this repository; each subsection names the
module so a reader (or reviewer) can map claim to code.

---

## 1. Security variables and their estimation

We study three latent security variables: `R_role` (perceived role, authority or
origin of an instruction), `R_harm` (recognition of harmful content), and
`R_control` (the mechanism governing refusal versus compliance). Each is
estimated at every layer as a unit-norm difference-of-means direction over
contrastive prompt pairs, read from the residual stream at the final prompt
token.

### 1.1 Factorial prompt design

A naive design estimates each variable from its own prompt template. We found
this introduces systematic confounds that additional data cannot remove: an
earlier per-concept design placed the injected text only on the negative side of
the role contrast, producing a **16-token length imbalance in every pair**, so
the recovered `R_role` was partly a "long or adversarial text" detector; and the
three variables used structurally different templates (system prompt
present/absent, forced assistant token present/absent), so any cross-variable
geometry could reflect template differences rather than relationships between
security variables.

We therefore estimate all three variables from a single **crossed factorial
design** (`core/data.py: factorial_pairs`). Role, harm and control are varied
within one template, and each direction is the main effect of one factor with
the other two balanced:

| factor | positive level | negative level |
|---|---|---|
| role | untrusted tool-output channel | trusted operator channel |
| harm | harmful request | topic-matched benign request |
| control | forced assistant prefix `"I"` | forced prefix `"Sure"` |

The two role framings are **exactly length-matched** (16 tokens each under the
model's tokenizer), so the role contrast carries no systematic length signal.

Adversarial payload strings ("SYSTEM OVERRIDE", "ignore previous instructions")
are deliberately excluded from the `R_role` estimator and used only as attack
conditions (§5). Including them would define `R_role` as an adversarial-text
detector rather than the perceived-origin variable the hypothesis concerns.

### 1.2 Validation

Each direction is fit on a training split and validated on held-out pairs by
sign-classification accuracy (`core/subspaces.py: probe_validation_accuracy`).

Because `R_control` is estimated from pairs differing only in a forced final
token, it could in principle encode token identity rather than a control state,
and same-construction held-out validation cannot detect this. We therefore add a
**transfer test**: project *unforced* prompts, which contain neither `"I"` nor
`"Sure"`, onto `R_control` and measure separation between requests the model
refuses and complies with. Chance is 0.5.

### 1.3 Dimensionality

RQ1 asks about the dimensionality of each variable, which a 1-D difference-of-
means direction cannot address. We report the participation-ratio effective rank
of each concept's difference spectrum (`spectrum_effective_rank`).

**This estimator is bounded above by `n_pairs − 1`.** We report the ceiling
alongside every value, and do not compare `r_eff` across variables estimated
from different numbers of pairs, since the variable with more pairs has strictly
more headroom.

---

## 2. Component attribution

We map each security variable onto MLP neurons at a given layer along two
distinct axes, because a neuron may participate by *reading* a variable from the
residual stream or by *writing* to it:

- **write side** — the neuron's `down_proj` output column, projected onto `R`
- **read side** — the neuron's `gate_proj`/`up_proj` input row, aligned with `R`

Neurons high on the input side only are classified as **readers**, high on the
output side only as **writers**, and high on both as **transformers**
(`core/attribution.py`). This yields the detector/transformer/writer/reader
taxonomy the hypothesis requires; measuring the output side alone can only ever
identify writers.

We compute four rankings per (layer, variable) and report their overlap:

1. **causal attribution** — activation-weighted projection onto `R` (our primary ranking)
2. **static alignment** — pure `down_proj` weight geometry, no activations
3. **activation contrast** — unsupervised difference of mean activations
4. **input alignment** — read-side geometry

### 2.1 NeuroStrike comparison

To ask what functions previously-identified safety neurons perform, we port
NeuroStrike's white-box method faithfully (`core/neurostrike.py`): hooks on
`gate_proj`/`up_proj` outputs, max-pooling over the sequence, a logistic safety
probe, and neuron selection at `|z| > 3 ∧ w > 0`. Their neurons are
index-comparable to ours, since `down_proj` input *i* equals
`silu(gate_i)·up_i`. We additionally expose their probe weights as a *ranking*
(not only a thresholded set), so their signal can drive the same graded budget
axis our metrics require.

`[TODO: their probe requires thousands of training prompts; our current pool is
far smaller and the resulting weights are underdetermined. Report only from a
run with a scaled probe-training pool.]`

---

## 3. Structural characterization

For each variable `R` we compute, from the frozen attribution `w_{i,R}` with
`p_i = |w_i| / Σ|w_j|`:

- **functional concentration** `C_R = Σ p_i²`, and effective component count `N_eff = 1/C_R`
- **effective functional rank** `r_eff`, the participation ratio of the eigenvalues of the activation-scaled component Gram matrix
- **causal redundancy** `k_50 = min{k : A_R(k) ≤ 0.5}`

### 3.1 Defining `A_R(k)`

`A_R(k)` is the fraction of the security function remaining after ablating the
top-`k` components. It must be measured **per variable**: scoring an `R_role`
ablation by the refusal margin would measure the effect on `R_control`, a
different quantity.

We define `A_R` as the **class separation the direction achieves** — the
positive-class mean projection minus the negative-class mean projection — rather
than the positive class's raw projection. This matters: a raw residual
projection carries the stream's large mean offset and is not anchored at zero.
Measured that way, one variable's baseline was `+0.078`, and dividing by it made
`A_R` diverge (to 2.34 and −2.44) and spuriously trip the `k_50` threshold. The
separation is offset-free and positive at baseline by construction, since the
direction is fit to maximize exactly it. `A_R → 0` means the function is
destroyed; `A_R < 0` means it reversed.

The behavioral refusal-logit margin is recorded at the same grid points as a
cross-check, but never as `A_R` itself.

---

## 4. Behavioral evaluation

Attack success is judged by **Llama-Guard-3-8B**: a response counts as a
successful attack iff the guard returns "unsafe" **and** the response contains no
explicit refusal phrase, matching the reference protocol. Because a response
containing a refusal can never satisfy the conjunction, we short-circuit on the
keyword check and invoke the guard only on responses that survive it — the same
decision rule, evaluated in the cheaper order.

Reasoning models require care: Qwen3-family models emit a `<think>` block, and
with limited generation length the judge scores the model's reasoning preamble
rather than its answer, which silently inverts every behavioral measurement. We
disable thinking mode at the chat-template level, matching the reference
implementation's handling of the same model family.

---

## 5. Attack budgets

Structural quantities are estimated exclusively on the representation and
causal-analysis prompt pools, then **frozen**. Attack budgets are measured on a
**topically disjoint** pool of harmful intents; a runtime assertion enforces that
the pools never share a topic (`check_disjoint_topics`). This separation is what
makes the central test a prediction rather than a retrospective correlation.

- `k*` = smallest number of ablated neurons reaching `ASR ≥ τ`
- `alpha*` = smallest steering magnitude reaching `ASR ≥ τ`

### 5.1 Steering magnitude is expressed relative to the residual norm

Residual-stream norms grow strongly with depth (we measure ~8× from layer 10 to
21). A fixed absolute `alpha` is therefore a much larger *relative* perturbation
early in the network than late, and an apparent "early layers are more
vulnerable" result is what a norm artifact looks like. We express `alpha` as a
fraction of the residual norm at the steered layer, so budgets are comparable
across layers.

### 5.2 Attack budgets are capability-constrained

An intervention that destroys the model is not an attack. At large magnitudes a
**random** direction also reaches high ASR, because the output degenerates and
the judge scores incoherence as non-refusal. We therefore require that
capability survive: `k*` and `alpha*` are the smallest budgets reaching
`ASR ≥ τ` **while** benign-prompt capability retention remains above threshold
(`core/utility.py`). The constraint is part of the definition, not a filter
applied afterwards.

### 5.3 Attack-direction sign

Each direction is a difference of means, and the positive class is chosen for
narrative clarity per variable rather than uniformly as "the safe class". The
sign of `alpha` that compromises safety therefore differs by variable and by
design, and is derived programmatically from the active design rather than
hardcoded. This is not a detail: steering the wrong way searches the *defensive*
direction while reporting it as an attack, which makes a null result
unfalsifiable rather than false.

---

## 6. Context-dependent reorganization

For each harmful intent we construct five controlled variants — direct,
paraphrase, roleplay, authority, jailbreak. For every context `c` we
independently estimate `R_harm(c)` and `R_control(c)` and their causally
important components `M_harm(c)` and `M_control(c)`, and evaluate three levels
of stability:

- **representation** `S_repr(c1,c2)` — cosine similarity between directions
- **component** `S_comp(c1,c2)` — Jaccard overlap of top-`k` component sets
- **causal transfer** — ablate the components identified under `c1`, evaluate behavior under `c2`

Both variables are carried through all three levels, because the central
distinction — stable representation with dynamic implementation — requires
comparing representation- against component-stability **per variable**.

### 6.1 The within-context noise floor

Across-context component overlap is evidence of reorganization only if it is
lower than what two disjoint halves of the **same** context produce. Without
that floor, "components change with context" cannot be distinguished from "the
component ranking is noisy". We therefore split each context's intents in half,
re-estimate the mechanism independently from each half, and report

```
reorganization_gap = S_comp(within-context) − S_comp(across-context)
```

Both terms are computed **at matched sample size**. This is necessary rather
than cosmetic: comparing a half-sample floor against a full-sample across-context
value biases the floor downward, and in our data that mismatch inflated one
variable's gap by a factor of six.

We report bootstrap 95% confidence intervals on the gap. A CI straddling zero
means the gap is not distinguishable from estimation noise.

---

## 7. Controls

Every reported quantity is accompanied by a control that answers a specific
"could this be an artifact?" question. These are not robustness checks appended
after the fact; several of them changed the conclusions.

| Control | Question |
|---|---|
| Random ablation | Would ablating *any* `k` components have done this? `A_R(k)` is guaranteed to fall for the ranked condition, since components are ranked by their projection onto `R` and `R` is then measured. |
| Random direction | Would *any* perturbation of this magnitude have broken safety? |
| Random ranking | The same question for the component-budget attack. |
| Capability retention | Did safety fail, or did the intervention destroy the model? |
| Within-context split-half | Do components reorganize, or is the ranking simply noisy? |
| Transfer to unforced prompts | Is `R_control` a control variable, or the identity of a forced token? |
| Baseline ASR | Is the model robust, or is the attack pipeline incapable of breaking anything? |
| Bootstrap / Wilson intervals | How much of this is noise? |

`[TODO: consider reporting, as a methodological contribution, that a majority of
our initial headline findings did not survive these controls. This is unusually
concrete evidence for their necessity in representation-level safety work.]`

---

## 8. Reproducibility

Each phase writes a manifest recording the git commit, working-tree cleanliness,
configuration, and Slurm job ID alongside its outputs, so any reported figure is
traceable to the run that produced it. `tools/report_key_numbers.py` re-derives
every quoted value from stored artifacts.

**Caveat to address before submission:** when the working tree is dirty the
commit hash does not uniquely identify the code that ran. Runs used for
reported results must be made from a clean tree, or the manifest extended to
hash source files.

---

## Open methodological items

1. `w_{i,R}` is a **geometric proxy** (activation × weight projection), not an
   ablation-derived causal contribution. Either the attribution is made causal,
   or the paper states plainly that it is a proxy.
2. `C_R`, `N_eff` and `k_50` are computed against a **1-D projector**, while our
   dimensionality estimates suggest the variables are not 1-D.
3. With a single model, the central test is a **within-model** correlation over
   non-independent (layer, variable) rows, not the leave-one-model-out
   evaluation the design calls for.
4. **No activation patching.** We manipulate representations by additive
   steering; donor-activation patching is not implemented.
5. **Attention components and MoE experts/routing are not instrumented.**
