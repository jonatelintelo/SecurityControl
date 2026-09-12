# RQ1 — findings

> **Scaffold, not results.** Every `<<...>>` is filled from the named artifact of the
> single frozen run. Nothing is written here that is not read out of an artifact, and
> **no experiment is added during the write-up** — anything new becomes an open issue
> for RQ2. That rule is what makes this terminate.

**Run identity.** Commit `<<git_commit>>`, `run_manifest__<model>.json` per model,
`results/` primary and `results_verify/` reproduction at identical configuration.
Roster: six models, four vendors, both architectures — `qwen2.5-7b`, `qwen3.5-9b`,
`llama3.1-8b`, `yi-6b-chat` (dense) and `qwen3.5-35b-a3b`, `nemotron-3-nano-30b-a3b`
(MoE).

---

## The question

*Is LLM safety functionally decomposable — are role perception, harmfulness
recognition, and refusal/compliance control causally distinguishable latent
variables?* Four pillars: geometry, dimensionality, emergence across layers, causal
interactions, **without assuming a fixed sequential architecture**.

## Answer in one paragraph

<<Written last, from C1–C6. State the GATE 1 verdict and its bound-sensitivity
plainly, including if it is FAIL or bound-sensitive — a negative gate is a result,
not a failure of the run.>>

---

## C1 — the three variables are recoverable and validated
`direction_validation.csv`, `null_distributions.csv`, `role_probe.csv`, `fidelity.json`

Per model and concept: held-out AUC with cluster-bootstrap CI, **length-only baseline**,
split-half stability, random-direction null. Layer selected on **train**.
`R_role` reported from both estimators the plan requires — the multiclass probe and the
activation-space contrast.

### C1b — WHAT `R_control` ENCODES: explicit refusal, not compliance broadly
`soft_refusal_projection.csv`, `soft_refusal_axis.csv`, `soft_refusal_summary.json`,
`label_audit/undetermined_adjudicated.csv`

**State this; do not let a reader infer it from the variable's name.**

`R_control` is fitted as `refused` vs `complied` on harmful prompts, where
`refused` means the response contains a prefix from Arditi's list. On every model
a third group exists and is excluded from the fit: harmful prompts where the rule
found no refusal marker but Llama-Guard, reading the full generation, found no
harmful content. **The degeneracy gate fires on none of them** — 22% of Qwen2.5's
harmful side, and reading them shows soft refusals ("that is illegal and
unethical... instead, let's...").

Because they are held out, carry no refusal marker, and are read at `t_post_inst`
(the prompt's last token, so response length and truncation cannot reach the
activation), where they project is a clean test of what the direction encodes:

**This conclusion reversed, and the reversal is the point.** Projecting the whole
`undetermined` pool put it near the compliance pole on every model, and
`soft_refusal_summary.json` still records that reading as *"MARKER DETECTOR"*.
**That artifact is superseded** — the inference assumed the pool was soft refusals.
It is not. Adjudicated by two off-roster judges, it splits **847 `complied` / 513
`refused`** (plus 843 the judges disputed). A pool that is 62% compliance projecting
with the compliances is what a *correct* `R_control` does, and says nothing about
markers.

Splitting the pool by what the readers said each item *is* (`soft_refusal_split.csv`,
both judges agreeing, at `t_post_inst`):

| model | judge=refused: position / AUC vs complied | judge=complied: position / AUC |
|---|---|---|
| qwen2.5-7b | **0.60** / **0.912** | −0.11 / 0.431 |
| qwen3.5-35b-a3b | **0.61** / **0.986** | 0.04 / 0.565 |
| nemotron-3-nano-30b-a3b | **0.30** / **0.760** | 0.05 / 0.569 |
| yi-6b-chat | **0.26** / **0.674** | −0.08 / 0.435 |
| llama3.1-8b | *too few items to fit* | 0.03 / 0.551 |

**`R_control` separates reader-confirmed soft refusals from compliances** — a gap of
+0.25 to +0.71 in position, AUC 0.67–0.99 against 0.43–0.57 — on held-out items that
carry **no refusal marker at all**. It is therefore a **control variable, not a
vocabulary detector**: the missing marker does not fool it.

*Report the reversal, not just the conclusion.* The first version of this test is
what the pooled projection looked like, and pooling two populations that a reader
can tell apart is the error. `soft_refusal_probe.py` produces the superseded
artifact and `soft_refusal_split.py` the decisive one; the tools say so in their
own docstrings.

**Residual caveat, unchanged:** `llama3.1-8b` has too few judge-confirmed soft
refusals to fit, so the claim rests on five of six models.

## C2 — `R_harm` is separable from refusal behaviour
`harm_controls.csv`, `harm_controls_geometry.csv`

The PLAN-EXTRACT contrast: `R_harm` with refusal held constant, in both directions
(`in_refused`, `in_complied`). Role-balanced and length-baselined — **an unbalanced fit
carries a role component that depresses its cosine against a role-balanced `R_control`
and overstates separability**. Cosines layer-MATCHED only.

## C3 — geometry
`geometry_cosines.csv` (pass 1), `geometry_subspace.csv` (pass 2)

Pass 1: all pairs × all layers against the random null band and the split-half floor.
Pass 2 at E1.4b's measured `k`: principal angles, projection metric, data-weighted CCA
against a **matched random-subspace null** (a rank-1 band is the wrong reference).
If every `k = 1`, state the degeneracy explicitly and defer to pass 1.

## C4 — dimensionality
*(see also C1b: the soft-vs-hard refusal contrast is NOT a second axis of `R_control` — |cos| ~0.95 with it, far outside the random band.)*
`dimensionality.csv` (spectral `r_eff`), `behavioural_k.json` (smallest `k` reproducing
the intervention effect). The behavioural `k` is the one `main.tex` asks for; report both
and say where they disagree.

## C5 — emergence and persistence
`emergence_summary.csv`, `emergence_curves.csv`. Onset depth with bootstrap CI;
conclusions must hold across {80, 90, 95}%. Cross-model comparison is of **relative**
depth profiles, never absolute layer indices.

## C6 — causal distinguishability → GATE 1
`causal_matrix*.csv`, `causal_gate*.json`, `gate_sensitivity.csv`

3×3 cross-intervention, α-matched nulls, BH-FDR. G1 is a sanity check and is **not**
evidence; the verdict is G2 ∧ G3 across a swept capability bound.
Report the verdict under **both** `gate_layers` arms (`all`, `relative3`) — that choice
is swept, not argued.
Also report the **depth and token profile** descriptively: every layer downstream of the
steer layer is read, and both `t_inst` and `t_post_inst`.

### Double dissociation
Both halves from the same matrix: steering `R_control` should raise refusal on
**harmless** prompts (`d_refusal_harmless`); steering `R_harm` should move the harm
readout without necessarily forcing refusal. State it explicitly.

### Steered-position profile (Stage B)
`causal_stage_b*.csv` — α × 4 token sets on train. Which tokens carry the intervention.

## C7 — role is metadata, not linguistic style
`style_vs_metadata.csv` (Level 1), `style_level2.csv` (Level 2)

**Appendix, not a narrative pillar** — E1.6 is the gate; this is a robustness check on
C1. Level 2 crosses register with tag on a frozen generated corpus; the template arm is
a control that upper-bounds how detectable register can be.
State the corpus limitation: lexical verification cannot separate a faithful paraphrase
from a changed request; the residual is bounded by within-base pairing, the verbatim
template arm, and the **reported coverage distribution**.

## C8 — dense and MoE
The same artifact set for `qwen3.5-35b-a3b`. Its harmful-and-complied cell is too thin
for the under-refusal variant, so it uses `R_control_harmless`, as does `qwen3.5-9b`;
the matched cross-model comparison is the `over` variant on all three.

## C9 — literature anchor
`zhao_replication.csv` — the `advbench|alpaca` contrast swept layer-wise.
`published_auc` **must be read off Zhao et al. (2507.11878), not recalled**.

## C10 — limitations, stated not buried
- **Four vendors, not more.** Qwen (x3), Meta, 01-AI, NVIDIA — six models, both
  architectures. Related work for calibration: the role paper (2603.12277) used 10
  models over 5 families, Arditi (2406.11717) 13 over 5, Zhao (2507.11878) 3 over 2.
  We sit above Zhao, below the two larger studies. Say the numbers; do not imply
  broader coverage.
- **Count families as VENDORS (4), not template families (5).** `qwen2.5` and
  `qwen3.5` are separate template families but the same organisation. Using 5 would
  overstate the cross-family claim.
- **No chain-of-thought role class.** The role paper uses five role classes
  including CoT; we use four. Not an oversight and not a cost decision: on every
  roster model `cot` is either silently dropped or rendered as a tag the model was
  never trained on (see EXPERIMENTS.md). CoT is a *channel inside an assistant
  turn*, not a role. State this explicitly — it is the most likely place a reader
  familiar with 2603.12277 will expect a five-way comparison.
- **`t_inst` template bleed on the `tool` role.** On Qwen2.5, Llama-3.1 and
  Nemotron-3, the read token absorbs one template character (`\n`, or `"` on
  Llama, which quotes tool content), and it does so **more often on harmless items
  than harmful ones** (~0.81 vs ~0.31) because the sources differ in final
  punctuation. Bounded (tool role only, ≤1 char) and verified; controlled by the
  surface baseline `R_harm` must beat. Report the rates — do not omit them.
- **O-1: the refusal label is validated only by AUTOMATED instruments.**
  `results/label_audit/` is adjudicated by **three** off-roster readers, each blind
  to the rule label: `google/gemma-3-27b-it`, `Mixtral-8x7B-Instruct-v0.1` and a
  Claude session. (`openai/gpt-oss-20b` was the intended judge and could **not** be
  loaded — its MXFP4 quantiser calls `torch.accelerator`, which needs torch >= 2.6
  against our pinned 2.5.1.) Report agreement and Cohen's kappa from
  `adjudicator_agreement.csv`. The readers agree with **each other** (kappa
  0.67-0.89) far better than with the prefix rule (0.54-0.71), and the pipeline's
  final `rule+Guard` label agrees with gemma at 0.907 and with Claude at 0.868.
  Report it as what it is: **not a human audit**. A language model judging language-model
  outputs can fail where the rule fails — hedged preambles, partial compliance,
  refuse-then-answer — so agreement bounds the instrument *less* tightly than a
  human pass would. Disagreements are fully informative and should be read.
  **O-1 remains open** until a person reads a sample; `adjudicator=human:<name>`
  rows can be mixed into the same sheet and are scored separately.
  Every `R_control` claim inherits this.
- Control-family CIs are conditional on the labels; greedy decoding is bit-reproducible
  only at fixed batch size, so run-to-run label drift adds uncertainty on top.
- Attention-component analysis is RQ3 and absent here; no claim in this document is a
  circuit-level claim.

---

## Reproduction
`tests/verify_rq1_run.py` standalone and with `VERIFY_AGAINST`; corpus and style-corpus
byte-identity; label-independent concepts EXACT; label-dependent within tolerance.

## What moved since the previous run
`tools/diff_against_archive.py` — MATCH / DRIFT / NEW / GONE per tracked quantity.
**Every DRIFT must be explainable by a specific change; an unexplained one is a bug.**
