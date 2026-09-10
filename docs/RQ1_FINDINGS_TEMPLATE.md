# RQ1 — findings

> **Scaffold, not results.** Every `<<...>>` is filled from the named artifact of the
> single frozen run. Nothing is written here that is not read out of an artifact, and
> **no experiment is added during the write-up** — anything new becomes an open issue
> for RQ2. That rule is what makes this terminate.

**Run identity.** Commit `<<git_commit>>`, `run_manifest__<model>.json` per model,
`results/` primary and `results_verify/` reproduction at identical configuration.
Roster: `qwen2.5-7b`, `qwen3.5-9b` (dense, one family) and `qwen3.5-35b-a3b` (MoE).

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
- **One dense family.** Two Qwen models plus a Qwen MoE. Cross-family is deferred by the
  scoped plan (PEP item 7); Llama-3.1 is sequenced after this run.
- **O-1: the refusal label is an unvalidated instrument.** `results/label_audit/` is
  built and awaiting a human pass. Every `R_control` claim inherits this.
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
