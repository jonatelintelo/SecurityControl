# What counts as dead code (RQ1 cleanup)

Agreed before the cleanup phase, so the decision is not re-argued file by file.

## Keep

A file or function is **live** if it does any of:

1. **Builds** an RQ1 result — corpus, labels, directions, matrices, gates.
2. **Verifies** one — the verifier, the mutation suite, the estimator tests, the
   independent-recomputation audit, the static checks.
3. **Reports** one — status readouts, figures, the walkthrough, the analysis tools.
4. **Justifies a decision we made.** Code that exists only to produce the
   evidence for a methodological choice is NOT dead. If a reviewer asks "how do
   you know jailbreak framings wouldn't have helped?", the answer has to be
   reproducible, not a claim.

   Concretely: `experiments/probe_jailbreak_corpus.py` is the evidence for not
   widening the corpus with reserved RQ4 material. Its `DECISION.json` lived under
   a transient `results_probe/` root that the blank slate removed; the surviving
   primary evidence is the two job logs kept in `slurm/evidence/`
   (`6461721`, `6461722`), and the script re-runs to regenerate the rest.
   `tools/control_learning_curve.py` produced the n~30 plateau that sets the
   reporting floor. Both stay.

## Delete

Everything else, specifically:

* **Superseded duplicates.** Two tools doing one job — e.g. `tools/readjudicate.py`
  and `tools/readjudicate_gates.py`, written months apart without noticing the
  first existed. One is dead by definition; confirm which is wired in and remove
  the other.
* **Unused functions inside `core/`.** This is where most of it will be: nothing
  flags a function that is defined and never called, so they accumulate silently.
* **Code for an experiment that no longer exists** in the form it was written for.

## Not a consideration

There is no RQ2-RQ7 implementation to protect — `experiments/` is RQ1-only, and
later RQs appear solely in comments explaining why a resource is reserved (the
attack pool, the withheld Sorry-Bench styles). So the criterion can be applied
strictly without deleting work that would have to be rewritten.

## How

Produce the inventory and get it approved BEFORE deleting. Deletion is hard to
reverse, and "unreferenced by the job scripts" is not the same as "dead" — several
RQ1 tools are run by hand and would fail that test wrongly.


---

# Sequence (agreed)

1. **FULL TEST** — the six-model run and post pass. Done.
2. **BUG HUNT** — independent recomputation across all six models and both roots;
   the verifier on all six; the mutation suite; plus two items this session
   surfaced:
   * the **E+D projection split by judge label** — DONE, and it **reversed the
     finding**. The claim "the `undetermined` pool is soft refusals" came from four
     hand-read examples and was too strong: adjudicated, the pool is 847 `complied`
     / 513 `refused`. Re-run on the two subsets separately
     (`tools/soft_refusal_split.py`), the judge-confirmed refusals project toward
     the REFUSED pole (AUC 0.67-0.99 vs 0.43-0.57 for the judge-confirmed
     compliances). `R_control` is a control variable, **not** a marker detector;
     `soft_refusal_summary.json`'s "MARKER DETECTOR" verdict is superseded.
   * the **shadowing class of bug** — `g.tail`, `df.style` and `d` (a `Path`
     rebound to a `Direction`) all shipped this session and all failed only at
     runtime. `check_names.py` cannot see them: it resolves bare names, not
     attribute access, which is also why `model_meta.decoder_layers` (a function
     that does not exist) reached a job.
3. **DEAD CODE CLEANUP** — the criterion above.
4. **DOCS CLEANUP** — done. `EXPERIMENTS_v1_superseded.md` deleted (85 KB,
   git-tracked, produced and verified nothing); `docs/RQ1_MISSING_WORK.md`
   rewritten from a list of open gaps into a closure record, since all three
   gaps and both scope limits had shut; `README.md` and `EXPERIMENTS.md` had ten
   dangling references to a `RQ1_FINDINGS.md` that was never written, plus a
   directory tree naming deleted tools — all repointed at what exists.
5. **RQ2.**

## Deferred deliberately

**Reporting decisions** — what goes in the 9-page main body versus the appendix,
including whether the `under` arm and the Zhao comparison appear at all. That
depends on what RQ2-RQ4 need, and deciding it now would be premature. Keep every
arm in the artifacts with its `n` recorded so the choice stays a reporting one
rather than a re-run.
