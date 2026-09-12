# Slurm logs kept as evidence

Everything else from the exploratory runs was deleted; these three are cited by
decisions recorded in the code and docs, so they are kept where the claim can be
checked against the raw output.

| log | what it is evidence for |
|---|---|
| `6459301-cs_gpu.err` | The NVIDIA-Nemotron labels-gate failure that exposed the **typographic-apostrophe bug**. Shows the rule labelling `"I’m sorry, but I can’t help with that."` as `complied`, Guard reclassifying 1527/1543, and the gate failing on `refusal_variance` + `undetermined_rate`. The fix is `core.refusal.normalise_typography`, with a regression test in `tests/test_estimators.py`. |
| `6461721-cs_gpu.err` | Jailbreak-framing probe, qwen3.5-9b. Persuasion styles moved `undetermined` 3 -> 45-99 but compliant instructions 0 -> 1. |
| `6461722-cs_gpu.err` | Same probe, nemotron-3-nano. `undetermined` 24 -> 120-160, compliant instructions 0 -> 8 of 60. |

The two probe logs are now the **primary** evidence for the decision NOT to widen
the corpus with jailbreak framings: the run's `DECISION.json` lived under a
transient `results_probe/` root that the blank slate removed, so these logs are
where the claim is checked. Both probe jobs ended
in a `KeyError` at the summary step (`df.style` is the pandas Styler property,
not the column); the generation and labelling completed and the CSVs were
written before that, so the data is intact. Fixed in the script.
