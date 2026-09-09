# Slurm logs retained

Kept because each produced a live artifact. Job id -> what it produced;
the per-experiment `run_manifest.json` records only the last job for a
directory, so this is the mapping for multi-job pipeline runs.

| job | produced |
|---|---|
| `6421361` | E1.0 corpus, full scale -> results/e1_0_corpus (FROZEN) |
| `6422011` | E1.1 Stage 1 labels, qwen2.5-7b (full-generation Guard) |
| `6422012` | E1.1 Stage 1 labels, qwen3.5-9b (full-generation Guard) |
| `6422312` | E1.1 extraction, qwen2.5-7b |
| `6422314` | E1.1 extraction, qwen3.5-9b |
| `6426909` | E1.2-E1.4 analysis, qwen2.5-7b (emergence crashed; see 6426988) |
| `6426988` | E1.5 emergence, both models (after train_auc fix) |
| `6427076` | E1.2-E1.5 analysis, qwen3.5-9b |
| `6427102` | E1.1 fidelity + cross-corpus transfer, qwen2.5-7b |
| `6427103` | E1.1 fidelity + cross-corpus transfer, qwen3.5-9b |
| `6427824` | tests/test_invariants.py — 22/22 environment invariants |
| `6427895` | reproduction: E1.0 corpus -> results_verify |
| `6427897` | reproduction: full RQ1 pipeline, qwen2.5-7b |
| `6427899` | reproduction: full RQ1 pipeline, qwen3.5-9b |
| `6428149` | tests/verify_rq1_run.py on results/ — 16/16 |
| `6429332` | tests/verify_rq1_run.py, reproduction vs original (first pass) |
| `6430069` | E1.6 gate, qwen2.5-7b (generation readout) |
| `6430070` | E1.6 gate, qwen3.5-9b (generation readout) |
| `6430103` | download meta-llama/Llama-3.1-8B-Instruct |
| `6430148` | E1.0 render sweep, llama3.1-8b — FAILED (tool role quote-wrapping) |
| `6430264` | llama3.1-8b under-refusal probe (user role only) |
| `6430805` | E1.6 gate, qwen2.5-7b, CONTROL_VARIANT=over (matched cross-model test) |
| `6430807` | E1.5 emergence with onset CIs, qwen2.5-7b |
| `6430873` | E1.5 emergence with onset CIs, qwen3.5-9b |
| `6432187` | E1.4b behavioural k, qwen2.5-7b (with the behavioural-null guard) |
| `6432189` | E1.4b behavioural k, qwen3.5-9b (with the behavioural-null guard) |
| `6432742` | E1.7 Level 1 style vs metadata, qwen2.5-7b |
| `6432744` | E1.7 Level 1 style vs metadata, qwen3.5-9b |

Pruned 1 superseded and 1 empty log files.
Superseded runs are those whose outputs were withdrawn or
overwritten by a later fix; see docs/REMOVED_ARCHIVES.md.
