# Activation caches pruned

`activations.pt` removed from this root to reclaim disk. It is a
**cache, not a result**: regenerate with

    RESULTS_ROOT=./results_verify MODELS=<slug> \
        sbatch slurm/scripts/run_gpu.sh experiments/rq1.py --only extract

Extraction is deterministic — an independent re-run reproduced every
label-independent direction at max |delta AUC| = 0.000000 — so the
regenerated cache is identical for those concepts. Every CSV/JSON
artifact derived from it is retained here unchanged.
