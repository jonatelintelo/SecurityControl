#!/bin/bash
# Post-run pass for the blank-slate rerun: verification first, then the
# derived analyses and figures.
#
# Order matters. Verification runs BEFORE the tools, so that if the run is
# invalid we find out from an exit code rather than from a plausible-looking
# figure. Every tool here reads `results/` and writes either stdout (captured
# into the Slurm log) or a file under `results/`.
#
#   sbatch slurm/scripts/run_cpu.sh slurm/scripts/blank_slate_post.sh
# or, since it is plain bash rather than python, submit it directly:
#   sbatch --export=ALL slurm/scripts/blank_slate_post.sh
#SBATCH --job-name=cs_post
#SBATCH --output=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/1_run_phase/out/%j-%x.out
#SBATCH --error=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/1_run_phase/err/%j-%x.err
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
set -uo pipefail   # NOT -e: a failing check must not hide the checks after it

module purge
module load craype-network-ofi PrgEnv-nvidia cuda/12.6 craype-arm-grace craype-accel-nvidia90
export HF_HOME="/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate venv_causal_safety
cd /home/b6aj/jtelintelo.b6aj/SecurityControl

FAILED=0
step() {
  echo; echo "################################################################"
  echo "# $1"; echo "################################################################"
  shift
  "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo ">>> EXIT $rc"; FAILED=$((FAILED+1)); }
  return 0
}

echo "=== $(date -Is) blank-slate post pass ==="

# ---------------------------------------------------------------- verification
step "verify results/ (standalone)" \
  env RESULTS_ROOT=./results MODELS=qwen2.5-7b,qwen3.5-9b,qwen3.5-35b-a3b python tests/verify_rq1_run.py
step "verify results/ against the independent reproduction" \
  env RESULTS_ROOT=./results VERIFY_AGAINST=./results_verify MODELS=qwen2.5-7b,qwen3.5-9b,qwen3.5-35b-a3b python tests/verify_rq1_run.py
step "reproduction diagnostic (per-concept AUC drift)" \
  python tools/repro_diag.py

# ---------------------------------------------------------------- completeness
step "walkthrough: every experiment, artifact, verification" \
  python tools/rq1_walkthrough.py
step "diff against the pre-rerun archive (what moved, and by how much)" \
  python tools/diff_against_archive.py

# ---------------------------------------------------------------- derived
step "E1.6 gate comparison across control variants" python tools/gate_compare.py
step "E1.6 asymmetry structure" python tools/asymmetry_structure.py
step "E1.1d harm/control cross-check (layer-matched)" python tools/harm_control_crosscheck.py
step "E1.7 style decomposition" python tools/style_decompose.py
step "Guard filter impact on the 2x2" python tools/guard_filter_impact.py
step "O-1 label audit sheet" python tools/build_label_audit.py

# ---------------------------------------------------------------- figures
step "figures" python tools/make_figures.py

# ---------------------------------------------------------------- last, and slow
# ~3.1s per adjudication on a ~23k-row matrix x 432 settings x several matrices
# = ~90 min. Deliberately final: everything above is cheap and must not be
# starved by it, and if this alone times out it re-runs standalone on the saved
# matrices without repeating any GPU work.
step "E1.6 gate sensitivity sweep (slow: ~90 min)" python tools/gate_sensitivity.py

echo
echo "================================================================"
echo "post pass complete: $FAILED step(s) exited non-zero"
echo "=== $(date -Is) ==="
exit $FAILED
