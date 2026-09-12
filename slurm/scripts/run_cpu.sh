#!/bin/bash
#SBATCH --job-name=cs_cpu
#SBATCH --output=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/logs/adhoc/out/%j-%x.out
#SBATCH --error=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/logs/adhoc/err/%j-%x.err
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#
# CPU-only stages: corpus construction (E1.0) and the analysis stages
# (E1.2-E1.5), which read cached activations and load no model weights.
#
#   sbatch slurm/scripts/run_cpu.sh experiments/e1_0_corpus.py
#   sbatch slurm/scripts/run_cpu.sh experiments/rq1.py --from nulls
#
# Login nodes cannot run even these: loading a few GB of cached activations is
# enough to trigger a policy SIGKILL. A GPU is still requested because the
# partition allocates whole nodes; nothing here uses it.

# NOTE: `--export=ALL,MODELS=a,b` does NOT work — sbatch splits --export on commas,
# so only the first model is passed and the rest are parsed as further assignments.
# Submit one job per model, or set MODELS in the environment before sbatch.
set -euo pipefail
EXPERIMENT="${1:?Usage: sbatch run_cpu.sh <path/to/experiment.py> [args...]}"
shift
EXTRA_ARGS=("$@")

module purge
module load craype-network-ofi
module load PrgEnv-nvidia
module load cuda/12.6
module load craype-arm-grace
module load craype-accel-nvidia90

export HF_HOME="/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate venv_causal_safety

cd /home/b6aj/jtelintelo.b6aj/SecurityControl

echo "=== $(date -Is) [cpu] ${EXPERIMENT} ${EXTRA_ARGS[*]:-} ==="
echo "RESULTS_ROOT=${RESULTS_ROOT:-./results} FAST_DEV=${FAST_DEV:-0} SEED=${SEED:-0}"
python "${EXPERIMENT}" "${EXTRA_ARGS[@]}"
echo "=== $(date -Is) done rc=$? ==="

conda deactivate
