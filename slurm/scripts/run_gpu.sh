#!/bin/bash
#SBATCH --job-name=cs_gpu
#SBATCH --output=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/logs/adhoc/out/%j-%x.out
#SBATCH --error=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/logs/adhoc/err/%j-%x.err
#SBATCH --gpus=2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
# NOTE: `--export=ALL,MODELS=a,b` does NOT work — sbatch splits --export on commas,
# so only the first model is passed. Submit one job per model.
set -euo pipefail
EXPERIMENT="${1:?Usage: sbatch run_gpu.sh <experiment.py> [args...]}"
shift
module purge
module load craype-network-ofi PrgEnv-nvidia cuda/12.6 craype-arm-grace craype-accel-nvidia90
export HF_HOME="/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate venv_causal_safety
cd /home/b6aj/jtelintelo.b6aj/SecurityControl
echo "=== $(date -Is) [gpu] ${EXPERIMENT} $* ==="
echo "MODELS=${MODELS:-<default>} BATCH_SIZE=${BATCH_SIZE:-<default>} RESULTS_ROOT=${RESULTS_ROOT:-./results}"
python "${EXPERIMENT}" "$@"
echo "=== $(date -Is) done ==="
