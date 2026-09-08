#!/bin/bash
#SBATCH --job-name=causal_safety
#SBATCH --output=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/1_run_phase/out/%j-%x.out
#SBATCH --error=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/1_run_phase/err/%j-%x.err
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#
# Run one experiment from EXPERIMENTS.md on GPU.
#
#   sbatch run_experiment.sh experiments/e1_1_directions.py
#   FAST_DEV=1 MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct sbatch run_experiment.sh experiments/e1_1_directions.py
#
# Login nodes kill CPU-heavy processes (observed: SIGKILL shortly after model
# load, with 119 GB free — a policy kill, not OOM), so everything runs here.

set -euo pipefail
EXPERIMENT="${1:?Usage: sbatch run_experiment.sh <path/to/experiment.py>}"

module purge
module load craype-network-ofi
module load PrgEnv-nvidia
module load cuda/12.6
module load craype-arm-grace
module load craype-accel-nvidia90

export HF_HOME="/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate venv_causal_safety

cd /home/b6aj/jtelintelo.b6aj/SecurityControl

# Environment passes through: MODEL_ID, FAST_DEV, RESULTS_ROOT, SEED, N_HARMFUL,
# N_HARMLESS, BATCH_SIZE, LAYER_STRIDE, MAX_CONTENT_TOKENS. See core/config.py.
echo "=== $(date -Is) running ${EXPERIMENT} ==="
echo "MODEL_ID=${MODEL_ID:-<default>} FAST_DEV=${FAST_DEV:-0} RESULTS_ROOT=${RESULTS_ROOT:-./results}"
python "${EXPERIMENT}"

conda deactivate
