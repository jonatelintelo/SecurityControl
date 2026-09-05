#!/bin/bash
#SBATCH --job-name=causal_safety_phase
#SBATCH --output=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/1_run_phase/out/%j-%x.out
#SBATCH --error=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/1_run_phase/err/%j-%x.err
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#
# Usage: sbatch run_phase.sh <phase_number>
#   e.g. sbatch run_phase.sh 1
# Requires phase1_geometry's results/ directory to already exist on shared
# storage before running phases 2-6 (each phase reads the previous phase's
# frozen outputs).

set -euo pipefail
PHASE_NUM="${1:?Usage: sbatch run_phase.sh <phase_number 1-6>}"

# 1. Load System Modules
module purge
module load craype-network-ofi
module load PrgEnv-nvidia
module load cuda/12.6
module load craype-arm-grace
module load craype-accel-nvidia90

# 2. Environment Variables & Caches
export HF_HOME="/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir"
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER="PCI_BUS_ID"

# 3. Hugging Face Authentication
# export HF_TOKEN="your_new_token_here" prior to running, or set via ~/.bashrc
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

# 4. Activate Conda Environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate venv_causal_safety

# 5. Run the requested phase
export MODEL_ID="${MODEL_ID:-Qwen/Qwen3.5-9B}"
cd /home/b6aj/jtelintelo.b6aj/SecurityControl
python run_phase.py "$PHASE_NUM"

conda deactivate
