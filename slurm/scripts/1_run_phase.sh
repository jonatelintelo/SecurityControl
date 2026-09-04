#!/bin/bash
#SBATCH --job-name=qwen_causal_phase1
#SBATCH --output=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/1_run_phase/out/%j-%x.out
#SBATCH --error=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/1_run_phase/err/%j-%x.err
#SBATCH --gpus=2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16

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
if [ -n "$HF_TOKEN" ]; then
    huggingface-cli login --token "$HF_TOKEN"
fi

# 4. Activate Conda Environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv_causal_safety

# Create log directory if it doesn't exist
mkdir -p /scratch/b6aj/jtelintelo.b6aj/logs

# 5. Run Phase 1 Causal Architecture Script
python /home/b6aj/jtelintelo.b6aj/SecurityControl/1_run_phase.py

conda deactivate