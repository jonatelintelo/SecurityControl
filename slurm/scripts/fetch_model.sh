#!/bin/bash
#SBATCH --job-name=fetch
#SBATCH --output=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/logs/adhoc/out/%j-%x.out
#SBATCH --error=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/logs/adhoc/err/%j-%x.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
set -euo pipefail
MODEL="${1:?usage: sbatch fetch_model.sh <hf-model-id>}"
module purge; module load craype-network-ofi PrgEnv-nvidia cuda/12.6 craype-arm-grace craype-accel-nvidia90
export HF_HOME="/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir"
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate venv_causal_safety
echo "fetching $MODEL"
python -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL', allow_patterns=['*.json','*.safetensors','*.model','tokenizer*'])
print('done ->', p)
"
