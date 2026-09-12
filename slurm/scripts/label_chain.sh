#!/bin/bash
#SBATCH --job-name=cs_label
#SBATCH --gpus=2
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
set -uo pipefail
module purge
module load craype-network-ofi PrgEnv-nvidia cuda/12.6 craype-arm-grace craype-accel-nvidia90
export HF_HOME="/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate venv_causal_safety
cd /home/b6aj/jtelintelo.b6aj/SecurityControl
F=0
step(){ echo; echo "### $1"; shift; "$@"; rc=$?; [ $rc -ne 0 ] && { echo ">>> EXIT $rc"; F=$((F+1)); }; return 0; }
step "O-1 build blind audit sheet"        python tools/build_label_audit.py
step "O-1 adjudicate (2 judges)"          python tools/adjudicate_labels.py
step "O-1 score against the adjudication" python tools/score_label_audit.py
step "O-1b adjudicate undetermined pool"  python tools/adjudicate_undetermined.py
step "sensitivity: refit R_control"       python tools/refit_control_labels.py
echo; echo "label chain complete: $F step(s) non-zero"; exit $F
