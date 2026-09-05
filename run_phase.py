#!/usr/bin/env python
"""Entry point: `python run_phase.py <n>` runs one phase of the causal-
architecture-of-safety pipeline. Phases share core/ and read the previous
phase's frozen outputs from results/<phase_name>/ (see each phase module's
docstring for its inputs/outputs). Config comes entirely from environment
variables (see core/config.py) — e.g.:

    MODEL_ID=Qwen/Qwen3.5-9B python run_phase.py 1
    FAST_DEV=1 MODEL_ID=Qwen/Qwen2.5-7B-Instruct python run_phase.py 1   # CPU smoke test
"""
import argparse
import os

if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and not os.environ.get("SLURM_JOB_ID"):
    # CPU dev/smoke runs on shared HPC login nodes often have a low ulimit -u
    # (max user processes) that torch's default OMP thread pool (= CPU count)
    # can blow past, segfaulting via libgomp thread-creation failures. Must be
    # set before torch's native libs initialize, i.e. before any import that
    # pulls in torch. GPU Slurm jobs (SLURM_JOB_ID set) are left untouched.
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")

from core.config import load_config
from phases import (
    phase1_geometry,
    phase2_causal_structure,
    phase3_components,
    phase4_architecture_prediction,
    phase5_context_and_attacks,
    phase6_feature_interactions,
)

PHASES = {
    1: phase1_geometry,
    2: phase2_causal_structure,
    3: phase3_components,
    4: phase4_architecture_prediction,
    5: phase5_context_and_attacks,
    6: phase6_feature_interactions,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one phase of the causal-architecture-of-safety pipeline.")
    parser.add_argument("phase", type=int, choices=sorted(PHASES), help="Phase number to run (1-6).")
    args = parser.parse_args()

    cfg = load_config()
    PHASES[args.phase].run(cfg)


if __name__ == "__main__":
    main()
