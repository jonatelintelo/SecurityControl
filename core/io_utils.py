"""Small shared helpers: logging setup and result-file save/load.

Every phase writes into results/<phase_name>/ using these helpers so later
phases can load an earlier phase's frozen outputs instead of recomputing them.
"""
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch


def get_logger(name: str, log_dir: Path, suffix: str = "") -> logging.Logger:
    """`suffix` distinguishes concurrent jobs writing into the same directory.

    Models run as separate Slurm jobs (sbatch splits `--export` on commas, so one
    job cannot carry two model slugs), and every one of them opens this log in
    truncate mode in the shared `results/<experiment>/`. Without a per-job
    suffix the last job to start silently erases the others' logs.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name + suffix)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    file_handler = logging.FileHandler(log_dir / f"{name}{suffix}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def save_torch(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_torch(path: Path, mmap: bool = False) -> Any:
    """Load a torch artifact.

    `mmap=True` maps tensors lazily instead of reading the whole file into RAM.
    Use it for the ACTIVATION BLOB, which is 1.3-1.8 GB per model: an analysis
    tool that opens one and then runs over the roster needs gigabytes it does
    not use, and is simply OOM-killed on a login node — measured, not
    hypothetical (0.25 GB resident with mmap vs 1.3 GB without, on Qwen2.5).

    Not the default. Memory-mapped tensors are read-only and are backed by the
    file for the object's lifetime, which is right for read-only analysis and
    wrong for anything that mutates or outlives the file. `directions.pt` also
    carries custom objects rather than plain tensors, so it is loaded normally.
    """
    return torch.load(path, map_location="cpu", mmap=True) if mmap else \
        torch.load(path, map_location="cpu")


def save_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_run_manifest(cfg, experiment_name: str) -> Path:
    """Record the exact code state and configuration that produced a phase's
    outputs, alongside those outputs.

    Phases run as separate Slurm jobs and can therefore be produced under
    different code states; mixing them silently is a real failure mode (it has
    already caused stale numbers to be quoted as current). A per-phase manifest
    makes any figure traceable to the run and commit that produced it.
    """
    import subprocess
    from dataclasses import asdict
    from datetime import datetime, timezone

    def _git(*args: str) -> Optional[str]:
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10,
                                  cwd=Path(__file__).resolve().parent.parent).stdout.strip() or None
        except Exception:
            return None

    cfg_dict = cfg.as_dict() if hasattr(cfg, "as_dict") else {
        k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()
    }
    manifest = {
        "experiment": experiment_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": os.environ.get("HOSTNAME") or os.uname().nodename,
        "config": cfg_dict,
    }
    out_dir = cfg.dir(experiment_name) if hasattr(cfg, "dir") else cfg.phase_dir(experiment_name)
    # One manifest per job, keyed by the models that job ran. A single shared
    # `run_manifest.json` is overwritten by whichever model's job finishes last,
    # which loses the record of how the others were produced — exactly the
    # provenance this file exists to preserve.
    path = out_dir / f"run_manifest__{'+'.join(getattr(cfg, 'models', []) or ['all'])}.json"
    save_json(path, manifest)
    return path
