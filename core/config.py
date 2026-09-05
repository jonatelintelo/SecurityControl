"""Runtime configuration shared by every phase script.

All tunables come from environment variables so the same code runs
unmodified on a CPU dev box (small/fast settings) and on the Slurm GPU
node (full-size settings) — see FAST_DEV below.
"""
import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    model_id: str
    device: str
    results_root: Path
    seed: int

    # FAST_DEV=1 shrinks dataset sizes / sweep grids / generation length so the
    # full phase sequence can be smoke-tested on a CPU box in minutes instead
    # of requiring the GPU cluster. It does not change *which* computations
    # run, only how much data/how many sweep points each one covers.
    fast_dev: bool

    max_new_tokens: int
    attack_success_threshold: float  # tau in the project plan
    train_fraction: float  # train/held-out split for direction extraction + probe validation

    def phase_dir(self, name: str) -> Path:
        d = self.results_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d


def load_config() -> Config:
    import torch

    fast_dev = _env_bool("FAST_DEV", False)
    return Config(
        model_id=os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        results_root=Path(os.environ.get("RESULTS_ROOT", "./results")),
        seed=_env_int("SEED", 0),
        fast_dev=fast_dev,
        max_new_tokens=_env_int("MAX_NEW_TOKENS", 16 if fast_dev else 40),
        attack_success_threshold=_env_float("ATTACK_SUCCESS_THRESHOLD", 0.5),
        train_fraction=_env_float("TRAIN_FRACTION", 0.75),
    )
