"""Small shared helpers: logging setup and result-file save/load.

Every phase writes into results/<phase_name>/ using these helpers so later
phases can load an earlier phase's frozen outputs instead of recomputing them.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch


def get_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    logger.addHandler(stderr_handler)

    file_handler = logging.FileHandler(log_dir / f"{name}.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def save_torch(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, path)


def load_torch(path: Path) -> Any:
    return torch.load(path, map_location="cpu")


def save_df(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_df(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def require_phase_output(path: Path, phase_name: str) -> Path:
    """Fails fast with an actionable message if an upstream phase hasn't been run yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing expected output {path}. Run phase '{phase_name}' first: `python run_phase.py <n>`."
        )
    return path
