"""Experiment configuration, from environment variables.

Rewritten for the experiment-per-RQ structure in EXPERIMENTS.md. The previous
version was organised around the six abandoned phases and carried tunables
(k-grids, alpha-grids, factorial-design switches) belonging to code that has been
removed.

Scaling up should mean changing values here, never editing experiment logic.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    model_id: str
    device: str
    results_root: Path
    seed: int

    # FAST_DEV shrinks sizes so the full experiment can be smoke-tested in
    # minutes. It changes how much data each computation sees, never which
    # computations run.
    fast_dev: bool

    n_harmful: int
    n_harmless: int
    roles: List[str]
    train_fraction: float

    batch_size: int
    # Token-level capture across every layer is the memory-dominant path; this
    # bounds it. See core.capture.at_content_tokens.
    max_content_tokens: int
    # None = every layer. A stride subsamples layers for the expensive
    # token-level capture only.
    layer_stride: Optional[int]

    # Short generations suffice to label refuse-vs-comply; refusal is apparent in
    # the opening tokens. Full-length generation is only needed for ASR (RQ4).
    refusal_max_new_tokens: int

    def dir(self, name: str) -> Path:
        d = self.results_root / name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def layers(self, num_layers: int) -> List[int]:
        return list(range(num_layers))

    def probe_layers(self, num_layers: int) -> List[int]:
        s = self.layer_stride or 1
        return list(range(0, num_layers, s))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["results_root"] = str(d["results_root"])
        return d


def load_config() -> Config:
    import torch

    fast = _b("FAST_DEV", False)
    return Config(
        model_id=os.environ.get("MODEL_ID", "Qwen/Qwen3.5-9B"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        results_root=Path(os.environ.get("RESULTS_ROOT", "./results")),
        seed=_i("SEED", 0),
        fast_dev=fast,
        n_harmful=_i("N_HARMFUL", 8 if fast else 200),
        n_harmless=_i("N_HARMLESS", 8 if fast else 200),
        roles=os.environ.get("ROLES", "system,user,tool,assistant").split(","),
        train_fraction=_f("TRAIN_FRACTION", 0.75),
        batch_size=_i("BATCH_SIZE", 4 if fast else 16),
        max_content_tokens=_i("MAX_CONTENT_TOKENS", 4 if fast else 8),
        layer_stride=(_i("LAYER_STRIDE", 0) or None),
        refusal_max_new_tokens=_i("REFUSAL_MAX_NEW_TOKENS", 16 if fast else 48),
    )
