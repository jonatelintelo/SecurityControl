"""Experiment configuration and the model registry.

Model is a **loop dimension, not a constant**: every experiment runs on both dense
models and writes to `results/<experiment>/<model_slug>/`. Model-independent
artifacts — above all the frozen corpus — live in `results/<experiment>/` with no
slug, so both models consume byte-identical inputs.

Scaling up means changing values here, never editing experiment logic.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    family: str
    kind: str          # "dense" | "moe"
    text_only: bool
    note: str = ""


# Verified against published configs (see EXPERIMENTS.md > Implementation notes).
MODELS: Dict[str, ModelSpec] = {
    "qwen2.5-7b": ModelSpec(
        "qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5", "dense", True,
        "28L d=3584; text-only; the model where NeuroStrike lands (ASR 0.727)"),
    "qwen3.5-9b": ModelSpec(
        "qwen3.5-9b", "Qwen/Qwen3.5-9B", "qwen3.5", "dense", False,
        "32L d=4096; ForConditionalGeneration wrapper with a vision tower"),
    # Candidate third model, for the cross-model claim. Different family, text-only,
    # and — unlike Gemma-2 (which rejects system messages) and Mistral (no tool
    # role) — its chat template supports all four role classes our corpus needs.
    # Registered so the labels stage can PROBE whether its harmful-and-complied
    # cell populates; it is not part of any result until that probe passes.
    "llama3.1-8b": ModelSpec(
        "llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "llama3.1", "dense", True,
        "32L d=4096; candidate 3rd model — probe only until its R_control_under cell is shown usable"),
    # Smoke-test only: never a result-bearing model. Exists so the GPU code path
    # can be exercised in ~1 minute rather than ~25.
    "qwen2.5-0.5b": ModelSpec(
        "qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5", "dense", True,
        "smoke tests only"),
    "qwen3.5-35b-a3b": ModelSpec(
        "qwen3.5-35b-a3b", "Qwen/Qwen3.5-35B-A3B", "qwen3.5", "moe", False,
        "40L d=2048, 256 experts; deferred until dense RQ1-4 settle"),
}

# Run order for the dense arm. Qwen2.5 first: it is text-only and is where the
# NeuroStrike signal demonstrably exists, so a null there is informative rather
# than ambiguous.
DENSE_MODELS: List[str] = ["qwen2.5-7b", "qwen3.5-9b"]


def _b(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    return default if v is None else v.strip().lower() in {"1", "true", "yes", "on"}


def _i(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _f(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Config:
    results_root: Path
    seed: int
    fast_dev: bool

    # corpus
    n_harmful: int
    n_harmless: int
    n_attack: int
    n_transfer: int
    train_fraction: float
    roles: List[str]
    designs: List[str]

    # extraction / generation
    models: List[str]
    batch_size: int
    max_content_tokens: int
    layer_stride: Optional[int]
    refusal_max_new_tokens: int
    refusal_budget_ladder: List[int]

    def dir(self, experiment: str, model_slug: Optional[str] = None) -> Path:
        d = self.results_root / experiment / model_slug if model_slug else self.results_root / experiment
        d.mkdir(parents=True, exist_ok=True)
        return d

    def spec(self, slug: str) -> ModelSpec:
        if slug not in MODELS:
            raise KeyError(f"unknown model {slug!r}; known: {sorted(MODELS)}")
        return MODELS[slug]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["results_root"] = str(d["results_root"])
        return d


def load_config() -> Config:
    fast = _b("FAST_DEV", False)
    return Config(
        results_root=Path(os.environ.get("RESULTS_ROOT", "./results")),
        seed=_i("SEED", 0),
        fast_dev=fast,
        n_harmful=_i("N_HARMFUL", 16 if fast else 200),
        n_harmless=_i("N_HARMLESS", 16 if fast else 200),
        n_attack=_i("N_ATTACK", 16 if fast else 150),
        # E1.0b transfer corpus. 150 matches the role paper's own scale
        # (150 base passages x role classes) so the reproduction is comparable.
        n_transfer=_i("N_TRANSFER", 16 if fast else 150),
        train_fraction=_f("TRAIN_FRACTION", 0.75),
        roles=os.environ.get("ROLES", "system,user,tool,assistant").split(","),
        designs=os.environ.get("DESIGNS", "fixed_slot,natural_slot").split(","),
        models=os.environ.get("MODELS", ",".join(DENSE_MODELS)).split(","),
        batch_size=_i("BATCH_SIZE", 4 if fast else 16),
        max_content_tokens=_i("MAX_CONTENT_TOKENS", 4 if fast else 8),
        layer_stride=(_i("LAYER_STRIDE", 0) or None),
        # NOT reduced under FAST_DEV. The generation budget changes what a label
        # *means*, not just how long the run takes: too short, and the model has
        # produced only a hedging preamble — a refusal in progress carrying no
        # refusal marker yet — which is then labelled `complied`. Smoke runs get
        # their speed from fewer items, never from shorter generations.
        #
        # Class: measured. EXPERIMENTS.md fixes the rule — the smallest budget in
        # REFUSAL_BUDGET_LADDER with `undetermined` < 30% AND >= 95% label
        # agreement with the next-larger budget. E1.1 Stage 1 runs that ladder and
        # records the choice; this value is the ladder's first rung, not a default.
        refusal_max_new_tokens=_i("REFUSAL_MAX_NEW_TOKENS", 48),
        refusal_budget_ladder=[int(x) for x in
                               os.environ.get("REFUSAL_BUDGET_LADDER", "48,128,256").split(",")],
    )


def relative_depth(layer: int, n_layers: int) -> float:
    """Cross-model curves are plotted against relative depth.

    Qwen2.5-7B has 28 layers and Qwen3.5-9B has 32, so absolute indices are not
    comparable; only within-model claims may use them.
    """
    return layer / (n_layers - 1) if n_layers > 1 else 0.0
