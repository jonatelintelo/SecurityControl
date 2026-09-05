"""Shared measurement functions used across phases 3-5: causal redundancy (k_50)
sweeps, attack-success-rate sweeps (k*/alpha* search), and stability metrics
(component-set overlap, representation similarity, simple correlation)."""
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from core.interventions import InterventionEngine
from core.model_io import generate_text, is_refusal_text, refusal_logit_margin


# ---------------------------------------------------------------------------
# Causal redundancy: A_R(k) ablation curves and k_50 (phase 3)
# ---------------------------------------------------------------------------
def component_ablation_curve(
    model,
    tokenizer,
    device: str,
    mlp_module,
    ranked_indices: Sequence[int],
    probe_toks_list: List[dict],
    score_fn: Callable[..., float],
    k_grid: Sequence[int],
) -> pd.DataFrame:
    """Ablates the top-k ranked neurons for each k in k_grid and records the
    fraction of the baseline score remaining (A_R(k)), averaged over
    probe_toks_list. score_fn(model, tokenizer, toks, device) -> float."""
    base_scores = [score_fn(model, tokenizer, toks, device) for toks in probe_toks_list]
    base_mean = float(np.mean(base_scores))

    rows = []
    for k in k_grid:
        handle = InterventionEngine.ablate_neurons(mlp_module, list(ranked_indices[:k]))
        scores = [score_fn(model, tokenizer, toks, device) for toks in probe_toks_list]
        handle.remove()
        ablated_mean = float(np.mean(scores))
        ratio = ablated_mean / (base_mean + 1e-9)
        rows.append({"k": k, "base_score": base_mean, "ablated_score": ablated_mean, "A_R_k": ratio})

    return pd.DataFrame(rows)


def find_k50(curve_df: pd.DataFrame, threshold: float = 0.5) -> Optional[int]:
    hit = curve_df[curve_df["A_R_k"] <= threshold]
    if hit.empty:
        return None
    return int(hit.iloc[0]["k"])


def refusal_margin_score(model, tokenizer, toks, device) -> float:
    return refusal_logit_margin(model, tokenizer, toks, device)


# ---------------------------------------------------------------------------
# Attack success rate + k*/alpha* search (phase 4)
# ---------------------------------------------------------------------------
def attack_success_rate(model, tokenizer, prompts: List[str], device: str, max_new_tokens: int) -> float:
    """Behavioral ASR: fraction of prompts for which the (possibly intervened-upon)
    model's generation is NOT classified as a refusal, i.e. the attack "succeeded"."""
    successes = 0
    for prompt in prompts:
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        text = generate_text(model, tokenizer, toks, max_new_tokens)
        if not is_refusal_text(text):
            successes += 1
    return successes / max(1, len(prompts))


def find_min_k_for_asr(
    model,
    tokenizer,
    device: str,
    mlp_module,
    ranked_indices: Sequence[int],
    prompts: List[str],
    k_grid: Sequence[int],
    tau: float,
    max_new_tokens: int,
) -> Tuple[Optional[int], pd.DataFrame]:
    """Sweeps k, ablating the top-k ranked neurons, and returns k* = min k with
    ASR(k) >= tau (None if no k in the grid reaches tau) plus the full ASR(k) curve."""
    rows = []
    k_star = None
    for k in k_grid:
        handle = InterventionEngine.ablate_neurons(mlp_module, list(ranked_indices[:k]))
        asr = attack_success_rate(model, tokenizer, prompts, device, max_new_tokens)
        handle.remove()
        rows.append({"k": k, "ASR": asr})
        if k_star is None and asr >= tau:
            k_star = k
    return k_star, pd.DataFrame(rows)


def find_min_alpha_for_asr(
    model,
    tokenizer,
    device: str,
    layer_module,
    direction: torch.Tensor,
    prompts: List[str],
    alpha_grid: Sequence[float],
    tau: float,
    max_new_tokens: int,
) -> Tuple[Optional[float], pd.DataFrame]:
    """Sweeps steering magnitude alpha and returns alpha* = min |alpha| with
    ASR(alpha) >= tau (None if no alpha in the grid reaches tau) plus the full curve."""
    rows = []
    alpha_star = None
    for alpha in sorted(alpha_grid, key=abs):
        handle = InterventionEngine.steer_subspace(layer_module, direction, alpha)
        asr = attack_success_rate(model, tokenizer, prompts, device, max_new_tokens)
        handle.remove()
        rows.append({"alpha": alpha, "ASR": asr})
        if alpha_star is None and asr >= tau:
            alpha_star = alpha
    return alpha_star, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stability metrics (phase 5: RQ6 context reorganization)
# ---------------------------------------------------------------------------
def component_set_overlap(ranking_a: Sequence[int], ranking_b: Sequence[int], k: int) -> float:
    """Jaccard overlap between the top-k index sets of two component rankings."""
    set_a, set_b = set(ranking_a[:k]), set(ranking_b[:k])
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def simple_correlation(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """Returns (pearson_r, spearman_rho) between two equal-length sequences,
    used in phase 4 to relate frozen architecture metrics to attack budgets."""
    x_arr, y_arr = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x_arr) < 2 or np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return float("nan"), float("nan")
    pearson_r = float(np.corrcoef(x_arr, y_arr)[0, 1])
    rank_x, rank_y = pd.Series(x_arr).rank(), pd.Series(y_arr).rank()
    spearman_rho = float(np.corrcoef(rank_x, rank_y)[0, 1])
    return pearson_r, spearman_rho
