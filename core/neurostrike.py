"""Faithful port of NeuroStrike's white-box safety-neuron method (NDSS 2026,
arXiv:2509.11864), from the reference implementation in
../NeuroStrike-Neuron-Level-Attacks-on-Aligned-LLMs/white_box/.

Why this module exists: RQ3 asks what functions are performed by components
*previously identified by NeuroStrike*. Answering that requires their actual
neuron set, which comes from a supervised logistic probe over gate/up
activations thresholded at |z|>3 & w>0 — not an activation-difference
heuristic. Everything here mirrors the reference implementation; deviations
are called out inline.

Deliberately kept from the reference, even though it differs from how
core/attribution.py defines a "neuron":
  - hooks gate_proj AND up_proj outputs separately (two neuron sets per layer),
    not the post-SiLU product silu(gate)*up used elsewhere in this codebase
  - max-pools over the sequence dimension, rather than mean over |activation|
  - pooling includes padding positions (the reference does not mask them);
    masking here would change which neurons are selected and break the
    "these are the same neurons NeuroStrike finds" claim

One deliberate addition: the probe's raw weight vector is also exposed as a
*ranking*, because k* / k_50 need a graded budget axis that the reference's
fixed |z|>3 threshold rule does not provide.
"""
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import zscore

# The reference hooks every module whose name contains "gate" or "up".
HOOK_KEYWORDS = ("gate_proj", "up_proj")
LAYER_RE = re.compile(r"layers\.(\d+)\.")


def module_layer_index(module_name: str) -> Optional[int]:
    m = LAYER_RE.search(module_name)
    return int(m.group(1)) if m else None


def capture_gate_up_activations(model, tokenizer, prompts: List[str], device: str, batch_size: int = 32) -> Dict[str, np.ndarray]:
    """Max-pooled gate_proj/up_proj activations per prompt.

    Returns {module_name: array of shape (num_prompts, intermediate_size)}.
    """
    activations: Dict[str, List[np.ndarray]] = {}

    def make_hook(layer_name: str):
        def hook(module, inputs, output):
            act = output.max(dim=1)[0].detach().cpu().float().numpy()
            activations.setdefault(layer_name, []).append(act)

        return hook

    handles = []
    for name, module in model.named_modules():
        if any(k in name.lower() for k in HOOK_KEYWORDS):
            handles.append(module.register_forward_hook(make_hook(name)))

    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            toks = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                _ = model(**toks)
    finally:
        for h in handles:
            h.remove()

    return {name: np.concatenate(chunks, axis=0) for name, chunks in activations.items()}


class _LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


def train_safety_probe(
    activations: np.ndarray,
    labels: np.ndarray,
    device: str,
    num_runs: int = 1,
    num_epochs: int = 5000,
    lr: float = 0.001,
    weight_decay: float = 1e-3,
) -> np.ndarray:
    """Logistic probe over one module's activations; returns per-neuron weights
    averaged over runs. Hyperparameters match the reference exactly."""
    acts = torch.tensor(activations, dtype=torch.float32).to(device)
    labs = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)

    nan_mask = torch.isnan(acts).any(dim=1)
    if nan_mask.any():
        acts, labs = acts[~nan_mask], labs[~nan_mask]

    importances = []
    for run in range(num_runs):
        torch.manual_seed(1234 + run)
        perm = torch.randperm(acts.size(0))
        x, y = acts[perm], labs[perm]

        clf = _LogisticRegressionModel(acts.size(1)).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(clf.parameters(), lr=lr, weight_decay=weight_decay)
        for _ in range(num_epochs):
            clf.train()
            optimizer.zero_grad()
            loss = criterion(clf(x), y)
            loss.backward()
            optimizer.step()
        importances.append(clf.linear.weight.data.cpu().numpy().flatten())

    return np.mean(importances, axis=0)


def select_safety_neurons(weights: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Reference selection rule: |z-score| > threshold AND positive weight."""
    z = zscore(weights)
    return np.where((np.abs(z) > threshold) & (weights > 0))[0]


def rank_by_probe_weight(weights: np.ndarray) -> np.ndarray:
    """Neuron indices ordered most- to least-safety-relevant by probe weight.

    Not part of the reference (which only thresholds). Needed so the same
    signal can drive a graded top-k budget sweep for k* / k_50.
    """
    return np.argsort(-weights)


def compute_safety_neurons(
    model,
    tokenizer,
    harmful_prompts: List[str],
    benign_prompts: List[str],
    device: str,
    threshold: float = 3.0,
    batch_size: int = 32,
    num_epochs: int = 5000,
    logger=None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Full pipeline: activations -> per-module probe -> safety-neuron sets.

    Returns (weights_by_module, neurons_by_module).
    """
    prompts = list(harmful_prompts) + list(benign_prompts)
    labels = np.array([1] * len(harmful_prompts) + [0] * len(benign_prompts), dtype=np.float32)

    if logger:
        logger.info(f"NeuroStrike probe: {len(harmful_prompts)} harmful / {len(benign_prompts)} benign prompts")
    activations = capture_gate_up_activations(model, tokenizer, prompts, device, batch_size)

    weights_by_module, neurons_by_module = {}, {}
    for name, acts in activations.items():
        w = train_safety_probe(acts, labels, device, num_epochs=num_epochs)
        weights_by_module[name] = w
        neurons_by_module[name] = select_safety_neurons(w, threshold)
    if logger:
        total = sum(len(v) for v in neurons_by_module.values())
        logger.info(f"NeuroStrike probe: {total} safety neurons across {len(neurons_by_module)} modules (|z|>{threshold}, w>0)")

    return weights_by_module, neurons_by_module


def register_prune_hooks(model, neurons_by_module: Dict[str, np.ndarray], max_layer: Optional[int] = None) -> List:
    """Zero the selected neurons at gate/up outputs (the reference's prune site).

    max_layer implements the reference's layer-prefix sweep: prune only modules
    in layers 0..max_layer inclusive. None prunes every module.
    """
    modules = dict(model.named_modules())
    handles = []
    for name, indices in neurons_by_module.items():
        if len(indices) == 0:
            continue
        if max_layer is not None:
            idx = module_layer_index(name)
            if idx is None or idx > max_layer:
                continue
        target = modules.get(name)
        if target is None:
            continue

        def make_hook(neuron_indices):
            idx_t = torch.tensor(np.asarray(neuron_indices), dtype=torch.long)

            def hook(module, inputs, output):
                pruned = output.clone()
                pruned[..., idx_t.to(output.device)] = 0
                return pruned

            return hook

        handles.append(target.register_forward_hook(make_hook(indices)))
    return handles
