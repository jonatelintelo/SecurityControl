"""Experiment configuration and the model registry.

Model is a **loop dimension, not a constant**: every experiment runs on both dense
models and writes to `results/<experiment>/<model_slug>/`. Model-independent
artifacts — above all the frozen corpus — live in `results/<experiment>/` with no
slug, so every roster model consumes byte-identical inputs.

Scaling up means changing values here, never editing experiment logic.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ModelSpec:
    slug: str
    model_id: str
    family: str
    kind: str          # "dense" | "moe"
    text_only: bool
    note: str = ""
    # `family` above is the TEMPLATE family (qwen2.5 and qwen3.5 differ in chat
    # template and are not interchangeable). `vendor` is the organisation that
    # trained the model. They are different counts and the paper must not use the
    # larger one: our roster spans 4 template families but only 3 vendors, and
    # "three families" is the honest claim for a cross-family generalisation.
    vendor: str = ""
    # Role classes to render for this model. POLICY: maximal — every model runs
    # every role its chat template can express, because each role class is a
    # level of the R_role variable and dropping one narrows what R_role means.
    # This field is NOT a convenience knob: it exists only for a template that
    # genuinely cannot express a class (Mixtral raises on `assistant` and drops
    # `system`), and a role that merely renders awkwardly is a bug to fix in
    # `core/positions.py`, never a role to drop. Where a class IS unavailable the
    # loss is scientific and must be reported: `tool` is the stand-in for the
    # plan's *untrusted external content* class (D1), so without it R_role is an
    # authority/speaker-identity variable only. `None` means the global default.
    roles: Optional[Tuple[str, ...]] = None


# Verified against published configs (see EXPERIMENTS.md > Implementation notes).
MODELS: Dict[str, ModelSpec] = {
    "qwen2.5-7b": ModelSpec(
        "qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5", "dense", True,
        "28L d=3584; text-only; the model where NeuroStrike lands (ASR 0.727)", vendor="Qwen"),
    "qwen3.5-9b": ModelSpec(
        "qwen3.5-9b", "Qwen/Qwen3.5-9B", "qwen3.5", "dense", False,
        "32L d=4096; ForConditionalGeneration wrapper with a vision tower", vendor="Qwen"),
    # The cross-family dense arm. Different family, text-only, and — unlike Gemma-2
    # (which rejects system messages) and Mistral (no tool role) — its chat
    # template expresses all four role classes.
    # Its `tool` template QUOTES the content, so BPE fuses the closing `"` into
    # the instruction's last token. That is a constant template character, the
    # same phenomenon as Qwen's `.` + `\n`, and is handled by the bleed accounting
    # in `core/positions.py` — not a reason to drop the role.
    "llama3.1-8b": ModelSpec(
        "llama3.1-8b", "meta-llama/Llama-3.1-8B-Instruct", "llama3.1", "dense", True,
        "32L d=4096; cross-family dense arm (Meta); tool template quotes content", vendor="Meta"),
    # Smoke-test only: never a result-bearing model. Exists so the GPU code path
    # can be exercised in ~1 minute rather than ~25.
    "qwen2.5-0.5b": ModelSpec(
        "qwen2.5-0.5b", "Qwen/Qwen2.5-0.5B-Instruct", "qwen2.5", "dense", True,
        "smoke tests only", vendor="Qwen"),
    # The ALIGNMENT-ERA control. Evaluated by Arditi et al. (2406.11717), whose
    # refusal-direction results were obtained on this generation of models.
    #
    # It is on the roster for a specific measurement problem, not for coverage.
    # `R_control` is fitted within a harm stratum (refused vs complied among
    # harmful), and on current models that minority cell has collapsed: measured
    # at 200/200, qwen3.5-9b yielded ONE compliant instruction and the Qwen MoE
    # two. The literature does not hit this because it fits refusal as harmful
    # vs harmless — a contrast with 1,200 items per class that cannot starve —
    # but that contrast is approximately `R_harm` renamed, which is exactly what
    # RQ1 must not assume.
    #
    # Yi-6B-Chat is an older, less heavily aligned checkpoint, so it should
    # actually comply with some harmful requests and give `under` a populated
    # cell. If it does, RQ1's control claims rest on at least one model where
    # Zhao's own refuse-vs-accept contrast is properly estimable. It also adds a
    # fourth vendor (01-AI) and renders all four role classes, so `R_role`
    # stays the same variable across the roster.
    "yi-6b-chat": ModelSpec(
        "yi-6b-chat", "01-ai/Yi-6B-Chat", "yi", "dense", True,
        "32L d=4096; Arditi-era checkpoint, less aligned — the model where the "
        "`under` control contrast is expected to be estimable", vendor="01-AI"),
    # The MoE arm of RQ1. The scoped plan's first pass is "one representative
    # dense model and one representative MoE model", and its success criterion is
    # "if both results hold on one dense and one MoE architecture" — so this is
    # part of RQ1, not the roster expansion (which is PEP item 7).
    # Verified: config nests under `text_config`, 40 layers, d=2048, 256 experts
    # (8 active), and the chat template renders all four role classes. Its stack
    # is 30 linear-attention + 10 full-attention layers, which does not affect
    # residual-stream capture but is worth stating when depth profiles are
    # compared against the dense models.
    "qwen3.5-35b-a3b": ModelSpec(
        "qwen3.5-35b-a3b", "Qwen/Qwen3.5-35B-A3B", "qwen3.5", "moe", False,
        "40L d=2048, 256 experts (8 active); RQ1 MoE arm", vendor="Qwen"),
    # The cross-family MoE arm, and the one model on the roster that the role
    # paper (2603.12277) itself evaluated — which is the point of including it.
    # 52 layers makes it the most expensive model here: E1.6 reads every
    # downstream layer, so causal cost grows with roughly L^2, and the 40-layer
    # Qwen MoE already needed more than 6h. Budget accordingly.
    "nemotron-3-nano-30b-a3b": ModelSpec(
        "nemotron-3-nano-30b-a3b", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "nemotron3", "moe", True,
        "52L d=2688, 128 experts; cross-family MoE arm (NVIDIA); evaluated in 2603.12277", vendor="NVIDIA"),
}

# Run order for the dense arm. Qwen2.5 first: it is text-only and is where the
# NeuroStrike signal demonstrably exists, so a null there is informative rather
# than ambiguous.
DENSE_MODELS: List[str] = ["qwen2.5-7b", "qwen3.5-9b", "llama3.1-8b", "yi-6b-chat"]

# The full RQ1 roster: five models, three families, both architectures, and all
# four role classes on every one — so `R_role` is the SAME 4-class variable
# everywhere and the cross-model comparison is like-for-like.
#
#   qwen2.5-7b                dense  Qwen      28L
#   qwen3.5-9b                dense  Qwen      32L
#   llama3.1-8b               dense  Meta      32L
#   yi-6b-chat                dense  01-AI     32L   (Arditi-era, less aligned)
#   qwen3.5-35b-a3b           MoE    Qwen      40L
#   nemotron-3-nano-30b-a3b   MoE    NVIDIA    52L
#
# THIS IS THE SINGLE SOURCE OF TRUTH. Ten tools previously kept their own
# hardcoded copy, so adding a model silently left them reporting on a subset —
# a figure or a verification that quietly covers 3 of 5 models is worse than one
# that fails. Import from here; never re-list slugs.
#
# Not the default for `MODELS` — jobs pass one slug each, because `sbatch
# --export` splits on commas and would silently drop the rest of a list.
MOE_MODELS: List[str] = ["qwen3.5-35b-a3b", "nemotron-3-nano-30b-a3b"]
RQ1_MODELS: List[str] = DENSE_MODELS + MOE_MODELS

# The control variant EVERY model is run on, so the cross-model gate compares
# like with like. `under` (refused vs complied among harmful) is additionally
# run wherever that model's harmful-and-complied cell is large enough to fit it
# — which is discovered at E1.1 Stage 1, never assumed, and is why the `under`
# arm carries CONTROL_VARIANT_OPTIONAL and may legitimately be absent.
MATCHED_CONTROL_VARIANT = "over"

# Large regenerable caches live off the home quota. Overridable so a run can be
# made self-contained (set CACHE_ROOT to a path inside the results root) when
# disk is not the constraint.
CACHE_ROOT = Path(os.environ.get(
    "CACHE_ROOT", "/scratch/b6aj/jtelintelo.b6aj/rq1-activation-cache"))


def roster_table() -> str:
    """One-line-per-model description, for run logs and verification output."""
    rows = []
    for slug in RQ1_MODELS:
        m = MODELS[slug]
        rows.append(f"  {slug:26s} {m.vendor:8s} {m.family:10s} {m.kind:6s} "
                    f"roles={','.join(m.roles) if m.roles else 'default(4)'}")
    fams = sorted({MODELS[s].family for s in RQ1_MODELS})
    vends = sorted({MODELS[s].vendor for s in RQ1_MODELS})
    kinds = sorted({MODELS[s].kind for s in RQ1_MODELS})
    return ("\n".join(rows)
            + f"\n  => {len(RQ1_MODELS)} models; {len(vends)} vendors ({', '.join(vends)}); "
              f"{len(fams)} template families ({', '.join(fams)}); "
              f"architectures: {', '.join(kinds)}")


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

    def roles_for(self, slug: str) -> List[str]:
        """Role classes to render for one model — its own set, else the global one.

        Every role loop must go through here. A stage that reads `cfg.roles`
        directly would render four roles for a three-role model and crash in the
        chat template, or worse, render a substitute and quietly change what the
        role contrast contrasts.
        """
        r = MODELS[slug].roles if slug in MODELS else None
        return list(r) if r else list(self.roles)

    def cache_dir(self, experiment: str, model_slug: str) -> Path:
        """Where the large, regenerable activation blob lives.

        NOT under `results_root`. `activations.pt` is 1-2 GB per model per root,
        so the full roster across two roots is ~16 GB — against a ~101 GB home
        quota that is the binding constraint on this project, while /scratch has
        terabytes free. It is a CACHE, not evidence: it is regenerated
        deterministically from the frozen corpus, no drift comparison reads it,
        and `tools/prune_caches.py` already exists to delete it.

        KEYED BY THE ABSOLUTE RESULTS ROOT, and that is the whole point of the
        hash. `results` and `results_verify` are independent repetitions of the
        same models at the same config; if they shared one cache path, the
        second run would silently load the first run's activations and the
        verifier's exactness assertion would be comparing a run against itself —
        producing a perfect agreement that means nothing. The basename is kept
        alongside the hash only so the directories are readable by a human.
        """
        root = str(self.results_root.resolve())
        key = f"{self.results_root.resolve().name}-{hashlib.sha256(root.encode()).hexdigest()[:8]}"
        d = CACHE_ROOT / key / experiment / model_slug
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
        # 500/500, not 200/200. Sized from MEASURED cell sizes, not guessed.
        #
        # `R_harm` and `R_role` are fitted on the whole corpus and are well
        # powered at any of these scales. `R_control` is fitted only on the
        # subset the model's own behaviour creates, and at 200/200 its minority
        # class was 1-35 instructions across the roster — one to two orders of
        # magnitude less data than the other two variables, for the variable
        # RQ1 compares them against.
        #
        # Projected minority instructions at 500/500, from each model's own
        # measured per-source rates (over / under):
        #     qwen2.5-7b 40/56   qwen3.5-9b 88/1   llama3.1-8b 50/39
        #     qwen3.5-35b-a3b 68/5   nemotron 5/14
        # The matched `over` arm clears the ~20 floor on four of five models;
        # `under` is rescued only where the compliance rate is non-zero, and a
        # jailbreak-framing probe confirmed persuasion styles do NOT raise it
        # (they convert refusals into soft refusals, not into compliance).
        #
        # 500 is chosen by the source caps, not arbitrarily: it exhausts XSTest
        # (250 safe rows — the source that actually produces over-refusals) and
        # JBB (100 behaviours), while `_take`'s round-robin keeps the source
        # balance that stops `R_harm` collapsing into a sensitive-topic detector.
        # Going further buys little: Alpaca's over-refusal rate is 0.01-0.05, so
        # extra Alpaca adds items without adding cells.
        n_harmful=_i("N_HARMFUL", 16 if fast else 500),
        n_harmless=_i("N_HARMLESS", 16 if fast else 500),
        n_attack=_i("N_ATTACK", 16 if fast else 150),
        # E1.0b transfer corpus. 150 matches the role paper's own scale
        # (150 base passages x role classes) so the reproduction is comparable.
        n_transfer=_i("N_TRANSFER", 16 if fast else 150),
        train_fraction=_f("TRAIN_FRACTION", 0.75),
        roles=[r for r in re.split(r"[,+]", os.environ.get("ROLES", "system,user,tool,assistant")) if r],
        designs=[d for d in re.split(r"[,+]", os.environ.get("DESIGNS", "fixed_slot,natural_slot")) if d],
        # `+` is accepted as a separator alongside `,`.
        # NOT a style choice: `sbatch --export=ALL,MODELS=a,b` is unusable because
        # sbatch splits --export on commas, so `b` is misparsed as a separate
        # assignment and silently vanishes. Per-model jobs pass one slug and do
        # not care, but the corpus builder must render for the WHOLE roster in
        # one job, so it needs a separator sbatch will not eat.
        models=[m for m in re.split(r"[,+]", os.environ.get("MODELS", ",".join(DENSE_MODELS))) if m],
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
