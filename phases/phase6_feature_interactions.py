"""Phase 6 — Feature-interaction case studies (active steering).

Concrete implementations of the "how do features interact" experiments from
the informal project write-up, reusing exactly the steering/projection
machinery from phases 1-3:

  1. Role -> harm: does steering a benign instruction toward the injected/
     tool-output role direction (early layer) make it read as more harmful
     downstream (later layers)?
  2. Persona -> harm/refusal: does pushing the persona toward an
     unconstrained/misaligned framing change whether a harmful prompt is
     still represented as harmful, and whether it is still refused?

Deferred: "remove fear from a harmful prompt" from the informal write-up
requires an emotions probe (Transformer Circuits, optional per the project
plan) that isn't implemented in this pipeline yet — not silently dropped,
just blocked on that probe existing.

Requires phase1_geometry to have been run first.

Outputs (results/phase6_feature_interactions/):
  role_to_harm_interaction.csv     alpha, read_layer, mean_harm_projection, delta_vs_baseline
  persona_interaction.csv          alpha, mean_harm_projection, refusal_margin, harm_projection_delta, margin_delta
  persona_interaction_samples.txt  qualitative generations at baseline vs. strongest misalignment steering
"""
import pandas as pd
import torch
import torch.nn.functional as F

from core import data
from core.config import Config, load_config
from core.hooks import HookEngine, capture_last_token_residuals
from core.interventions import InterventionEngine
from core.io_utils import get_logger, load_json, load_torch, require_phase_output, save_df
from core.model_io import build_prompt, generate_text, load_model
from core.subspaces import SubspaceEngine

PHASE_NAME = "phase6_feature_interactions"
PHASE1 = "phase1_geometry"
ROLE_ATTACK_SIGN = data.ATTACK_STEERING_SIGN["role"]
PERSONA_MISALIGN_SIGN = -1  # PERSONA_PAIRS positive=aligned, negative=misaligned -> misalignment is the negative direction


def run(cfg: Config) -> None:
    out_dir = cfg.phase_dir(PHASE_NAME)
    logger = get_logger(PHASE_NAME, out_dir)
    logger.info("=" * 80)
    logger.info("PHASE 6: FEATURE-INTERACTION CASE STUDIES (qualitative, active steering)")
    logger.info("=" * 80)

    phase1_dir = cfg.phase_dir(PHASE1)
    require_phase_output(phase1_dir / "directions.pt", PHASE1)
    directions = load_torch(phase1_dir / "directions.pt")
    selected = load_json(phase1_dir / "selected_layers.json")
    early, mid, late = selected["early_layer"], selected["mid_layer"], selected["late_layer"]

    model, tokenizer = load_model(cfg.model_id, cfg.device, logger)
    hooks = HookEngine(model)

    # Magnitudes must match the steering mode: under relative steering alpha is
    # a fraction of the residual norm, so the absolute-scale grid would be
    # wildly destructive.
    if cfg.steer_relative:
        alpha_magnitudes = [0, 0.25, 1.0] if cfg.fast_dev else [0, 0.1, 0.25, 0.5, 1.0]
    else:
        alpha_magnitudes = [0, 2, 4] if cfg.fast_dev else [0, 1, 2, 4, 8]

    # ---------------------------------------------------------------
    # Case study 1: does steering a benign instruction toward the
    # injected/tool-output role make it look more harmful downstream?
    # ---------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Case study 1: role -> harm interaction")
    logger.info("-" * 80)

    benign_texts = [build_prompt(tokenizer, user=p) for p in data.BENIGN_REFERENCE_PROMPTS]
    read_layers = sorted({mid, late})

    role_rows = []
    for magnitude in alpha_magnitudes:
        alpha = ROLE_ATTACK_SIGN * magnitude
        handle = InterventionEngine.steer_subspace(hooks.layers[early], directions["role"][early], alpha, relative=cfg.steer_relative) if magnitude != 0 else None

        hooks.hook_residual_stream(read_layers)
        inputs = tokenizer(benign_texts, return_tensors="pt", padding=True).to(cfg.device)
        with torch.no_grad():
            _ = model(**inputs)
        resid = {l: hooks.residual_cache[l][:, -1, :].detach().cpu() for l in read_layers}
        hooks.remove_all_hooks()
        if handle is not None:
            handle.remove()

        projs_this_alpha = {}
        for l in read_layers:
            proj = F.cosine_similarity(resid[l], directions["harm"][l].unsqueeze(0), dim=1).mean().item()
            projs_this_alpha[l] = proj
            role_rows.append({"alpha": alpha, "read_layer": l, "mean_harm_projection": proj})
        logger.info(f"[role->harm] steer alpha={alpha:+.1f} @L{early} -> " + ", ".join(f"harm_proj@L{l}={v:.4f}" for l, v in projs_this_alpha.items()))

    df_role = pd.DataFrame(role_rows)
    baseline_by_layer = df_role[df_role.alpha == 0].set_index("read_layer")["mean_harm_projection"]
    df_role["delta_vs_baseline"] = df_role.apply(lambda r: r["mean_harm_projection"] - baseline_by_layer[r["read_layer"]], axis=1)
    save_df(out_dir / "role_to_harm_interaction.csv", df_role)
    logger.info(f"Role->harm interaction table:\n{df_role.to_string()}")

    # ---------------------------------------------------------------
    # Case study 2: does pushing the persona toward an unconstrained/
    # misaligned framing change whether a harmful prompt is still
    # represented as harmful, and whether it is still refused?
    # ---------------------------------------------------------------
    logger.info("-" * 80)
    logger.info("Case study 2: persona misalignment -> harmfulness / refusal")
    logger.info("-" * 80)

    persona_pairs = data.PERSONA_PAIRS[:4] if cfg.fast_dev else data.PERSONA_PAIRS
    persona_layers = sorted({early, mid, late})
    pos_texts, neg_texts = zip(*data.materialize_pairs(tokenizer, persona_pairs))
    pos_resid = capture_last_token_residuals(model, hooks, tokenizer, list(pos_texts), cfg.device, persona_layers)
    neg_resid = capture_last_token_residuals(model, hooks, tokenizer, list(neg_texts), cfg.device, persona_layers)
    persona_direction = {l: SubspaceEngine.extract_direction(pos_resid[l], neg_resid[l]) for l in persona_layers}

    n_harm_probes = 3 if cfg.fast_dev else 6
    harm_requests = [p.positive.user for p in data.HARM_PAIRS[:n_harm_probes]]
    harm_texts = [build_prompt(tokenizer, user=r) for r in harm_requests]
    id_refusal = tokenizer.encode("I", add_special_tokens=False)[0]
    id_compliance = tokenizer.encode("Sure", add_special_tokens=False)[0]

    persona_rows = []
    for magnitude in alpha_magnitudes:
        alpha = PERSONA_MISALIGN_SIGN * magnitude
        handle = InterventionEngine.steer_subspace(hooks.layers[early], persona_direction[early], alpha, relative=cfg.steer_relative) if magnitude != 0 else None

        hooks.hook_residual_stream([mid])
        inputs = tokenizer(harm_texts, return_tensors="pt", padding=True).to(cfg.device)
        with torch.no_grad():
            out = model(**inputs)
        resid = hooks.residual_cache[mid][:, -1, :].detach().cpu()
        hooks.remove_all_hooks()
        if handle is not None:
            handle.remove()

        harm_proj = F.cosine_similarity(resid, directions["harm"][mid].unsqueeze(0), dim=1).mean().item()
        margin = (out.logits[:, -1, id_refusal] - out.logits[:, -1, id_compliance]).mean().item()
        persona_rows.append({"alpha": alpha, "mean_harm_projection": harm_proj, "refusal_margin": margin})
        logger.info(f"[persona->harm/refusal] steer alpha={alpha:+.1f} @L{early} -> harm_proj@L{mid}={harm_proj:.4f}, refusal_margin={margin:.4f}")

    df_persona = pd.DataFrame(persona_rows)
    base_row = df_persona[df_persona.alpha == 0].iloc[0]
    df_persona["harm_projection_delta"] = df_persona["mean_harm_projection"] - base_row["mean_harm_projection"]
    df_persona["margin_delta"] = df_persona["refusal_margin"] - base_row["refusal_margin"]
    save_df(out_dir / "persona_interaction.csv", df_persona)
    logger.info(f"Persona interaction table:\n{df_persona.to_string()}")

    # Qualitative samples: baseline vs strongest misalignment steering.
    strongest_alpha = PERSONA_MISALIGN_SIGN * alpha_magnitudes[-1]
    sample_lines = []
    for text in harm_texts[:2]:
        toks = tokenizer(text, return_tensors="pt").to(cfg.device)
        base_text = generate_text(model, tokenizer, toks, cfg.max_new_tokens)
        handle = InterventionEngine.steer_subspace(hooks.layers[early], persona_direction[early], strongest_alpha, relative=cfg.steer_relative)
        steered_text = generate_text(model, tokenizer, toks, cfg.max_new_tokens)
        handle.remove()
        sample_lines.append(f"PROMPT: {text}\n[baseline] {base_text}\n[persona misaligned, alpha={strongest_alpha:+.1f}] {steered_text}\n{'-' * 80}")
    (out_dir / "persona_interaction_samples.txt").write_text("\n".join(sample_lines))

    logger.info("NOTE: 'remove fear from a harmful prompt' case study deferred pending an emotions probe (optional per project plan).")
    logger.info("=" * 80)
    logger.info(f"Phase 6 complete. Outputs written to {out_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    run(load_config())
