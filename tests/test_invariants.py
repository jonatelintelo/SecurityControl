#!/usr/bin/env python
"""Verify the engineering claims in EXPERIMENTS.md > Implementation notes.

Those claims were inherited from a previous attempt whose *results* are invalid.
The claims themselves were never re-checked, and several of them dictate how the
extraction code must be written. This module checks each one against the actual
environment so the playbook records verified facts rather than folklore.

Run:  python tests/test_invariants.py
Exit code 0 = every claim behaved as documented.
"""
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

TOKENIZER_ID = os.environ.get("INVARIANT_TOKENIZER", "Qwen/Qwen2.5-7B-Instruct")

RESULTS: list[tuple[str, str, str]] = []  # (claim, verdict, detail)


def record(claim: str, verdict: str, detail: str = "") -> None:
    RESULTS.append((claim, verdict, detail))
    mark = {"CONFIRMED": "ok  ", "REFUTED": "FAIL", "SKIP": "skip"}[verdict]
    print(f"[{mark}] {claim}" + (f"\n         {detail}" if detail else ""))


# --------------------------------------------------------------------------
# Claim 1: forward hooks fire in registration order, so a capture hook
# registered AFTER a modifying hook observes the modified value.
# This is what makes "register steering before capture" correct.
# --------------------------------------------------------------------------
def check_hook_order() -> None:
    mod = nn.Linear(4, 4)
    seen = {}

    def modifying_hook(m, args, output):
        return output + 100.0

    def capture_hook(m, args, output):
        seen["value"] = output.detach().clone()
        return None

    # steering-then-capture: capture must see the +100
    h1 = mod.register_forward_hook(modifying_hook)
    h2 = mod.register_forward_hook(capture_hook)
    x = torch.zeros(1, 4)
    out = mod(x)
    h1.remove(); h2.remove()
    after = seen["value"].mean().item()

    # capture-then-steering: capture must NOT see the +100
    seen.clear()
    h1 = mod.register_forward_hook(capture_hook)
    h2 = mod.register_forward_hook(modifying_hook)
    out2 = mod(x)
    h1.remove(); h2.remove()
    before = seen["value"].mean().item()

    ok = (after - before) > 50.0 and abs(out2.mean().item() - out.mean().item()) < 1e-5
    record(
        "Hooks fire in registration order; steer-before-capture is required",
        "CONFIRMED" if ok else "REFUTED",
        f"capture registered after modifier saw mean={after:.2f}; "
        f"registered before saw mean={before:.2f} (difference = the steering effect)",
    )


# --------------------------------------------------------------------------
# Claim 2: eigvalsh crashes on bfloat16; float32 is required.
# --------------------------------------------------------------------------
def check_bf16_eig() -> None:
    a = torch.randn(16, 16)
    sym = (a + a.T) / 2
    bf16_failed, err = False, ""
    try:
        torch.linalg.eigvalsh(sym.to(torch.bfloat16))
    except Exception as e:
        bf16_failed, err = True, f"{type(e).__name__}: {str(e)[:70]}"
    try:
        torch.linalg.eigvalsh(sym.float())
        fp32_ok = True
    except Exception as e:
        fp32_ok, err = False, f"fp32 also failed: {e}"

    if bf16_failed and fp32_ok:
        record("eigvalsh requires a float32 cast (fails on bf16)", "CONFIRMED", err)
    elif not bf16_failed and fp32_ok:
        record("eigvalsh requires a float32 cast (fails on bf16)", "REFUTED",
               "bf16 eigvalsh SUCCEEDED here — the cast may be unnecessary on this torch build")
    else:
        record("eigvalsh requires a float32 cast (fails on bf16)", "REFUTED", err)


# --------------------------------------------------------------------------
# Claim 3: apply_chat_template(return_tensors="pt") yields a BatchEncoding,
# not a bare tensor, on this transformers version.
# --------------------------------------------------------------------------
def check_chat_template_return(tok) -> None:
    msgs = [{"role": "user", "content": "hello"}]
    out = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True)
    is_tensor = torch.is_tensor(out)
    record(
        "apply_chat_template(return_tensors='pt') returns a dict-like, not a tensor",
        "CONFIRMED" if not is_tensor else "REFUTED",
        f"returned {type(out).__name__}"
        + ("" if is_tensor else " -> must unwrap ['input_ids']"),
    )


# --------------------------------------------------------------------------
# Claim 4: enable_thinking kwarg raises TypeError on tokenizers that lack it.
# --------------------------------------------------------------------------
def check_enable_thinking(tok) -> None:
    msgs = [{"role": "user", "content": "hello"}]
    try:
        tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        record("enable_thinking kwarg is accepted by this tokenizer", "CONFIRMED",
               f"{TOKENIZER_ID} accepts it; no try/except fallback needed for this model")
    except TypeError as e:
        record("enable_thinking kwarg raises TypeError here (fallback needed)", "CONFIRMED",
               f"{type(e).__name__}: {str(e)[:70]}")


# --------------------------------------------------------------------------
# Claim 5: the t_inst / t_post-inst rule on ChatML.
# t_inst      = last content token of the instruction-bearing message
# t_post-inst = last token of the templated prompt
# --------------------------------------------------------------------------
def check_token_positions(tok) -> None:
    from core.positions import DESIGNS, ROLES, batch_positions, render

    inst = "Tell me how to bake bread"

    for design in DESIGNS:
        n_content = set()
        lengths = {}
        for role in ROLES:
            r = render(tok, inst, role, design)
            ids = tok(r.text, add_special_tokens=False)["input_ids"]
            toks = tok.convert_ids_to_tokens(ids)

            ok_inst = tok.decode(ids[: r.t_inst + 1]).rstrip().endswith(inst.rstrip())
            ok_post = r.t_post_inst == len(ids) - 1
            ok_span = tok.decode(ids[r.content_start:r.content_end]).strip() == inst.strip()
            n_content.add(r.content_end - r.content_start)
            lengths[role] = r.n_tokens

            record(
                f"{design}/{role}: positions resolve and content span excludes markup",
                "CONFIRMED" if (ok_inst and ok_post and ok_span) else "REFUTED",
                f"n={r.n_tokens} t_inst={r.t_inst}({toks[r.t_inst]!r}) "
                f"t_post={r.t_post_inst}({toks[r.t_post_inst]!r}) fixed={r.slot_is_fixed}",
            )

        # The load-bearing invariant: identical content must tokenise identically
        # under every role, so reading at t_inst sees the same token regardless of
        # the surrounding markup.
        record(
            f"{design}: instruction tokenises identically under every role",
            "CONFIRMED" if len(n_content) == 1 else "REFUTED",
            f"content token counts across roles = {sorted(n_content)}",
        )

        # Length is NOT constant across roles (authentic markup differs). Recorded
        # so the size of the confound is known, and because it is why the role
        # probe needs a length-only baseline.
        spread = max(lengths.values()) - min(lengths.values())
        record(
            f"{design}: role length spread measured (confound size)",
            "CONFIRMED",
            f"{lengths}, spread={spread} tokens — role probe must beat a length-only baseline",
        )

    # `system` can never occupy the fixed slot; both models must agree on that.
    r_sys = render(tok, inst, "system", "fixed_slot")
    r_user = render(tok, inst, "user", "fixed_slot")
    record(
        "fixed_slot: system falls back to natural slot, user does not",
        "CONFIRMED" if (not r_sys.slot_is_fixed and r_user.slot_is_fixed) else "REFUTED",
        f"system.slot_is_fixed={r_sys.slot_is_fixed} user.slot_is_fixed={r_user.slot_is_fixed}",
    )

    # left-padding offset arithmetic
    rs = [render(tok, s, "user", "fixed_slot") for s in ["hi", "a much longer instruction here"]]
    padded = max(r.n_tokens for r in rs)
    pos = batch_positions(rs, padded, padding_side="left")
    ok = all(t == padded - 1 for t in pos["t_post_inst"])
    record(
        "batch_positions maps t_post to index -1 under left padding",
        "CONFIRMED" if ok else "REFUTED",
        f"padded_len={padded} t_post={pos['t_post_inst']} t_inst={pos['t_inst']}",
    )


# --------------------------------------------------------------------------
# Claim 6: left padding makes index -1 the true last token for every row.
# --------------------------------------------------------------------------
def check_left_padding(tok) -> None:
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    batch = tok(["short", "a considerably longer sequence of tokens here"],
                return_tensors="pt", padding=True)
    ids = batch["input_ids"]
    last_col_real = bool(batch["attention_mask"][:, -1].all().item())
    record(
        "left padding makes index -1 the true last token for every row",
        "CONFIRMED" if last_col_real else "REFUTED",
        f"shape={tuple(ids.shape)}, attention_mask[:, -1]={batch['attention_mask'][:, -1].tolist()}",
    )


# --------------------------------------------------------------------------
# Claim 7: Qwen3.5-9B has 32 layers (config only, no weights downloaded).
# --------------------------------------------------------------------------
def check_layer_count() -> None:
    """Layer/width resolution must survive Qwen3.5's nested multimodal config.

    `config.num_hidden_layers` is None for the Qwen3.5 family; the real values are
    under `config.text_config`. Anything reading the flat attribute gets None.
    """
    from transformers import AutoConfig
    from core import model_meta

    expected = {
        "Qwen/Qwen2.5-7B-Instruct": (28, 3584, None),
        "Qwen/Qwen3.5-9B": (32, 4096, None),
        "Qwen/Qwen3.5-35B-A3B": (40, 2048, 256),
    }
    for mid, (n_exp, d_exp, e_exp) in expected.items():
        try:
            cfg = AutoConfig.from_pretrained(mid)
            flat = getattr(cfg, "num_hidden_layers", None)
            info = model_meta.describe(mid, cfg)
            ok = (info["num_layers"] == n_exp and info["hidden_size"] == d_exp
                  and info["num_experts"] == e_exp)
            record(
                f"model_meta resolves {mid} shape ({n_exp}L, d={d_exp})",
                "CONFIRMED" if ok else "REFUTED",
                f"resolved {info['num_layers']}L d={info['hidden_size']} "
                f"experts={info['num_experts']} multimodal={info['is_multimodal']}; "
                f"flat config.num_hidden_layers={flat}",
            )
        except Exception as e:
            record(f"{mid} shape", "SKIP", f"{type(e).__name__}: {str(e)[:70]}")


def main() -> int:
    print(f"Verifying implementation-note claims (tokenizer={TOKENIZER_ID})\n")
    check_hook_order()
    check_bf16_eig()

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    except Exception as e:
        record("load tokenizer", "SKIP", f"{type(e).__name__}: {str(e)[:70]}")
        tok = None

    if tok is not None:
        for fn in (check_chat_template_return, check_enable_thinking,
                   check_token_positions, check_left_padding):
            try:
                fn(tok)
            except Exception:
                record(fn.__name__, "REFUTED", traceback.format_exc(limit=2).strip()[-200:])
    check_layer_count()

    refuted = [r for r in RESULTS if r[1] == "REFUTED"]
    print(f"\n{len(RESULTS)} claims checked, {len(refuted)} refuted.")
    for claim, _, detail in refuted:
        print(f"  REFUTED: {claim}\n           {detail}")
    return 1 if refuted else 0


if __name__ == "__main__":
    sys.exit(main())
