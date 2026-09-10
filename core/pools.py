"""Instruction pools and the frozen, model-independent corpus.

The corpus is **built once and shared by every model**. Instructions, labels,
sources and the train/test split are model-independent; only tokenisation and
token positions resolve per model. Rebuilding the corpus per model would let
sampling differences confound the cross-model comparison, which is the whole point
of running two models.

What is frozen is the **instruction set**, not the crossed items: roles and
designs are rendering choices, and crossing is deterministic, so storing
`instructions x 4 roles x 2 designs` would be four-fold redundancy that can drift
out of sync with the renderer.

Schemas verified against the Hub:

    walledai/AdvBench             520   prompt, target
    JailbreakBench/JBB-Behaviors    -   config "behaviors"; CAPITALISED columns
    sorry-bench/sorry-bench-202503 9240 config "default"
    tatsu-lab/alpaca            52002   instruction, input, output
    natolambert/xstest-v2-copy    450   type, prompt          (split "prompts")
    walledai/StrongREJECT         313   prompt, category, source

`walledai/XSTest` is gated; the mirror above is not. XSTest's safe subset is the
rows whose `type` does not begin with `contrast_`.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Optional sources that failed to load, `name -> error`. Recorded rather than
# raised: the pools still work without them, but silently dropping one of the
# harmful sources is a deviation that must reach the manifest.
SOURCE_FAILURES: Dict[str, str] = {}

# What each source actually contributed, and under what filter. Written into the
# frozen corpus metadata, because the selection decisions here are consequential
# and invisible in the resulting text: nothing in a Sorry-Bench prompt reveals
# that 20 mutation styles were excluded, and nothing in an XSTest prompt reveals
# whether safety came from the official `label` column or the mirror's `type`.
SOURCE_PROVENANCE: Dict[str, Dict] = {}

# Only Sorry-Bench's unmutated prompts may enter the fitting corpus. See
# `harmful_instructions` for why the other 20 styles must not.
SORRY_BENCH_STYLE = "base"

# The 20 excluded styles are not waste — they are a ready-made, held-out resource
# for later experiments, already paired with their base prompt:
#   RQ4/E4.2 jailbreaks   role_play, authority_endorsement, expert_endorsement,
#                         logical_appeal, evidence-based_persuasion, misrepresentation
#   RQ6/E6.1 contexts     the same, as controlled variants of one intent
#   out of scope          translate-*, ascii, atbash, caesar, morse
SORRY_BENCH_JAILBREAK_STYLES = (
    "role_play", "authority_endorsement", "expert_endorsement",
    "logical_appeal", "evidence-based_persuasion", "misrepresentation",
)


@dataclass(frozen=True)
class Instruction:
    text: str
    harmful: bool
    source: str          # also the Level-1 style proxy (see E1.7)
    split: str = ""      # "train" | "test", assigned at corpus build
    category: str = ""   # source-native subcategory, where one exists

    # `category` exists because the harmful class is deliberately heterogeneous.
    # Sorry-Bench spans 44 categories across a wide severity range — grooming and
    # death threats alongside 401(k) allocation advice and song lyrics, since it is
    # built to probe *over*-refusal as much as refusal. That is not a defect to
    # filter away: borderline items are what populate the harmful-and-complied
    # cell, and without them the model refuses every harmful prompt and
    # `R_control` is unidentifiable. But the heterogeneity must be measurable, so
    # separation is reported per source and per category rather than pooled.

    @property
    def uid(self) -> str:
        return hashlib.sha1(_normalize(self.text).encode()).hexdigest()[:12]


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _token_set(s: str) -> set:
    return set(_normalize(s).split())


def _dedup(items: Sequence[Instruction]) -> List[Instruction]:
    seen, out = set(), []
    for it in items:
        k = _normalize(it.text)
        if k and k not in seen:
            seen.add(k)
            out.append(it)
    return out


def _take(items: Sequence[Instruction], n: Optional[int], seed: int) -> List[Instruction]:
    """Deterministic, **source-balanced** subsample.

    Pooling every source and shuffling is proportional-to-size, so Sorry-Bench
    (9,240) swamps AdvBench (520) and Alpaca (52,002) swamps XSTest's ~250 safe
    rows. That second case is actively harmful: XSTest is the benign-but-*sensitive*
    negative, the one thing stopping `R_harm` collapsing into a "sensitive topic"
    detector, and proportional sampling discards almost all of it (measured: 197
    Alpaca / 3 XSTest at n=200). Round-robin instead.
    """
    pool = list(items)
    if n is None:
        random.Random(seed).shuffle(pool)
        return pool

    rng = random.Random(seed)
    by_source: Dict[str, List[Instruction]] = {}
    for it in pool:
        by_source.setdefault(it.source, []).append(it)
    for lst in by_source.values():
        rng.shuffle(lst)

    out: List[Instruction] = []
    order = sorted(by_source)
    cur = {s: 0 for s in order}
    while len(out) < n and any(cur[s] < len(by_source[s]) for s in order):
        for s in order:
            if len(out) >= n:
                break
            if cur[s] < len(by_source[s]):
                out.append(by_source[s][cur[s]])
                cur[s] += 1
    return out


def _load(path: str, *args, **kwargs):
    from datasets import load_dataset
    return load_dataset(path, *args, **kwargs)


def _text_column(ds) -> str:
    """Case-insensitive: JBB-Behaviors capitalises its columns (`Goal`, not
    `goal`), and matching case-sensitively silently dropped that whole source."""
    lookup = {c.lower(): c for c in ds.column_names}
    for cand in ("prompt", "goal", "instruction", "behavior", "question", "turns", "text"):
        if cand in lookup:
            return lookup[cand]
    raise ValueError(f"no recognisable text column in {ds.column_names}")


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
def harmful_instructions(n: Optional[int] = None, seed: int = 0) -> List[Instruction]:
    """AdvBench + JBB + Sorry-Bench — the three harmful sources Zhao draw on."""
    adv = [Instruction(r["prompt"], True, "advbench") for r in _load("walledai/AdvBench")["train"]]
    SOURCE_PROVENANCE["advbench"] = {
        "dataset": "walledai/AdvBench", "split": "train", "filter": None, "n_available": len(adv)}
    out = list(adv)

    try:
        d = _load("JailbreakBench/JBB-Behaviors", "behaviors")
        split = d[list(d.keys())[0]]
        col = _text_column(split)          # capitalised: "Goal"
        for r in split:
            if isinstance(r[col], str) and r[col].strip():
                out.append(Instruction(r[col], True, "jbb", category=str(r.get("Category", ""))))
        SOURCE_PROVENANCE["jbb"] = {
            "dataset": "JailbreakBench/JBB-Behaviors", "config": "behaviors",
            "text_column": col, "filter": None, "n_available": len(split)}
    except Exception as e:
        SOURCE_FAILURES["JailbreakBench/JBB-Behaviors"] = f"{type(e).__name__}: {str(e)[:120]}"

    # Sorry-Bench ships 9,240 rows = 440 base prompts x 21 `prompt_style`
    # MUTATIONS, and taking them indiscriminately is wrong three times over:
    #   * `role_play`, `authority_endorsement`, `expert_endorsement`,
    #     `logical_appeal`, `evidence-based_persuasion`, `misrepresentation` are
    #     jailbreak framings — fitting `R_harm`/`R_control` on them would make
    #     RQ4's jailbreak family partly circular and breaks the plan's
    #     "completely disjoint attack intents";
    #   * `translate-fr|ml|mr|ta|zh-cn` are multilingual, which the plan lists as
    #     explicitly out of scope;
    #   * `ascii`, `atbash`, `caesar`, `morse` are encoded strings, not natural
    #     harmful instructions, and blow the length distribution apart (measured:
    #     mean 260 tokens vs ~12 for every other source, max 3053).
    # Only `base` is a clean harmful-instruction pool.
    try:
        d = _load("sorry-bench/sorry-bench-202503", "default")
        split = d[list(d.keys())[0]]
        if "prompt_style" not in split.column_names:
            raise ValueError("prompt_style column missing; cannot isolate base prompts")
        for r in split:
            if r["prompt_style"] != SORRY_BENCH_STYLE:
                continue
            v = r["turns"]
            v = (v[0] if v else "") if isinstance(v, list) else v
            if isinstance(v, str) and v.strip():
                out.append(Instruction(v, True, "sorry-bench", category=f"cat{r['category']}"))
        SOURCE_PROVENANCE["sorry-bench"] = {
            "dataset": "sorry-bench/sorry-bench-202503", "config": "default",
            "filter": f"prompt_style == {SORRY_BENCH_STYLE!r}",
            "n_available_all_styles": len(split),
            "n_available_after_filter": sum(1 for r in split if r["prompt_style"] == SORRY_BENCH_STYLE),
            "excluded_styles_note": "20 mutation styles withheld: jailbreak framings "
                                    "(reserved for RQ4/RQ6), multilingual and encoded "
                                    "(out of scope)"}
    except Exception as e:
        SOURCE_FAILURES["sorry-bench/sorry-bench-202503"] = f"{type(e).__name__}: {str(e)[:120]}"

    return _take(_dedup(out), n, seed)


def harmless_instructions(n: Optional[int] = None, seed: int = 0) -> List[Instruction]:
    """Alpaca (standalone rows only) + XSTest's safe subset.

    XSTest is benign-but-*sensitive* and therefore the hard negative; Zhao use it
    for the same reason.
    """
    alp = [Instruction(r["instruction"], False, "alpaca")
           for r in _load("tatsu-lab/alpaca")["train"] if not r["input"].strip()]
    SOURCE_PROVENANCE["alpaca"] = {
        "dataset": "tatsu-lab/alpaca", "split": "train",
        "filter": "input == '' (standalone instructions only)", "n_available": len(alp)}
    out = list(alp)

    # Prefer the official set: it carries an explicit `label` column (250 safe /
    # 200 unsafe), so the safe subset is read rather than inferred. It is gated,
    # hence the mirror fallback, where safety must be derived from `type` — the
    # eight `contrast_*` types are the unsafe ones.
    try:
        ds = _load("walledai/XSTest")
        split = ds[list(ds.keys())[0]]
        if "label" in split.column_names:
            rows = [(r["prompt"], str(r.get("type", ""))) for r in split if r["label"] == "safe"]
        else:
            rows = [(r["prompt"], str(r["type"])) for r in split if not str(r["type"]).startswith("contrast_")]
        out += [Instruction(t, False, "xstest_safe", category=c) for t, c in rows]
        SOURCE_PROVENANCE["xstest_safe"] = {
            "dataset": "walledai/XSTest", "split": list(ds.keys())[0],
            "safety_from": "label" if "label" in split.column_names else "type-prefix",
            "filter": "label == 'safe'" if "label" in split.column_names
                      else "not type.startswith('contrast_')",
            "n_available": len(rows)}
    except Exception as official_err:
        try:
            for r in _load("natolambert/xstest-v2-copy")["prompts"]:
                if not str(r["type"]).startswith("contrast_"):
                    out.append(Instruction(r["prompt"], False, "xstest_safe"))
            SOURCE_PROVENANCE["xstest_safe"] = {
                "dataset": "natolambert/xstest-v2-copy", "split": "prompts",
                "safety_from": "type-prefix", "filter": "not type.startswith('contrast_')"}
            SOURCE_FAILURES["walledai/XSTest"] = (
                f"gated ({type(official_err).__name__}); used mirror "
                f"natolambert/xstest-v2-copy with type-derived safety")
        except Exception as e:
            SOURCE_FAILURES["XSTest"] = f"{type(e).__name__}: {str(e)[:120]}"

    return _take(_dedup(out), n, seed)


def attack_intents(
    n: Optional[int] = None,
    seed: int = 0,
    exclude: Optional[Sequence[Instruction]] = None,
    jaccard_threshold: float = 0.8,
) -> List[Instruction]:
    """Held-out attack intents (StrongREJECT), disjoint from the fitting pool.

    A different dataset is not a disjoint dataset: StrongREJECT aggregates from
    other benchmarks and does overlap AdvBench — measured at 3 exact and 4
    near-duplicate collisions against a 400-instruction fitting pool. Disjointness
    is enforced here by construction rather than checked afterwards.
    """
    items = _dedup([Instruction(r["prompt"], True, "strongreject")
                    for r in _load("walledai/StrongREJECT")["train"]])
    SOURCE_PROVENANCE["strongreject"] = {
        "dataset": "walledai/StrongREJECT", "split": "train",
        "filter": f"exclude fitting pool (exact + token-Jaccard >= {jaccard_threshold})",
        "n_available": len(items)}

    if exclude:
        excl_norm = {_normalize(i.text) for i in exclude}
        excl_tokens = [_token_set(i.text) for i in exclude]

        def collides(t: str) -> bool:
            if _normalize(t) in excl_norm:
                return True
            ts = _token_set(t)
            return bool(ts) and any(
                len(ts & f) / len(ts | f) >= jaccard_threshold for f in excl_tokens if (ts | f)
            )

        items = [i for i in items if not collides(i.text)]

    return _take(items, n, seed)


def webtext_documents(n: int = 150, seed: int = 0, max_chars: int = 2000) -> List[str]:
    """Non-instruct webtext for the role paper's constant-content design.

    C4 requires the config name `en`; passing "en" as a *split* is the error that
    made it look unavailable. Streamed, so nothing large is downloaded.
    """
    out: List[str] = []
    for row in _load("allenai/c4", "en", streaming=True)["validation"]:
        t = (row.get("text") or "").strip()
        if len(t) > 200:
            out.append(t[:max_chars])
        if len(out) >= n:
            break
    return out


def transfer_corpus(
    n: int = 150,
    seed: int = 0,
    min_words: int = 25,
    max_words: int = 60,
    train_fraction: float = 0.75,
) -> List[Instruction]:
    """E1.0b — the constant-content role corpus, for the cross-corpus transfer check.

    E1.1 accepts a role direction only if it transfers between two corpora that
    share role marking and share nothing else. Our crossed corpus is *instructions*
    under role tags; this one is **C4 prose** under the same tags. A probe that has
    learned "instruction-ness", turn position, or harm rather than role will
    classify one and fail the other, and that is the failure the check exists to
    catch — which is why the transfer set must not be instruction-shaped.

    Content is held constant across roles by construction: one passage is rendered
    under every role, exactly as the role paper does it. Passages are truncated by
    **word** count, not tokens, so the frozen corpus stays model-independent and
    both models receive byte-identical text.

    Splits are assigned here because transfer is tested in *both* directions
    (PLAN-EXTRACT), and the crossed -> C4 direction needs held-out C4 items.
    """
    docs = webtext_documents(n=n * 4, seed=seed)      # over-draw; truncation rejects some

    out: List[Instruction] = []
    for d in docs:
        # First paragraph only: C4 documents concatenate unrelated sections, and a
        # passage spanning a topic break is not one constant content.
        para = next((p.strip() for p in d.split("\n") if len(p.split()) >= min_words), None)
        if para is None:
            continue
        words = para.split()
        if len(words) < min_words:
            continue
        text = " ".join(words[:max_words])
        if _has_chat_special_tokens(text):
            continue                                   # would corrupt the role marking
        out.append(Instruction(text, False, "c4"))
        if len(out) >= n:
            break

    return assign_splits(_dedup(out), train_fraction, seed)


# Any of these inside an instruction would let the content forge or terminate a
# role turn, so the role marking under test would no longer be what we set.
CHAT_SPECIAL_TOKENS = (
    "<|im_start|>", "<|im_end|>", "<|endoftext|>",
    "<think>", "</think>", "<tool_response>", "</tool_response>",
)


def _has_chat_special_tokens(text: str) -> bool:
    return any(t in text for t in CHAT_SPECIAL_TOKENS)


# ---------------------------------------------------------------------------
# corpus assembly
# ---------------------------------------------------------------------------
def assign_splits(
    instructions: Sequence[Instruction],
    train_fraction: float = 0.75,
    seed: int = 0,
) -> List[Instruction]:
    """Split **by instruction**, stratified by (harm label, source).

    Every role/design rendering of one instruction must land on the same side: a
    per-item split would put the `user` rendering in train and the `tool` rendering
    in test, letting a probe score by memorising content.

    Stratifying by label alone is not enough. The sources differ in character —
    AdvBench is terse imperatives, Sorry-Bench spans a wide severity range, XSTest
    is benign-but-sensitive — so an unstratified draw lets source proportions drift
    between the sides (measured: AdvBench 28% of train-harmful but 50% of
    test-harmful). Held-out AUC would then partly reflect a distribution shift
    rather than generalisation. `source` is also the Level-1 style proxy, so
    keeping it balanced protects E1.7 as well.
    """
    rng = random.Random(seed)
    by_stratum: Dict[Tuple[bool, str], List[str]] = {}
    seen = set()
    for it in instructions:
        if it.uid not in seen:
            seen.add(it.uid)
            by_stratum.setdefault((it.harmful, it.source), []).append(it.uid)

    train: set = set()
    for key in sorted(by_stratum, key=lambda k: (k[0], k[1])):
        uids = sorted(by_stratum[key])
        rng.shuffle(uids)
        train.update(uids[: int(round(len(uids) * train_fraction))])

    return [Instruction(i.text, i.harmful, i.source,
                        "train" if i.uid in train else "test", i.category)
            for i in instructions]


def build_corpus(
    n_harmful: int = 200,
    n_harmless: int = 200,
    n_attack: int = 150,
    train_fraction: float = 0.75,
    seed: int = 0,
) -> Tuple[List[Instruction], List[Instruction]]:
    """The frozen, model-independent corpus: fitting instructions + attack pool."""
    bad = harmful_instructions(n_harmful, seed=seed)
    good = harmless_instructions(n_harmless, seed=seed)
    if len(bad) < n_harmful or len(good) < n_harmless:
        raise ValueError(f"insufficient instructions: harmful {len(bad)}/{n_harmful}, "
                         f"harmless {len(good)}/{n_harmless}")

    fitting = assign_splits([*bad, *good], train_fraction, seed)
    attacks = attack_intents(n_attack, seed=seed, exclude=fitting)
    if len(attacks) < n_attack:
        raise ValueError(f"attack pool too small after disjointness filtering: "
                         f"{len(attacks)}/{n_attack}")
    return fitting, attacks


# ---------------------------------------------------------------------------
# E1.7 Level 2 — the controlled-style corpus
# ---------------------------------------------------------------------------
# Level 1 asks whether `R_role` correlates with linguistic style across the
# corpus as it happens to be. It cannot separate them, because in natural data
# the role TAG and the register of the content are not independent. Level 2
# crosses them: the same request, rewritten into three registers, placed under
# every role tag. If `R_role` is metadata it tracks the tag and is largely
# indifferent to register; if it is a style detector, the reverse.
#
# **The rewrites are model-GENERATED, not templated.** An earlier draft used
# deterministic templates with the instruction embedded verbatim, which is
# cheaper and makes content preservation true by construction. It was rejected
# because it narrows the claim: with a template the register lives entirely in a
# fixed framing phrase at the edges, so the fitted "style" contrast is really a
# framing-phrase contrast, and orthogonality to it would only license "R_role is
# not the framing phrase" — not the claim PLAN-EXTRACT actually asks for, about
# linguistic style. Generation redistributes register through the text, which is
# the real competitor hypothesis.
#
# The two costs of generating are handled rather than avoided: the corpus is
# FROZEN to JSONL so both models see byte-identical text and reruns are stable,
# and content drift — which would confound precisely the contrast being isolated
# — is verified rather than assumed (`verify_style_items`).
#
# `TEMPLATE_REGISTERS` survives as a CONTROL ARM, not the main design. Templated
# register is trivially separable, so it upper-bounds how detectable style can
# be; if `R_role` is orthogonal even to that, the generated result is easier to
# read.
STYLE_REGISTERS: Dict[str, str] = {
    "system_register":
        "Rewrite the following request so it reads like a system prompt or "
        "operating directive: imperative, second person, policy voice. Keep the "
        "request itself completely unchanged in meaning and specifics. Output "
        "only the rewrite.",
    "user_register":
        "Rewrite the following request so it reads like a casual message from a "
        "person: conversational, first person, hedged. Keep the request itself "
        "completely unchanged in meaning and specifics. Output only the rewrite.",
    "tool_register":
        "Rewrite the following request so it reads like structured output from a "
        "tool or API: third person, machine-readable, no direct address. Keep the "
        "request itself completely unchanged in meaning and specifics. Output "
        "only the rewrite.",
}

# Control arm — deterministic, content embedded verbatim.
TEMPLATE_REGISTERS: Dict[str, List[str]] = {
    "system_register": ["You are required to address the following request. {c}",
                        "Operating directive: {c} Respond in accordance with policy."],
    "user_register":   ["Hey, could you help me with something? {c}",
                        "I was wondering — {c} Thanks!"],
    "tool_register":   ['RESULT: retrieved_request="{c}" status=awaiting_response',
                        "[api_response] payload: {c} | handler: pending"],
}

_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "so", "that", "the", "then", "this", "to", "was", "what", "when",
    "which", "will", "with", "would", "you", "your",
}


def _request_text(text: str) -> str:
    """The content-bearing part of a rewrite, with structural markup removed.

    A structured register carries its register in KEYS and its content in VALUES:

        {"request": "Describe the three branches of the US government.",
         "subject": "US government", "task": "describe"}

    Counting `request`, `subject` and `task` as content makes every JSON rewrite
    look like it invented material, which is how the `tool_register` arm kept
    failing an *added content* bound while scoring coverage 1.0 — content
    perfectly preserved, structure miscounted as content. Measured, not guessed:
    31 of 38 rejects in the diagnostic run were `added_content` at coverage 1.0.

    So for JSON-ish output only the values are returned. This is the same
    principle as the register/tag crossing itself — the check must be blind to
    register, or it penalises the registers that differ most from prose.
    """
    t = text.strip()
    if t.startswith(("{", "[")):
        try:
            import json as _json
            parsed = _json.loads(t)
        except Exception:
            # Unparseable (truncated, trailing commas): fall back to stripping
            # anything in the `"key":` position, which is the same intent.
            return re.sub(r'"[^"]*"\s*:', " ", t)
        vals: List[str] = []

        def _walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    _walk(v)
            elif isinstance(o, list):
                for v in o:
                    _walk(v)
            elif isinstance(o, str):
                vals.append(o)
            elif o is not None:
                vals.append(str(o))
        _walk(parsed)
        return " ".join(vals)
    return t


def _content_words(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9]+", _request_text(text).lower())
            if w not in _STOPWORDS and len(w) > 2}


@dataclass(frozen=True)
class StyleItem:
    """One base instruction rewritten into one register."""
    text: str
    base_uid: str
    base_text: str
    register: str
    arm: str             # "generated" | "template"
    template_id: int     # -1 for generated
    harmful: bool
    source: str
    split: str
    coverage: float = float("nan")   # content-word recall against the base
    reject_reason: str = ""          # "" when kept; else ";"-joined reason codes

    @property
    def uid(self) -> str:
        return f"{self.base_uid}:{self.arm}:{self.register}:{self.template_id}"


# Level 2 rewrites HARMLESS instructions only. Measured, not assumed: an aligned
# rewriter refuses to restyle harmful requests ("Instruct someone how to commit a
# violent crime" -> "I cannot provide instructions for..."), so harmful bases fail
# content verification systematically and the survivors are a biased subset —
# precisely the failure `verify_style_items` exists to catch, and it did.
#
# This costs nothing the experiment needs. Level 2 asks whether `R_role` tracks
# the role TAG or the linguistic REGISTER; harmfulness is not a term in that
# question, and `R_role` is estimated from role contrasts regardless of harm.
# Level 1 still covers the full corpus including harmful items. The limitation to
# state is narrow: Level 2 cannot detect a `R_role` that behaves differently on
# harmful content.
STYLE_MIN_CONTENT_WORDS = 4


def select_style_bases(fitting: Sequence[Instruction], n_base: int,
                       seed: int = 0, harmless_only: bool = True,
                       min_content_words: int = STYLE_MIN_CONTENT_WORDS
                       ) -> List[Instruction]:
    """Stratified subsample of instructions to rewrite.

    Stratified by split so the style corpus inherits the fitting corpus's
    train/test boundary — an instruction that is test-side there must stay
    test-side here, or a role contrast fitted on this corpus could be evaluated
    on prompts it was fitted on.

    `min_content_words` keeps out bases so short that coverage is unstable: with
    four content words, a rewrite that drops one scores 0.75 and a legitimate
    paraphrase can fail verification for arithmetic rather than for drift.
    """
    elig = [i for i in fitting
            if (not harmless_only or not i.harmful)
            and len(_content_words(i.text)) >= min_content_words]
    by: Dict[Tuple[bool, str], List[Instruction]] = {}
    for ins in elig:
        by.setdefault((bool(ins.harmful), ins.split), []).append(ins)
    rng = random.Random(seed)
    per = max(1, n_base // max(len(by), 1))
    base: List[Instruction] = []
    for k in sorted(by, key=lambda t: (t[0], t[1])):
        pool = sorted(by[k], key=lambda i: i.uid)
        rng.shuffle(pool)
        base.extend(pool[:per])
    return sorted(base, key=lambda i: i.uid)[:n_base]


def build_template_arm(bases: Sequence[Instruction]) -> List[StyleItem]:
    """The control arm: deterministic rewrites, content verbatim."""
    out: List[StyleItem] = []
    for ins in bases:
        for reg, templates in TEMPLATE_REGISTERS.items():
            for ti, tpl in enumerate(templates):
                out.append(StyleItem(
                    text=tpl.format(c=ins.text), base_uid=ins.uid, base_text=ins.text,
                    register=reg, arm="template", template_id=ti,
                    harmful=bool(ins.harmful), source=ins.source, split=ins.split,
                    coverage=1.0))
    return out


# Negations, for the inversion check below. Observed failure mode: asked for a
# system-register rewrite of "What's the best way of destroying my opponent in
# chess?", the rewriter returned "You must NOT seek to destroy your opponent" —
# same words, opposite request.
_NEGATIONS = {"not", "cannot", "can't", "don't", "never", "prohibited",
              "forbidden", "refuse", "unable", "won't", "shouldn't", "must not"}


def verify_style_items(items: Sequence[StyleItem], min_coverage: float = 0.25,
                       max_added_content: float = 3.0) -> Tuple[List[StyleItem], Dict]:
    """Keep only rewrites that still request the same thing.

    **The threshold is deliberately permissive, and that is a correction.** An
    earlier version required 75% content-word overlap, which measured the wrong
    thing: a genuine register change necessarily rewords, so lexical overlap
    penalises precisely the rewrites that restyle most thoroughly. Measured on
    the first build — `tool_register` was bimodal, passing at coverage 1.0 when
    the model quoted the request verbatim inside JSON and failing when it
    genuinely restructured into fields. That gate was rewarding rewrites closest
    to templating, which is the opposite of what Level 2 needs.

    So coverage is kept only as a catastrophic-loss floor, and the two failures
    that actually corrupt the contrast get targeted checks:

    * **refusal** — the rewriter declined. The item does not exist.
    * **inversion** — a negation appears that the base did not have, so the
      rewrite requests the opposite. Same words, opposite meaning; coverage
      cannot see this at all.
    * **added content** — wholesale addition of material the base did not have.
      Counted in CONTENT WORDS, not characters. A character length-ratio bound
      was tried first and rejected `tool_register` almost entirely: JSON is
      inherently verbose, so a 44-character request becomes a 250-character
      object while adding only a handful of field names. Characters measured
      structure; the stated concern was added *content*, so content words are
      what the bound now counts.

    **A limitation to state in the paper, not to tune away.** Lexical overlap
    cannot separate a faithful paraphrase from a changed request: "the best way
    to break into the film industry" and "the most effective approach to entering
    the film industry" share two content words out of six. The floor is therefore
    set low enough to admit such paraphrases, which necessarily also admits some
    drift. Three things bound the residual risk, and none is a threshold:

    * the register contrast is fitted PAIRED WITHIN BASE, so content common to a
      base's rewrites cancels;
    * the template arm holds content exactly verbatim, so comparing the two arms
      shows what paraphrase freedom bought or cost;
    * `coverage` is recorded per item, so the corpus's lexical fidelity is a
      reported distribution rather than an assumption.

    A semantic-equivalence judge would measure this properly, but it is itself an
    unvalidated instrument, and validating it is a larger job than the experiment
    it would serve.

    Rejects are returned in the report rather than silently dropped, because the
    rejection rate and its causes are properties of the corpus worth stating.
    """
    from core.refusal import PRIMARY_RULE, has_refusal_marker

    kept, rejected = [], []
    n_refusal = n_inverted = 0
    for it in items:
        base_words = _content_words(it.base_text)
        cov = (len(base_words & _content_words(it.text)) / len(base_words)) if base_words else 1.0
        # TOTAL size, not the size of the added SET. An added-set bound fired on
        # register twice over — JSON field names, then conversational hedging
        # ("do you happen to know of any good spots around here") — because every
        # register adds words of its own; that is what a register IS. The failure
        # actually worth catching is a rewrite that answers the request or writes
        # an essay instead of restyling it, and that shows up as overall size.
        rw_words = _content_words(it.text)
        added_ok = len(rw_words) <= max(12, max_added_content * max(len(base_words), 1))
        # Coverage is a FRACTION, so on a short base it moves in coarse steps: with
        # four content words a single dropped word scores 0.75 and a perfectly good
        # paraphrase fails for arithmetic rather than for drift. Allow one missing
        # content word outright, which is the same tolerance the fraction encodes
        # on longer bases. Requiring 6+ content words instead would have discarded
        # 122 of the 200 harmless instructions.
        n_missing = len(base_words - _content_words(it.text))
        # POLARITY, not token identity. Comparing negation token SETS flagged
        # "why can X not resurrect the dead" -> "why is it that X can't bring
        # the dead back" as an inversion, because `not` became `can't`. Same
        # polarity, different word. What corrupts the contrast is a request
        # acquiring a negation it did not have, so that is what is tested.
        low = {w for w in re.findall(r"[a-z']+", _request_text(it.text).lower())}
        base_low = {w for w in re.findall(r"[a-z']+", it.base_text.lower())}
        inverted = bool(low & _NEGATIONS) and not bool(base_low & _NEGATIONS)
        refused = has_refusal_marker(it.text, PRIMARY_RULE)
        ok = ((cov >= min_coverage or n_missing <= 1)
              and added_ok
              and not inverted and not refused)
        rec = StyleItem(**{**asdict(it), "coverage": round(cov, 4)})
        # Per-item reason codes, so a rejection can be diagnosed instead of
        # guessed at. Four rounds of threshold tuning happened without ever
        # looking at the rejected text; this exists so that cannot recur.
        reasons = []
        if refused:   reasons.append("refusal")
        if inverted:  reasons.append("inversion")
        if not (cov >= min_coverage or n_missing <= 1): reasons.append("coverage")
        if not added_ok: reasons.append("added_content")
        rec = StyleItem(**{**asdict(rec), "reject_reason": ";".join(reasons)})
        if ok:
            kept.append(rec)
        else:
            # Two different failures wear the same coverage score, and conflating
            # them hides the more serious one. A REWRITER REFUSAL means the corpus
            # cannot be built for that base at all; content DRIFT means the rewrite
            # was attempted and wandered. Only the first is a reason to change the
            # rewriter or the base pool.
            if refused:
                n_refusal += 1
            if inverted:
                n_inverted += 1
            rejected.append(rec)
    by_reg: Dict[str, int] = {}
    for r in rejected:
        by_reg[r.register] = by_reg.get(r.register, 0) + 1
    report = {
        "n_in": len(items), "n_kept": len(kept), "n_rejected": len(rejected),
        "rejection_rate": round(len(rejected) / max(len(items), 1), 4),
        "rejected_by_register": by_reg,
        "n_rejected_rewriter_refused": n_refusal,
        "n_rejected_inverted": n_inverted,
        "n_rejected_content_drift": len(rejected) - n_refusal,
        "min_coverage": min_coverage, "max_added_content": max_added_content,
        "mean_coverage_kept": round(
            sum(k.coverage for k in kept) / max(len(kept), 1), 4),
        "rejected_by_reason": {
            k: sum(1 for r in rejected if k in (r.reject_reason or ""))
            for k in ("refusal", "inversion", "coverage", "added_content")},
        "rejected_by_register_reason": {
            f"{r.register}:{r.reject_reason}":
                sum(1 for q in rejected if q.register == r.register
                    and q.reject_reason == r.reject_reason)
            for r in rejected},
        "examples_rejected": [{"uid": r.uid, "coverage": r.coverage,
                               "register": r.register, "reason": r.reject_reason,
                               "base": r.base_text[:90], "rewrite": r.text[:120]}
                              for r in rejected[:5]],
        "all_rejected": [{"uid": r.uid, "register": r.register,
                          "reason": r.reject_reason, "coverage": r.coverage,
                          "base": r.base_text, "rewrite": r.text}
                         for r in rejected],
    }
    return kept, report


def require_complete_registers(items: Sequence[StyleItem],
                               registers: Sequence[str]) -> Tuple[List[StyleItem], Dict]:
    """Keep only bases that survived verification in EVERY register.

    This is what makes the crossing balanced, and it replaces the earlier
    per-register retention threshold. A per-register gate asked "did enough items
    survive in each register?", which can be satisfied while the survivors in one
    register are a different set of bases from the survivors in another — and
    then a register contrast is partly a contrast between different requests,
    which is the confound the whole design exists to remove.

    Requiring complete sets makes content identical across registers by
    construction, and makes the paired within-base contrast well defined.
    """
    by_base: Dict[str, Dict[str, StyleItem]] = {}
    for it in items:
        by_base.setdefault(it.base_uid, {})[it.register] = it
    need = set(registers)
    complete = [b for b, d in by_base.items() if need <= set(d)]
    kept = [d[r] for b, d in sorted(by_base.items()) if b in set(complete)
            for r in sorted(d) if r in need]
    dropped = {b: sorted(need - set(d)) for b, d in by_base.items() if b not in set(complete)}
    report = {
        "n_bases_seen": len(by_base), "n_bases_complete": len(complete),
        "n_items_kept": len(kept),
        "completion_rate": round(len(complete) / max(len(by_base), 1), 4),
        "missing_register_counts": {
            r: sum(1 for v in dropped.values() if r in v) for r in sorted(need)},
    }
    return kept, report


def save_style_corpus(out_dir: Path, items: Sequence[StyleItem], meta: Dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "style_corpus.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(asdict(it)) + "\n")
    (out_dir / "style_corpus_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def load_style_corpus(out_dir: Path) -> List[StyleItem]:
    items = []
    with open(out_dir / "style_corpus.jsonl") as f:
        for line in f:
            if line.strip():
                items.append(StyleItem(**json.loads(line)))
    return items


# ---------------------------------------------------------------------------
# freeze / load
# ---------------------------------------------------------------------------
def save_corpus(out_dir: Path, fitting: Sequence[Instruction],
                attacks: Sequence[Instruction], meta: Dict,
                transfer: Optional[Sequence[Instruction]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = [("instructions", fitting), ("attack_intents", attacks)]
    if transfer is not None:
        parts.append(("transfer_corpus", transfer))
    for name, items in parts:
        rows = [{**asdict(i), "uid": i.uid} for i in items]
        (out_dir / f"{name}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    (out_dir / "corpus_meta.json").write_text(json.dumps(meta, indent=2, default=str))


def _read_jsonl(path: Path) -> List[Instruction]:
    return [Instruction(r["text"], r["harmful"], r["source"], r.get("split", ""), r.get("category", ""))
            for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())]


def load_corpus(out_dir: Path) -> Tuple[List[Instruction], List[Instruction], Dict]:
    meta = json.loads((out_dir / "corpus_meta.json").read_text())
    return (_read_jsonl(out_dir / "instructions.jsonl"),
            _read_jsonl(out_dir / "attack_intents.jsonl"), meta)


def load_transfer_corpus(out_dir: Path) -> List[Instruction]:
    return _read_jsonl(out_dir / "transfer_corpus.jsonl")


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------
def describe_pool(items: Iterable[Instruction]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.source] = counts.get(it.source, 0) + 1
    return dict(sorted(counts.items()))


def verify_corpus(
    fitting: Sequence[Instruction],
    attacks: Sequence[Instruction],
    jaccard_threshold: float = 0.8,
) -> Dict[str, Dict]:
    """Every offline property the corpus must have. Each entry carries `ok`."""
    checks: Dict[str, Dict] = {}

    uids = [i.uid for i in fitting]
    checks["no_duplicate_instructions"] = {
        "ok": len(uids) == len(set(uids)), "n": len(uids), "unique": len(set(uids))}

    tr = [i for i in fitting if i.split == "train"]
    te = [i for i in fitting if i.split == "test"]
    leak = {i.uid for i in tr} & {i.uid for i in te}
    checks["split_no_leak"] = {"ok": not leak, "n_train": len(tr), "n_test": len(te),
                               "leaked": len(leak)}
    checks["split_labelled"] = {"ok": all(i.split in ("train", "test") for i in fitting)}

    def frac(xs):
        return round(sum(x.harmful for x in xs) / len(xs), 3) if xs else None
    checks["split_label_balance"] = {
        "ok": bool(tr and te and abs(frac(tr) - frac(te)) < 0.05),
        "train_harmful_frac": frac(tr), "test_harmful_frac": frac(te)}

    # Source proportions must match across the split, or held-out AUC partly
    # measures a distribution shift rather than generalisation.
    def shares(xs):
        c = describe_pool(xs)
        n = sum(c.values()) or 1
        return {k: round(v / n, 3) for k, v in c.items()}
    s_tr, s_te = shares(tr), shares(te)
    drift = {k: round(abs(s_tr.get(k, 0) - s_te.get(k, 0)), 3)
             for k in set(s_tr) | set(s_te)}
    checks["split_source_balance"] = {
        "ok": all(v <= 0.10 for v in drift.values()),
        "max_share_drift": max(drift.values()) if drift else 0.0,
        "train_shares": s_tr, "test_shares": s_te}

    bad_src = describe_pool([i for i in fitting if i.harmful])
    good_src = describe_pool([i for i in fitting if not i.harmful])

    def balanced(d):
        return (max(d.values()) - min(d.values())) <= max(2, 0.2 * max(d.values())) if d else False
    checks["source_balance_harmful"] = {"ok": balanced(bad_src), **bad_src}
    checks["source_balance_harmless"] = {"ok": balanced(good_src), **good_src}

    fit_norm = {_normalize(i.text) for i in fitting}
    fit_tokens = [_token_set(i.text) for i in fitting]
    exact = sum(_normalize(a.text) in fit_norm for a in attacks)
    near = 0
    for a in attacks:
        at = _token_set(a.text)
        if at and any(len(at & f) / len(at | f) >= jaccard_threshold for f in fit_tokens if (at | f)):
            near += 1
    checks["attack_pool_disjoint"] = {
        "ok": exact == 0 and near == 0, "exact": exact, "near_duplicate": near,
        "n_attack": len(attacks), "threshold": jaccard_threshold}

    # Length balance between the classes. If harmful prompts are systematically
    # longer than harmless ones, `R_harm` partly encodes length rather than harm —
    # and this is not hypothetical: pulling Sorry-Bench's mutated styles gave
    # harmful a mean of 95 tokens against 11 for harmless (8.4x), driven by
    # encoded and translated variants up to 3,053 tokens. Word count is used as a
    # tokenizer-free proxy so the corpus stays model-independent.
    def _wl(xs):
        return sorted(len(x.text.split()) for x in xs)

    def _median(v):
        return v[len(v) // 2] if v else 0.0
    lh, ln = _wl([i for i in fitting if i.harmful]), _wl([i for i in fitting if not i.harmful])
    med_h, med_n = _median(lh), _median(ln)
    ratio = (med_h / med_n) if med_n else float("inf")
    all_lens = sorted(lh + ln)   # concatenating two sorted lists is not sorted
    outlier = (max(all_lens) / _median(all_lens)) if all_lens and _median(all_lens) else float("inf")
    checks["length_balance"] = {
        "ok": 0.5 <= ratio <= 2.0 and outlier <= 10.0,
        "median_words_harmful": med_h, "median_words_harmless": med_n,
        "median_ratio": round(ratio, 2),
        "max_words": max(all_lens) if all_lens else 0,
        "max_over_median": round(outlier, 1)}

    checks["source_recorded"] = {"ok": all(i.source for i in fitting),
                                 "distinct_sources": len({i.source for i in fitting})}

    # An instruction containing a chat special token could close its own turn and
    # open another, so the role marking under test would not be the one we set —
    # a template injection inside our own corpus. Cheap to check, catastrophic to
    # miss, and it would look like a role effect rather than like a bug.
    offenders = [i.uid for i in [*fitting, *attacks] if _has_chat_special_tokens(i.text)]
    checks["no_chat_special_tokens"] = {"ok": not offenders, "n_offending": len(offenders),
                                        "uids": offenders[:5]}

    checks["dataset_sources_loaded"] = {"ok": not SOURCE_FAILURES, **SOURCE_FAILURES}
    return checks


def verify_transfer_corpus(
    transfer: Sequence[Instruction],
    fitting: Sequence[Instruction],
    min_words: int = 25,
) -> Dict[str, Dict]:
    """Offline properties of the E1.0b transfer corpus."""
    checks: Dict[str, Dict] = {}
    uids = [i.uid for i in transfer]
    checks["transfer_no_duplicates"] = {
        "ok": len(uids) == len(set(uids)), "n": len(uids), "unique": len(set(uids))}

    tr = [i for i in transfer if i.split == "train"]
    te = [i for i in transfer if i.split == "test"]
    checks["transfer_split"] = {
        "ok": bool(tr and te) and not ({i.uid for i in tr} & {i.uid for i in te}),
        "n_train": len(tr), "n_test": len(te)}

    wl = [len(i.text.split()) for i in transfer]
    checks["transfer_length"] = {
        "ok": bool(wl) and min(wl) >= min_words,
        "min_words": min(wl) if wl else 0, "max_words": max(wl) if wl else 0}

    # The transfer set must share role marking with the crossed corpus and nothing
    # else. Any shared content would let a memorising probe "transfer" for the
    # wrong reason.
    fit_norm = {_normalize(i.text) for i in fitting}
    overlap = sum(_normalize(i.text) in fit_norm for i in transfer)
    checks["transfer_disjoint_from_fitting"] = {"ok": overlap == 0, "overlap": overlap}

    checks["transfer_no_chat_special_tokens"] = {
        "ok": not any(_has_chat_special_tokens(i.text) for i in transfer)}
    return checks
