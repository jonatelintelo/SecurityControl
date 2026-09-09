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
