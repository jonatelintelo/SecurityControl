"""Prompt pools, from real datasets.

Replaces the curated in-repo lists of the previous attempt, which were far too
small to measure anything (15 attack prompts, 10 context intents).

Two kinds of corpus, and the distinction matters:

* :func:`crossed_corpus` — **our design**. One instruction set crossed with the
  four role tags, so `R_harm`, `R_role` and `R_control` are all estimated on the
  same substrate. Following Zhao and the role paper literally would estimate the
  harm directions on instruction data and the role probe on webtext, leaving any
  cosine between them confounded by that distribution gap.

* :func:`zhao_corpus`, :func:`role_paper_corpus` — the source recipes on their own
  data, run as validation and for cross-corpus transfer only.

Schemas below were verified against the Hub, not assumed:

    walledai/AdvBench            520    prompt, target
    tatsu-lab/alpaca           52002    instruction, input, output
    natolambert/xstest-v2-copy   450    type, prompt        (split "prompts")
    walledai/StrongREJECT        313    prompt, category, source
    allenai/c4                     -    text                (config "en", streaming)

`walledai/XSTest` is **gated**; the mirror above is not. XSTest's safe subset is
the rows whose `type` does *not* begin with `contrast_`.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from core.positions import ROLES

# Optional dataset sources that failed to load, `name -> error`. Recorded rather
# than raised (the pools still work without them) but written into the run
# manifest, so a missing source is a documented deviation and not a silent one.
SOURCE_FAILURES: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Instruction:
    text: str
    harmful: bool
    source: str

    @property
    def uid(self) -> str:
        """Stable id of the *base* instruction, so the four role renderings of one
        instruction can be kept on the same side of a train/test split."""
        return hashlib.sha1(_normalize(self.text).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class CrossedItem:
    """One instruction rendered under one role tag."""
    instruction: str
    harmful: bool
    role: str
    source: str
    uid: str


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _dedup(items: Sequence[Instruction]) -> List[Instruction]:
    seen, out = set(), []
    for it in items:
        key = _normalize(it.text)
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _take(items: Sequence[Instruction], n: Optional[int], seed: int) -> List[Instruction]:
    """Deterministic, **source-balanced** subsample.

    Pooling every source and shuffling lets the largest dataset swamp the rest:
    Sorry-Bench (9,240) drowns AdvBench (520), and Alpaca (52,002) drowns XSTest's
    ~250 safe rows. That second case is actively harmful — XSTest is the
    benign-but-*sensitive* negative, the one that stops `R_harm` collapsing into a
    "sensitive topic" detector, and it is precisely what proportional sampling
    throws away.

    So we round-robin across sources, shuffled within each, until the quota is
    filled or a source is exhausted.
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
    cursors = {s: 0 for s in order}
    while len(out) < n and any(cursors[s] < len(by_source[s]) for s in order):
        for s in order:
            if len(out) >= n:
                break
            if cursors[s] < len(by_source[s]):
                out.append(by_source[s][cursors[s]])
                cursors[s] += 1
    return out


def _load(path: str, *args, **kwargs):
    from datasets import load_dataset
    return load_dataset(path, *args, **kwargs)


def _text_column(ds) -> str:
    """Pick the instruction-bearing column for datasets whose schema we have not
    pinned. Case-insensitive: JBB-Behaviors capitalises its columns
    (`Index, Goal, Target, Behavior, ...`), and matching case-sensitively silently
    dropped that whole source. Raises rather than guessing wrong."""
    lookup = {c.lower(): c for c in ds.column_names}
    for cand in ("prompt", "goal", "instruction", "behavior", "question", "turns", "text"):
        if cand in lookup:
            return lookup[cand]
    raise ValueError(f"no recognisable text column in {ds.column_names}")


# ---------------------------------------------------------------------------
# harmful / harmless instruction pools
# ---------------------------------------------------------------------------
def harmful_instructions(
    n: Optional[int] = None,
    seed: int = 0,
    include_extra_sources: bool = True,
) -> List[Instruction]:
    """Harmful instructions. AdvBench is the pinned source; JBB and Sorry-Bench are
    added when loadable, since Zhao draw on all three."""
    out: List[Instruction] = []

    ds = _load("walledai/AdvBench")["train"]
    out += [Instruction(r["prompt"], True, "advbench") for r in ds]

    if include_extra_sources:
        for name, args in (("JailbreakBench/JBB-Behaviors", ("behaviors",)),
                           ("sorry-bench/sorry-bench-202503", ("default",))):
            try:
                d = _load(name, *args)
                split = d[list(d.keys())[0]]
                col = _text_column(split)
                tag = name.split("/")[0].lower()
                for r in split:
                    v = r[col]
                    if isinstance(v, list):        # sorry-bench stores turns as a list
                        v = v[0] if v else ""
                    if isinstance(v, str) and v.strip():
                        out.append(Instruction(v, True, tag))
            except Exception as e:
                # Optional sources, but a failure must be visible: absence from the
                # composition table is easy to miss, and quietly dropping one of
                # Zhao's three harmful sources is a deviation we must be able to
                # report. See `SOURCE_FAILURES`.
                SOURCE_FAILURES[name] = f"{type(e).__name__}: {str(e)[:120]}"

    return _take(_dedup(out), n, seed)


def harmless_instructions(n: Optional[int] = None, seed: int = 0) -> List[Instruction]:
    """Harmless instructions: Alpaca (no-input rows only, so each is standalone)
    plus XSTest's safe subset, which is benign-but-sensitive and therefore the
    harder negative — Zhao use XSTest for the same reason."""
    out: List[Instruction] = []

    alp = _load("tatsu-lab/alpaca")["train"]
    for r in alp:
        if not r["input"].strip():
            out.append(Instruction(r["instruction"], False, "alpaca"))

    try:
        xs = _load("natolambert/xstest-v2-copy")["prompts"]
        for r in xs:
            if not str(r["type"]).startswith("contrast_"):
                out.append(Instruction(r["prompt"], False, "xstest_safe"))
    except Exception:
        pass

    return _take(_dedup(out), n, seed)


def attack_intents(
    n: Optional[int] = None,
    seed: int = 0,
    exclude: Optional[Sequence[Instruction]] = None,
    jaccard_threshold: float = 0.8,
) -> List[Instruction]:
    """Held-out attack intents (StrongREJECT), disjoint from the fitting pool.

    The plan requires attacks be evaluated on *completely disjoint* intents (CPE).
    Being a different dataset is not sufficient: StrongREJECT aggregates from
    other benchmarks and does overlap AdvBench — measured at 3 exact and 4
    near-duplicate collisions against a 400-instruction fitting pool. Pass the
    fitting instructions as `exclude` and disjointness is enforced by
    construction rather than merely checked afterwards.
    """
    ds = _load("walledai/StrongREJECT")["train"]
    items = _dedup([Instruction(r["prompt"], True, "strongreject") for r in ds])

    if exclude:
        excl_norm = {_normalize(i.text) for i in exclude}
        excl_tokens = [_token_set(i.text) for i in exclude]

        def collides(text: str) -> bool:
            if _normalize(text) in excl_norm:
                return True
            t = _token_set(text)
            if not t:
                return False
            return any(
                len(t & ft) / len(t | ft) >= jaccard_threshold
                for ft in excl_tokens if (t | ft)
            )

        items = [i for i in items if not collides(i.text)]

    return _take(items, n, seed)


def webtext_documents(n: int = 150, seed: int = 0, max_chars: int = 2000) -> List[str]:
    """Non-instruct webtext for the role paper's constant-content design.

    C4 requires the config name `en` (passing "en" as a *split* is the error that
    made this look unavailable). Streamed so nothing large is downloaded.
    """
    ds = _load("allenai/c4", "en", streaming=True)["validation"]
    out: List[str] = []
    for row in ds:
        t = (row.get("text") or "").strip()
        if len(t) > 200:
            out.append(t[:max_chars])
        if len(out) >= n:
            break
    return out


# ---------------------------------------------------------------------------
# our crossed corpus
# ---------------------------------------------------------------------------
def crossed_corpus(
    n_harmful: int = 200,
    n_harmless: int = 200,
    roles: Sequence[str] = ROLES,
    seed: int = 0,
) -> List[CrossedItem]:
    """`instructions x {harmful, harmless} x roles`, fully crossed.

    Every instruction appears under *every* role, which is what lets `R_harm` be
    estimated with role balanced across both sides of the contrast and `R_role`
    with harm balanced — each direction a main effect rather than a mixture.
    """
    bad = harmful_instructions(n_harmful, seed=seed)
    good = harmless_instructions(n_harmless, seed=seed)
    if len(bad) < n_harmful or len(good) < n_harmless:
        raise ValueError(
            f"insufficient instructions: harmful {len(bad)}/{n_harmful}, "
            f"harmless {len(good)}/{n_harmless}"
        )

    items: List[CrossedItem] = []
    for inst in [*bad, *good]:
        for role in roles:
            items.append(CrossedItem(inst.text, inst.harmful, role, inst.source, inst.uid))
    return items


def split_by_instruction(
    items: Sequence[CrossedItem],
    train_fraction: float = 0.75,
    seed: int = 0,
) -> Tuple[List[CrossedItem], List[CrossedItem]]:
    """Split on the base instruction, never on the rendered item.

    Splitting per item would put the same instruction's `user` rendering in train
    and its `tool` rendering in test, so a probe could score well by memorising
    content. The split is stratified by harm label so both sides stay balanced.
    """
    rng = random.Random(seed)
    by_label: Dict[bool, List[str]] = {True: [], False: []}
    seen = set()
    for it in items:
        if it.uid not in seen:
            seen.add(it.uid)
            by_label[it.harmful].append(it.uid)

    train_uids = set()
    for label, uids in by_label.items():
        uids = sorted(uids)
        rng.shuffle(uids)
        train_uids.update(uids[: int(round(len(uids) * train_fraction))])

    train = [it for it in items if it.uid in train_uids]
    test = [it for it in items if it.uid not in train_uids]
    return train, test


# ---------------------------------------------------------------------------
# paper-faithful corpora (validation only)
# ---------------------------------------------------------------------------
def zhao_corpus(n_per_side: int = 100, seed: int = 0) -> Tuple[List[Instruction], List[Instruction]]:
    """Zhao's own contrast, on their own data, at their own scale (100/side)."""
    return harmful_instructions(n_per_side, seed=seed), harmless_instructions(n_per_side, seed=seed)


def role_paper_corpus(
    n_documents: int = 150,
    roles: Sequence[str] = ROLES,
    seed: int = 0,
) -> List[CrossedItem]:
    """The role paper's constant-content design: identical webtext under each role
    tag. Used to reproduce their probe and, more importantly, to test whether our
    crossed-corpus probe transfers here — if it does not, ours is reading
    "instruction-ness" rather than role."""
    docs = webtext_documents(n_documents, seed=seed)
    items: List[CrossedItem] = []
    for doc in docs:
        uid = hashlib.sha1(_normalize(doc).encode()).hexdigest()[:12]
        for role in roles:
            items.append(CrossedItem(doc, False, role, "c4", uid))
    return items


# ---------------------------------------------------------------------------
# disjointness + provenance
# ---------------------------------------------------------------------------
def _token_set(s: str) -> set:
    return set(_normalize(s).split())


def disjointness_report(
    fitting: Sequence[Instruction],
    evaluation: Sequence[Instruction],
    jaccard_threshold: float = 0.8,
) -> Dict[str, object]:
    """Check that attack intents are disjoint from direction-fitting prompts.

    Exact-match alone is too weak: AdvBench and StrongREJECT can phrase the same
    intent differently. Near-duplicates are flagged by token Jaccard so the check
    catches paraphrase, not just identity.
    """
    fit_norm = {_normalize(i.text) for i in fitting}
    exact = [e.text for e in evaluation if _normalize(e.text) in fit_norm]

    fit_tokens = [_token_set(i.text) for i in fitting]
    near = []
    for e in evaluation:
        et = _token_set(e.text)
        if not et:
            continue
        for ft in fit_tokens:
            union = et | ft
            if union and len(et & ft) / len(union) >= jaccard_threshold:
                near.append(e.text)
                break

    return {
        "n_fitting": len(fitting),
        "n_evaluation": len(evaluation),
        "exact_overlap": len(exact),
        "near_duplicate_overlap": len(near),
        "jaccard_threshold": jaccard_threshold,
        "disjoint": len(exact) == 0 and len(near) == 0,
        "examples": (exact + near)[:5],
    }


def describe_pool(items: Iterable[Instruction]) -> Dict[str, int]:
    """Source composition, for the run manifest — makes a silently missing
    optional dataset visible instead of invisible."""
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.source] = counts.get(it.source, 0) + 1
    return dict(sorted(counts.items()))
