#!/usr/bin/env python
"""Static check for the two bug classes that keep reaching production here.

WHY THESE TWO
Three bugs of the same shape shipped in a single session, each failing only at
runtime and none visible to `check_names.py`, which resolves bare names and
therefore cannot see either pattern:

  1. **DataFrame attribute shadowing.** `g.tail` is `DataFrame.tail`, the method,
     not a column named "tail" — so `g.tail.unique()` raised. `df.style` is the
     Styler accessor, so `df[df.style == s]` compared a Styler to a string, got
     False, and `df[False]` raised KeyError. Both look exactly like ordinary
     column access.
  2. **Rebinding a name that is still needed.** `d` held the model's results
     directory; a fit inside the loop rebound it to a `Direction`, and the save
     at the bottom failed with "unsupported operand type(s) for /: 'Direction'
     and 'str'". Four hundred lines apart, no warning.

A third pattern is included because it also reached a job: an attribute on an
imported module that does not exist (`model_meta.decoder_layers`), which no
name-resolution pass catches because the NAME `model_meta` resolves fine.

Run:  python tests/check_shadowing.py
Exit 0 = clean.
"""
from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# DataFrame METHODS. Legitimate use is always a CALL — `g.mean()`. Reaching one
# WITHOUT calling it (`g.tail.unique()`, `df.head + 1`) means the author expected
# a column and got the bound method instead. That is the exact `g.tail` bug.
DF_METHODS = {
    "tail", "head", "count", "mean", "min", "max", "sum", "std", "var",
    "median", "mode", "apply", "map", "sample", "abs", "all", "any", "clip",
    "copy", "round", "take", "where", "mask", "rank", "shift", "diff", "pop",
    "filter", "update", "items", "keys", "first", "last", "product", "add",
    "sub", "mul", "div", "pow", "isin", "sort_values", "groupby",
}
# Accessors that are legitimately read without calling (`df.columns`), so they
# cannot be flagged on that basis. `style` is the exception: it is the Styler,
# this project never styles anything, and `df.style == x` silently compares a
# Styler to a value — the `df.style` bug.
ALWAYS_SUSPECT = {"style"}

FINDINGS: list[str] = []


def _is_called(node: ast.Attribute, parents: dict) -> bool:
    par = parents.get(id(node))
    return isinstance(par, ast.Call) and par.func is node


def check_pandas_shadowing(path: Path, tree: ast.AST) -> None:
    parents = {}
    for n in ast.walk(tree):
        for ch in ast.iter_child_nodes(n):
            parents[id(ch)] = n
    for n in ast.walk(tree):
        if not isinstance(n, ast.Attribute):
            continue
        if n.attr in ALWAYS_SUSPECT:
            FINDINGS.append(
                f"{path.relative_to(ROOT)}:{n.lineno}: `.{n.attr}` is the pandas "
                f"Styler accessor, never a column — use [\"{n.attr}\"]")
        elif n.attr in DF_METHODS and not _is_called(n, parents):
            FINDINGS.append(
                f"{path.relative_to(ROOT)}:{n.lineno}: `.{n.attr}` is a DataFrame "
                f"method accessed without calling it — if a column was meant, use "
                f"[\"{n.attr}\"]")


def check_module_attrs(path: Path, tree: ast.AST) -> None:
    """Flag `mod.func` where `mod` is a first-party import and `func` is absent."""
    imported: dict[str, str] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("core"):
            for a in n.names:
                imported[a.asname or a.name] = f"{n.module}.{a.name}"
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("core"):
                    imported[a.asname or a.name] = a.name
    for alias, target in imported.items():
        try:
            mod = importlib.import_module(target)
        except Exception:
            continue
        if not hasattr(mod, "__file__"):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) \
                    and n.value.id == alias and not hasattr(mod, n.attr) \
                    and not n.attr.startswith("_"):
                FINDINGS.append(
                    f"{path.relative_to(ROOT)}:{n.lineno}: `{alias}.{n.attr}` does "
                    f"not exist in {target} — name resolution cannot see this")


# A THIRD RULE WAS ATTEMPTED AND DROPPED — recorded so it is not rebuilt.
#
# The `d` bug (a `Path` rebound to a `Direction` inside a loop, then used as a
# path after it) looks like it should be statically detectable. It is not, by
# this approach: the distinguishing feature is a TYPE change, and every proxy for
# it either over- or under-fires.
#
#   v1  flag any pre-loop name rebound in a loop and read after   -> 7 hits, 0 real
#       (accumulators `x = x + 1`, sentinels `bal_info = None`)
#   v2  exclude accumulators and sentinels                        -> 0 hits, and it
#       MISSED the reconstructed bug, because the original binding sat inside an
#       outer loop rather than at function top level
#
# Widening the scan to every nested block brings the accumulator noise straight
# back. A rule that misses the bug it was written for while looking authoritative
# is worse than no rule, so it is not shipped. The two rules above are kept
# because both are exact: they fire on the real `g.tail` and `df.style` bugs and
# on neither `g.mean()` nor `df.columns`, verified against reconstructions.
#
# Catching the rebinding class needs types, not syntax — mypy or pyright on the
# annotated parts of `core/` would be the honest route if it recurs.


def main() -> int:
    files = [p for p in ROOT.rglob("*.py")
             if "__pycache__" not in str(p) and "results" not in p.parts]
    for p in sorted(files):
        try:
            tree = ast.parse(p.read_text())
        except SyntaxError as e:
            FINDINGS.append(f"{p.relative_to(ROOT)}: SyntaxError {e}")
            continue
        check_pandas_shadowing(p, tree)
        check_module_attrs(p, tree)

    print(f"scanned {len(files)} files")
    if FINDINGS:
        print(f"\n{len(FINDINGS)} finding(s):")
        for f in FINDINGS:
            print(f"  {f}")
        return 1
    print("PASS — no pandas-attribute shadowing and no missing module attrs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
