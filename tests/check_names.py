#!/usr/bin/env python
"""Static check for names that do not resolve — the gap `py_compile` leaves.

`py_compile` validates SYNTAX only. A call to a function that was never defined
compiles happily and raises `NameError` at runtime, on whichever code path first
reaches it. That is how `refusal.label_all_rules` survived: `rule_disagreement`
called it, no commit ever contained a definition, and every compile check passed
— until the labels stage was exercised end-to-end and died 20 minutes into a run.

This walks each module's AST and reports any global name loaded inside a function
that is not a builtin, an import, a module-level binding, or a local/parameter of
the enclosing scopes. It is deliberately conservative: comprehension and nested
scopes are tracked, and anything it cannot resolve confidently it stays quiet
about, so a report here is worth acting on.

    python tests/check_names.py
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = ["core", "experiments", "tools", "tests"]
BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "self", "cls"}


def _bound_by(node: ast.AST) -> set:
    """Names a scope binds: assignments, imports, defs, args, comprehensions."""
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, ast.Global) or isinstance(n, ast.Nonlocal):
            out.update(n.names)
    return out


def _top_level_bindings(tree: ast.Module) -> set:
    """Names bound at module scope, without descending into function bodies."""
    out = set()
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(stmt.name)
        elif isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for a in stmt.names:
                out.add((a.asname or a.name).split(".")[0])
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign,
                               ast.For, ast.AsyncFor, ast.With, ast.AsyncWith,
                               ast.If, ast.Try, ast.While)):
            # Module-level control flow can bind names; collect its stores, but
            # only from statements that are themselves top level.
            for n in ast.walk(stmt):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                    out.add(n.id)
                elif isinstance(n, (ast.Import, ast.ImportFrom)):
                    for a in n.names:
                        out.add((a.asname or a.name).split(".")[0])
    return out


def check_file(path: Path) -> list:
    """Report Load-context names that resolve in no enclosing scope.

    Scope handling has to be exact in BOTH directions or the check is useless:

    * too wide — running `_bound_by` over the whole module treats every
      function's locals as globals, which is how `strat` (a local of
      `stage_extract`) looked defined inside `stage_dimensionality_behavioural`
      and the `NameError` shipped anyway;
    * too narrow — checking a nested function against only its own bindings plus
      module scope flags every closure over an enclosing function's variables,
      which produced 56 false positives in one pass and would have trained us to
      ignore the tool.

    So: walk the function nesting explicitly, carrying the chain of enclosing
    bindings down. A name is legal if any scope in its chain binds it.
    """
    tree = ast.parse(path.read_text(), filename=str(path))
    problems = []

    def visit(node, enclosing: set) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = _bound_by(child) | enclosing
                # Only names in THIS function's body, not nested ones — those are
                # checked against their own chain when we recurse into them.
                for n in ast.walk(child):
                    if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                        if n.id not in scope:
                            problems.append((n.lineno, child.name, n.id))
                visit(child, scope)
            elif isinstance(child, ast.ClassDef):
                visit(child, _bound_by(child) | enclosing)
            else:
                visit(child, enclosing)

    visit(tree, _top_level_bindings(tree) | BUILTINS)
    # De-duplicate: `ast.walk` on nested functions revisits the same Name nodes.
    return sorted(set(problems))


def main() -> int:
    total = 0
    for t in TARGETS:
        for path in sorted((ROOT / t).rglob("*.py")):
            if "__pycache__" in str(path):
                continue
            for lineno, fname, name in check_file(path):
                print(f"[FAIL] {path.relative_to(ROOT)}:{lineno} "
                      f"in {fname}(): undefined name {name!r}")
                total += 1
    print(f"\n{'PASS — every name resolves' if not total else f'{total} unresolved name(s)'}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
