"""Mutation tests: prove the verifier's checks actually FAIL when they should.

WHY THIS EXISTS
A check that can never fail is worse than no check, because it reports green.
This project has already produced one: the G3 null-band lookup missed every key,
defaulted to `inf`, and made a gate criterion unfireable — while every artifact
stayed well-formed and every existing check passed.

`verify_rq1_run.py` has ~50 checks and NOTHING established that any of them has
teeth. So this takes a known-good results tree, corrupts one thing at a time,
and asserts the verifier notices. A mutation the verifier survives is a blind
spot, reported by name.

The corpus is a real archived run, copied to a scratch tree per mutation, so the
mutations are applied to artifacts of the shape the verifier actually meets.

Run:  python tests/test_verifier_teeth.py
Exit 0 = every mutation was caught.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

# Default to the LIVE results root. It used to point at
# `results_archive/pre_5model_20260911/results`; that archive was deleted once it
# became uninformative (200/200 corpus against the live 500/500), and the live
# root is the better source anyway — the mutation suite should prove the verifier
# has teeth against the result set we actually report, not one two code
# generations old. Verified: 9/9 mutations caught against `./results`.
SOURCE = Path(os.environ.get("TEETH_SOURCE", ROOT / "results"))
MODEL = os.environ.get("TEETH_MODEL", "qwen2.5-7b")

RESULTS: list[tuple[str, bool, str]] = []


def failing_checks(out: str) -> dict[str, str]:
    """Map failing check name -> its detail line.

    Names alone are not enough. A check that ALREADY fails at baseline can never
    register as "newly failing", so a mutation it detects would be scored BLIND —
    which happened: the minority-class guard already failed on the baseline tree
    (R_control_harmless has 10 minority test items), so gutting R_harm's counts
    produced no NEW name. Comparing the detail too catches "same check, different
    complaint", which is the real signal.
    """
    out_lines = out.splitlines()
    fails: dict[str, str] = {}
    for i, l in enumerate(out_lines):
        if l.startswith("[FAIL]"):
            name = l.split("] ", 1)[1].strip()
            detail = (out_lines[i + 1].strip()
                      if i + 1 < len(out_lines) and out_lines[i + 1].startswith("      ")
                      else "")
            fails[name] = detail
    return fails


def run_verifier(root: Path) -> tuple[int, str]:
    env = dict(os.environ, RESULTS_ROOT=str(root), MODELS=MODEL)
    env.pop("VERIFY_AGAINST", None)
    p = subprocess.run([sys.executable, str(ROOT / "tests" / "verify_rq1_run.py")],
                       capture_output=True, text=True, env=env, cwd=str(ROOT))
    return p.returncode, p.stdout + p.stderr


# --------------------------------------------------------------- mutations
def mut_leak_train_into_test(root: Path) -> str:
    """Make one test instruction also appear in train — the leakage the whole
    train/test discipline exists to prevent."""
    f = root / "e1_0_corpus" / "instructions.jsonl"
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    tst = next(r for r in rows if r["split"] == "test")
    dup = dict(tst); dup["split"] = "train"
    f.write_text("\n".join(json.dumps(r) for r in rows + [dup]) + "\n")
    return "duplicated one test instruction into train"


def mut_drop_a_role(root: Path) -> str:
    """Remove the `tool` role from one model's renderings, so R_role silently
    becomes a 3-class variable on that model only."""
    f = root / "e1_0_corpus" / "rendered_index.csv"
    d = pd.read_csv(f)
    d = d[~((d.model == MODEL) & (d.role == "tool"))]
    d.to_csv(f, index=False)
    return f"dropped the `tool` role from {MODEL}"


def mut_break_length_baseline(root: Path) -> str:
    """Make a direction's AUC worse than its length-only baseline — i.e. the
    direction measures length, not the concept."""
    f = root / "rq1" / MODEL / "direction_validation.csv"
    d = pd.read_csv(f)
    c = d.concept.iloc[0]
    d.loc[d.concept == c, "length_only_auc"] = 0.999
    d.to_csv(f, index=False)
    return f"set length_only_auc=0.999 for {c} (direction can no longer beat it)"


def mut_kill_behavioural_band(root: Path) -> str:
    """THE G3 BUG, reproduced exactly: make the behavioural null band
    non-finite so G3 can never fire."""
    hits = list((root / "rq1" / MODEL).glob("causal_gate*.json"))
    if not hits:
        return ""
    g = json.loads(hits[0].read_text())
    for v in g["per_bound"].values():
        if isinstance(v.get("null_band_behaviour"), dict):
            for k in v["null_band_behaviour"]:
                v["null_band_behaviour"][k] = float("inf")
    hits[0].write_text(json.dumps(g, indent=2).replace("Infinity", "1e999"))
    return f"set every behavioural null band to inf in {hits[0].name}"


def mut_tiny_test_set(root: Path) -> str:
    """Report intervals computed from a handful of instructions."""
    f = root / "rq1" / MODEL / "direction_validation.csv"
    d = pd.read_csv(f)
    if "n_test_instructions" not in d.columns:
        return ""
    d.loc[d.index[:5], "n_test_instructions"] = 3
    d.to_csv(f, index=False)
    return "set n_test_instructions=3 on 5 reported fits"


def mut_unbalanced_e11d(root: Path) -> str:
    """Mark the E1.1d harm variants as NOT role-balanced — the bug that
    overstated separability by letting the variants carry a role component."""
    f = root / "rq1" / MODEL / "harm_controls.csv"
    if not f.exists():
        return ""
    d = pd.read_csv(f)
    if "role_balanced" not in d.columns:
        return ""
    d["role_balanced"] = False
    d.to_csv(f, index=False)
    return "set role_balanced=False on every E1.1d variant"


def mut_leak_undetermined_into_fit(root: Path) -> str:
    """Make R_control's recorded fit counts include the undetermined pool — the
    leak that would make the soft-refusal projection test circular, since the
    'held-out' items would then be part of what the direction was fitted on."""
    f = root / "rq1" / MODEL / "direction_validation.csv"
    d = pd.read_csv(f)
    m = d.concept == "R_control"
    if not m.any() or "n_negative" not in d.columns:
        return ""
    d.loc[m, "n_negative"] = d.loc[m, "n_negative"] + 273   # the train undetermined
    d.to_csv(f, index=False)
    return "added the undetermined items to R_control's negative class"


def mut_thin_minority_class(root: Path) -> str:
    """Leave the UNION test-instruction count healthy while gutting the minority
    class. This is the shape `R_control` always has — refusal is rare among
    harmless prompts, compliance is rare among harmful ones — and the union
    check passes it every time."""
    f = root / "rq1" / MODEL / "direction_validation.csv"
    d = pd.read_csv(f)
    if "n_test_pos" not in d.columns:
        return ""
    # Applied to R_harm, NOT R_control: gutting R_control's counts also trips the
    # held-out-pool check, so the mutation would be "caught" by an unrelated
    # check and tell us nothing about the minority guard.
    d.loc[d.concept == "R_harm", "n_test_pos"] = 2   # union stays large via n_test_neg
    d.to_csv(f, index=False)
    return "set R_harm n_test_pos=2 while leaving n_test_instructions untouched"


MUTATIONS = [
    ("train/test leakage", mut_leak_train_into_test),
    ("a role class silently dropped", mut_drop_a_role),
    ("direction loses to its length baseline", mut_break_length_baseline),
    ("G3 null band = inf (the real bug)", mut_kill_behavioural_band),
    ("CI from 3 instructions", mut_tiny_test_set),
    ("E1.1d not role-balanced", mut_unbalanced_e11d),
    ("undetermined leaked into the R_control fit", mut_leak_undetermined_into_fit),
    ("minority class gutted, union intact", mut_thin_minority_class),
]


def source_mutation_layer_selection() -> tuple[bool, str]:
    """Selection-on-test is a SOURCE property, so it needs a source mutation.

    `direction_validation.csv` never records which layer was reported — both the
    train-argmax and the test-argmax are re-derived from the same table at read
    time, so the artifact cannot disagree with itself and no artifact mutation
    can express this bug. That is why the artifact-level check was vacuous, and
    why the real guard lives in tests/test_invariants.py. This confirms THAT
    guard fires when the source is corrupted.
    """
    import importlib.util
    import re
    import tempfile as _tf

    src = (ROOT / "experiments" / "rq1.py").read_text()
    broken = re.sub(r'train_auc(["\']?\s*\]?\s*\.idxmax)', r'auc\1', src, count=1)
    if broken == src:
        return False, "could not construct the mutation (no train_auc.idxmax found)"

    with _tf.TemporaryDirectory() as td:
        corrupted = Path(td) / "rq1_mutated.py"
        corrupted.write_text(broken)
        spec = importlib.util.spec_from_file_location(
            "ti_teeth", ROOT / "tests" / "test_invariants.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["ti_teeth"] = mod
        spec.loader.exec_module(mod)
        try:
            mod.check_layer_selection_is_on_train(corrupted)
            return False, "invariant did NOT fire on a source mutation"
        except AssertionError as e:
            return True, f"invariant fired: {str(e)[:110]}"


def main() -> int:
    if not (SOURCE / "rq1" / MODEL).exists():
        print(f"SKIP - no source results at {SOURCE}/rq1/{MODEL}\n"
              f"       set TEETH_SOURCE / TEETH_MODEL, or run after a completed run.")
        return 0

    print(f"source tree: {SOURCE}\nmodel: {MODEL}\n")
    base_rc, base_out = None, ""
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "results"
        shutil.copytree(SOURCE, base)
        base_rc, base_out = run_verifier(base)
    print(f"baseline verifier exit code on the UNMUTATED tree: {base_rc}")
    print("  (a non-zero baseline is fine — the archived run is incomplete; what\n"
          "   matters is that each mutation adds a NEW failing check)\n")
    base_fails = failing_checks(base_out)

    blind: list[str] = []
    for name, fn in MUTATIONS:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "results"
            shutil.copytree(SOURCE, root)
            what = fn(root)
            if not what:
                print(f"[skip] {name}: artifact not present in the source tree")
                continue
            rc, out = run_verifier(root)
            fails = failing_checks(out)
            # newly failing, OR failing with a different complaint than baseline
            new = {n for n, d in fails.items()
                   if n not in base_fails or base_fails[n] != d}
            caught = bool(new)
            RESULTS.append((name, caught, what))
            if not caught:
                blind.append(name)
            print(f"[{'ok  ' if caught else 'BLIND'}] {name}")
            print(f"         mutation: {what}")
            if caught:
                for n in sorted(new)[:3]:
                    print(f"         caught by: {n}")
            else:
                print(f"         NO new check failed — the verifier cannot see this")

    # ---- the source-level mutation, which no artifact mutation can express
    ok, detail = source_mutation_layer_selection()
    RESULTS.append(("layer selected on test (source mutation)", ok, detail))
    if not ok:
        blind.append("layer selected on test (source mutation)")
    print(f"[{'ok  ' if ok else 'BLIND'}] layer selected on test (source mutation)")
    print(f"         {detail}")

    print()
    if blind:
        print(f"BLIND SPOTS: {len(blind)} mutation(s) the verifier does not catch:")
        for b in blind:
            print(f"  - {b}")
        return 1
    print(f"PASS - all {len(RESULTS)} mutations were caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
