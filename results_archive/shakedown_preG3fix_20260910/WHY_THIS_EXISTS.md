# Shakedown run — pre-G3-fix

One model (`qwen2.5-7b`), one root, produced by the single-model shakedown that
validated the 14 stages before the full chain.

**Kept because this is the artifact set that exposed the G3 bug.** Its
`causal_gate__under.json` reports `GATE 1 — FAIL` with `G3 n = 0` at every bound.
That FAIL was **not a result**: `_band` built the behavioural null with
`groupby(["abs_alpha"])`, which yields TUPLE keys `(0.25,)`, while `beh_nb` looked
up the scalar `0.25`, missed every key, and fell through to `float("inf")`. So
`moved_behaviour = shift > inf` was always False and **G3 could never fire**. The
gate is `G2 AND G3`, so it returned FAIL by construction.

Re-adjudicating this same matrix with the fix — CPU only, the matrix itself was
never wrong — turned FAIL into PASS at all five capability bounds.

Retained so the before/after is inspectable. **Nothing in here may be cited as a
result.**
