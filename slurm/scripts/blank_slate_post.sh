#!/bin/bash
# Post-run pass for the blank-slate rerun: verification first, then the
# derived analyses and figures.
#
# Order matters. Verification runs BEFORE the tools, so that if the run is
# invalid we find out from an exit code rather than from a plausible-looking
# figure. Every tool here reads `results/` and writes either stdout (captured
# into the Slurm log) or a file under `results/`.
#
#   sbatch slurm/scripts/run_cpu.sh slurm/scripts/blank_slate_post.sh
# or, since it is plain bash rather than python, submit it directly:
#   sbatch --export=ALL slurm/scripts/blank_slate_post.sh
#SBATCH --job-name=cs_post
#SBATCH --output=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/logs/adhoc/out/%j-%x.out
#SBATCH --error=/home/b6aj/jtelintelo.b6aj/SecurityControl/slurm/logs/adhoc/err/%j-%x.err
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
set -uo pipefail   # NOT -e: a failing check must not hide the checks after it

module purge
module load craype-network-ofi PrgEnv-nvidia cuda/12.6 craype-arm-grace craype-accel-nvidia90
export HF_HOME="/scratch/b6aj/jtelintelo.b6aj/hf-cache-dir"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate venv_causal_safety
cd /home/b6aj/jtelintelo.b6aj/SecurityControl

FAILED=0
step() {
  echo; echo "################################################################"
  echo "# $1"; echo "################################################################"
  shift
  "$@"
  local rc=$?
  [ $rc -ne 0 ] && { echo ">>> EXIT $rc"; FAILED=$((FAILED+1)); }
  return 0
}

# The roster comes from `core/config.py`, never from a list repeated here: a
# post pass that verifies 3 of 5 models still exits 0 and still draws figures.
MODELS_CSV=$(python -c 'from core.config import RQ1_MODELS; print(",".join(RQ1_MODELS))')

echo "=== $(date -Is) blank-slate post pass ==="
python -c 'from core.config import roster_table; print(roster_table())'
echo "verifying models: $MODELS_CSV"

# ---------------------------------------------------------------- verification
step "verify results/ (standalone)" \
  env RESULTS_ROOT=./results MODELS="$MODELS_CSV" python tests/verify_rq1_run.py
# The reproduction root gets the SAME standalone scrutiny as the primary one.
# Verifying it only through the cross-root comparison would check that the two
# agree while never checking that either is internally sound — two roots can
# agree perfectly on the same mistake.
step "verify results_verify/ (standalone)" \
  env RESULTS_ROOT=./results_verify MODELS="$MODELS_CSV" python tests/verify_rq1_run.py
step "verify results/ against the independent reproduction" \
  env RESULTS_ROOT=./results VERIFY_AGAINST=./results_verify MODELS="$MODELS_CSV" python tests/verify_rq1_run.py
step "reproduction diagnostic (per-concept AUC drift)" \
  python tools/repro_diag.py

# ---------------------------------------------------------------- completeness
step "walkthrough: every experiment, artifact, verification" \
  python tools/rq1_walkthrough.py
# The walkthrough says whether an artifact APPEARED. This says what it SAYS —
# every RQ1 pillar (geometry, dimensionality, emergence, causal) plus C1/C2/C7/C9,
# per model, plus the cross-model table. Without it the run is read through
# whichever arm happens to be the binding constraint.
step "full-scope RQ1 readout (all pillars, all models, all arms)" \
  python tools/rq1_status.py --csv
step "full-scope RQ1 readout — reproduction root" \
  env RESULTS_ROOT=./results_verify python tools/rq1_status.py --csv
step "diff against the pre-rerun archive (what moved, and by how much)" \
  python tools/diff_against_archive.py

# ---------------------------------------------------------------- derived
step "E1.6 gate comparison across control variants" python tools/gate_compare.py
step "E1.6 asymmetry structure" python tools/asymmetry_structure.py
step "E1.1d harm/control cross-check (layer-matched)" python tools/harm_control_crosscheck.py
step "E1.7 style decomposition" python tools/style_decompose.py
step "Guard filter impact on the 2x2" python tools/guard_filter_impact.py
step "O-1 label audit sheet (blind)" python tools/build_label_audit.py
# O-1, as far as it can be closed without a person. The judge is off-roster
# (different vendor from all five models under audit) and runs greedy, so it is
# re-runnable to the same answer. It is a SECOND AUTOMATED INSTRUMENT, not a
# human validation, and `score_label_audit.py` prints that caveat itself.
step "O-1 adjudicate the sheet with an off-roster LLM judge" \
  python tools/adjudicate_labels.py
step "O-1 score the refusal instrument against the adjudication" \
  python tools/score_label_audit.py

# O-1b — characterise the `undetermined` pool. The degeneracy gate fires on NONE
# of it; every row is `guard_says_response_safe`, i.e. the prefix rule said
# `complied` while Guard found no harmful content in the full generation. The
# interpretation downstream ("these are soft refusals") must be measured, not
# asserted from four hand-read examples. Writes a sensitivity table; does NOT
# modify refusal_labels.csv, because the rule is pre-registered.
step "O-1b adjudicate the undetermined pool (soft refusals?)" \
  python tools/adjudicate_undetermined.py

# Is the prefix rule the right detector? Refit R_control under the judge's labels
# and compare AUC, split-half stability, length baseline and cos with R_harm.
# Decides empirically what neither of us should decide by argument: a judge that
# wins on AUC while losing split-half stability is fitting noise. Depends on the
# adjudication above. Does NOT modify refusal_labels.csv.
step "sensitivity: refit R_control under the LLM-judge labels" \
  python tools/refit_control_labels.py

# E + D — what R_control actually encodes, and whether it is one axis.
# The soft-refusal pool is held out of the fit by construction, contains no
# refusal marker, and is read at t_post_inst (prompt only, so truncation cannot
# reach it). Where it projects separates "control variable" from "marker
# detector". CPU, from cached activations; refits nothing.
step "E+D soft-refusal projection and axis test" \
  python tools/soft_refusal_probe.py

# ---------------------------------------------------------------- figures
# --- analyses that were previously run by hand, and so were not part of a
# --- from-scratch run at all. An RQ1 result that only exists because someone
# --- remembered to invoke a tool is not reproducible.
step "independent recomputation audit (primary)" \
  env RESULTS_ROOT=./results python tools/audit_independent.py
step "independent recomputation audit (reproduction)" \
  env RESULTS_ROOT=./results_verify python tools/audit_independent.py
step "R_control learning curve — under arm (sets the reporting floor)" \
  python tools/control_learning_curve.py --concept R_control
step "R_control learning curve — over arm" \
  python tools/control_learning_curve.py --concept R_control_harmless
step "E+D soft-refusal projection, SPLIT by adjudicated label" \
  python tools/soft_refusal_split.py
step "adjudicator agreement (rules x judges x any human/Claude sheet)" \
  python tools/score_adjudicators.py

step "figures" python tools/make_figures.py

# ---------------------------------------------------------------- last, and slow
# ~3.1s per adjudication on a ~23k-row matrix x 432 settings x several matrices
# = ~90 min. Deliberately final: everything above is cheap and must not be
# starved by it, and if this alone times out it re-runs standalone on the saved
# matrices without repeating any GPU work.
step "E1.6 gate sensitivity sweep (slow: ~90 min)" python tools/gate_sensitivity.py

echo
echo "================================================================"
echo "post pass complete: $FAILED step(s) exited non-zero"
echo "=== $(date -Is) ==="
exit $FAILED
