#!/bin/bash
# Blank-slate rerun of every implemented RQ1 experiment, on the full roster,
# into two independent results roots.
#
# WHY A DRIVER AND NOT sbatch BY HAND
# The RQ1 evidence accumulated over ~30 jobs interleaved with corrections, so no
# single job ever produced the whole result set under one code state. This
# submits the complete DAG with Slurm dependencies, so every artifact in the run
# comes from the same commit.
#
# WHY THREE MODELS
# The scoped plan's first pass is "one representative dense model and one
# representative MoE model", and its success criterion is "if both results hold
# on one dense and one MoE architecture". Two dense Qwen models plus the MoE.
#
# WHY TWO ROOTS
# `results` is primary; `results_verify` is an independent repetition at
# IDENTICAL configuration. Identical is deliberate: it is what makes the
# verifier's exactness assertion on the label-independent concepts meaningful.
# Batch-size sensitivity is a different question and must not be confounded in.
#
# WHY ONE MODEL PER JOB
# `sbatch --export=ALL,MODELS=a,b` does NOT work: sbatch splits --export on
# commas, so only the first slug survives and the rest are misparsed as further
# assignments. Every job below carries exactly one slug.
#
#   bash slurm/scripts/blank_slate.sh          # submit
#   bash slurm/scripts/blank_slate.sh --dry    # print the plan only
set -euo pipefail
cd /home/b6aj/jtelintelo.b6aj/SecurityControl

DRY=""
[ "${1:-}" = "--dry" ] && DRY="echo [dry] "

GPU=slurm/scripts/run_gpu.sh
CPU=slurm/scripts/run_cpu.sh
Q25=qwen2.5-7b
Q35=qwen3.5-9b
MOE=qwen3.5-35b-a3b

sub() {  # sub <dependency-or-empty> <exports> <script> <args...>
  local dep="$1"; shift
  local exp="$1"; shift
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterok:${dep}"
  if [ -n "$DRY" ]; then
    echo "[dry] sbatch --parsable $depflag --export=ALL,$exp $*" >&2
    echo "DRYID"
  else
    sbatch --parsable $depflag --export="ALL,$exp" "$@"
  fi
}

echo "=== blank-slate RQ1 rerun, submitted $(date -Is) ==="

# ------------------------------------------------------------ preflight
# Cheap checks first: each guards a failure mode that is SILENT downstream, and
# each costs about two minutes against a multi-hour run.
# Static first: it needs no GPU and catches the class `py_compile` cannot see —
# a name that does not resolve, which raises only on the path that reaches it.
J_NAM=$(sub "" "PREFLIGHT=1" $CPU tests/check_names.py)
echo "static: every name resolves             : $J_NAM"
J_INV=$(sub "" "PREFLIGHT=1" $CPU tests/test_invariants.py)
echo "invariants (hooks, padding, dtypes)     : $J_INV"
J_RP=$(sub "" "PREFLIGHT=1" $GPU tests/smoke_read_positions.py)
echo "readout: per-item positions + all layers: $J_RP"
J_SP=$(sub "" "PREFLIGHT=1" $GPU tests/smoke_steer_positions.py)
echo "steering: position masks reach the hook : $J_SP"
PRE="${J_NAM}:${J_INV}:${J_RP}:${J_SP}"

# ------------------------------------------------------------ corpora
J_CORP=$(sub "$PRE" "RESULTS_ROOT=./results" $CPU experiments/e1_0_corpus.py)
echo "E1.0 corpus                -> results  : $J_CORP"
# E1.7 Level 2 needs generation, so it is a GPU job and its own frozen artifact.
J_STY=$(sub "$J_CORP" "RESULTS_ROOT=./results,N_STYLE_BASE=168,STYLE_MAX_NEW_TOKENS=200" \
        $GPU experiments/e1_7_style_corpus.py)
echo "E1.7 Level 2 style corpus  -> results  : $J_STY"

# ------------------------------------------------------------ primary root
# Qwen2.5 carries the under-refusal variant: it is the only model whose
# harmful-and-complied cell has been shown populated enough to estimate it.
J_Q25=$(sub "$J_STY" "RESULTS_ROOT=./results,MODELS=$Q25,CONTROL_VARIANT=under" $GPU experiments/rq1.py)
echo "RQ1 full  $Q25         -> results  : $J_Q25  (under)"
J_Q35=$(sub "$J_STY" "RESULTS_ROOT=./results,MODELS=$Q35,CONTROL_VARIANT=over" $GPU experiments/rq1.py)
echo "RQ1 full  $Q35         -> results  : $J_Q35  (over)"
J_MOE=$(sub "$J_STY" "RESULTS_ROOT=./results,MODELS=$MOE,CONTROL_VARIANT=over" $GPU experiments/rq1.py)
echo "RQ1 full  $MOE   -> results  : $J_MOE  (over, MoE arm)"

# The matched cross-model gate: `over` on BOTH dense models, so the comparison is
# not harm->under-refusal on one against harm->over-refusal on the other.
J_Q25O=$(sub "$J_Q25" "RESULTS_ROOT=./results,MODELS=$Q25,CONTROL_VARIANT=over" \
         $GPU experiments/rq1.py --only causal)
echo "E1.6 gate $Q25         -> results  : $J_Q25O (over, matched)"

# ------------------------------------------------------------ reproduction root
J_CORPV=$(sub "$PRE" "RESULTS_ROOT=./results_verify" $CPU experiments/e1_0_corpus.py)
echo "E1.0 corpus                -> verify   : $J_CORPV"
J_STYV=$(sub "$J_CORPV" "RESULTS_ROOT=./results_verify,N_STYLE_BASE=168,STYLE_MAX_NEW_TOKENS=200" \
         $GPU experiments/e1_7_style_corpus.py)
echo "E1.7 Level 2 style corpus  -> verify   : $J_STYV"
J_Q25V=$(sub "$J_STYV" "RESULTS_ROOT=./results_verify,MODELS=$Q25,CONTROL_VARIANT=under" $GPU experiments/rq1.py)
echo "RQ1 full  $Q25         -> verify   : $J_Q25V"
J_Q35V=$(sub "$J_STYV" "RESULTS_ROOT=./results_verify,MODELS=$Q35,CONTROL_VARIANT=over" $GPU experiments/rq1.py)
echo "RQ1 full  $Q35         -> verify   : $J_Q35V"
J_MOEV=$(sub "$J_STYV" "RESULTS_ROOT=./results_verify,MODELS=$MOE,CONTROL_VARIANT=over" $GPU experiments/rq1.py)
echo "RQ1 full  $MOE   -> verify   : $J_MOEV"

echo
echo "Watch:  squeue -u \$USER"
echo "Then:   sbatch --dependency=afterok:$J_Q25O:$J_Q35:$J_MOE:$J_Q25V:$J_Q35V:$J_MOEV \\"
echo "               --export=ALL slurm/scripts/blank_slate_post.sh"
