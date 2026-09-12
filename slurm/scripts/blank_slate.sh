#!/bin/bash
# Blank-slate rerun of every implemented RQ1 experiment, on the full roster,
# into two independent results roots.
#
# WHY A DRIVER AND NOT sbatch BY HAND
# The RQ1 evidence accumulated over ~30 jobs interleaved with corrections, so no
# single job ever produced the whole result set under one code state. This
# submits the complete DAG with Slurm dependencies, so every artifact in the run
# comes from the same code.
#
# THE ROSTER — five models, three vendors, both architectures
#   qwen2.5-7b               Qwen    dense  28L
#   qwen3.5-9b               Qwen    dense  32L
#   llama3.1-8b              Meta    dense  32L
#   yi-6b-chat               01-AI   dense  32L  (Arditi-era, less aligned)
#   qwen3.5-35b-a3b          Qwen    MoE    40L
#   nemotron-3-nano-30b-a3b  NVIDIA  MoE    52L
# All five express all four role classes, so `R_role` is the same 4-class
# variable on every model and the cross-model comparison is like-for-like.
# The roster lives in `core/config.py:RQ1_MODELS`; this script reads it rather
# than repeating it, because a roster that disagrees with the tools produces a
# result set that silently covers a subset.
#
# WHY TWO ROOTS
# `results` is primary; `results_verify` is an independent repetition at
# IDENTICAL configuration. Identical is deliberate: it is what makes the
# verifier's exactness assertion on the label-independent concepts meaningful.
#
# WHY ONE MODEL PER JOB
# `sbatch --export=ALL,MODELS=a,b` does NOT work: sbatch splits --export on
# commas, so only the first slug survives and the rest are misparsed as further
# assignments. Every job below carries exactly one slug.
#
# CONTROL VARIANT
# Every model runs the MATCHED arm (`over`) so the cross-model gate compares
# like with like. The `under` arm — refused vs complied among harmful prompts —
# is stronger where it is estimable, but that depends on the model's own refusal
# behaviour and is not knowable at submission time. It is therefore submitted
# for every model with CONTROL_VARIANT_OPTIONAL=1, which records a skip and
# exits 0 where the harmful-and-complied cell is too thin. The verifier then
# checks each skip against that run's own labels.
#
# ACTIVATION CACHES LIVE ON /scratch
# `activations.pt` is ~1-2 GB per model per root — ~16 GB across this roster and
# two roots, against a ~101 GB home quota. It is a regenerable cache, so it is
# written to `core.config.CACHE_ROOT` keyed by the ABSOLUTE results root, which
# is what keeps `results` and `results_verify` from sharing one (they must not:
# the verify root exists to be an independent repetition). Each run drops an
# `activations_cache.json` pointer in its results dir.
#
#   bash slurm/scripts/blank_slate.sh          # submit
#   bash slurm/scripts/blank_slate.sh --dry    # print the plan only
set -euo pipefail
cd /home/b6aj/jtelintelo.b6aj/SecurityControl

DRY=""
[ "${1:-}" = "--dry" ] && DRY="echo [dry] "

GPU=slurm/scripts/run_gpu.sh
CPU=slurm/scripts/run_cpu.sh

# One log directory PER CAMPAIGN. The old flat `slurm/1_run_phase` mixed every
# run ever submitted, so "which job produced this artifact" meant grepping 93
# files from six different code states. `RUN_TAG` is overridable; the default
# stamps the date and the corpus scale, which is what actually distinguishes
# one campaign from the next.
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d-%H%M)-n$(python -c 'from core.config import load_config as c; x=c(); print(x.n_harmful+x.n_harmless)')}"
LOGDIR="slurm/logs/${RUN_TAG}"
mkdir -p "${LOGDIR}/out" "${LOGDIR}/err"
LOGFLAGS="--output=${PWD}/${LOGDIR}/out/%j-%x.out --error=${PWD}/${LOGDIR}/err/%j-%x.err"
echo "logs -> ${LOGDIR}"

# Wall-clock per model. E1.6 reads EVERY downstream layer, so the causal stage
# grows with roughly L^2: the 40-layer MoE exceeded 6h and the 52-layer one is
# ~1.7x that again. Sized from measured runs, not guessed — an under-budgeted
# job dies in `causal` with 13 of 14 stages done, which is the worst outcome.
# WALL-TIME IS CAPPED AT 24h by `workq_qos` (MaxWall 1-00:00:00) — the partition
# reports `infinite`, the QOS does not, and sbatch rejects anything longer with
# QOSMaxWallDurationPerJobLimit. At the 500/500 corpus the two MoE models need
# more than that (projected ~35h and ~60h), so each model is submitted as a CHAIN
# OF 24h SEGMENTS instead of one long job.
#
# This works without any new machinery: `core/stages.py` skips a stage whose
# declared `produces` all exist, so a segment that hits the wall is resumed by
# the next one, which re-does nothing. Segments are chained with `afterany`
# rather than `afterok` precisely so a TIMEOUT still triggers the resume; the
# final segment's exit status is what the post pass depends on.
#
# A model that finishes inside its first segment still starts the remaining ones,
# which find every stage complete and exit in about a minute. That is cheap
# insurance against an under-estimated budget, which has already cost this
# project one lost MoE run.
SEG_HOURS=24
declare -A SEGMENTS=(
  [qwen2.5-7b]=2              # ~16h projected
  [qwen3.5-9b]=2              # ~22h
  [llama3.1-8b]=2             # ~22h
  [yi-6b-chat]=2              # ~20h, 32L/6B
  [qwen3.5-35b-a3b]=2         # ~35h
  [nemotron-3-nano-30b-a3b]=3 # ~60h, 52 layers
)

MODELS_LIST=$(python -c 'from core.config import RQ1_MODELS; print(" ".join(RQ1_MODELS))')
MATCHED=$(python -c 'from core.config import MATCHED_CONTROL_VARIANT as v; print(v)')
echo "=== blank-slate RQ1 rerun, submitted $(date -Is) ==="
python -c 'from core.config import roster_table; print(roster_table())'
echo "matched control variant: $MATCHED"
echo

sub_any() {  # resume segment: afterany on the PREVIOUS segment, afterok on the GATE
  # Both dependencies, and the second one is not optional.
  #
  # With `afterany` alone, a resume segment runs whatever happened upstream —
  # including the case where the corpus job FAILED its checks, segment 1 was
  # cancelled as unsatisfiable, and segment 2 then started anyway and consumed a
  # corpus that `e1_0_corpus.py` had explicitly declared unusable ("Artifacts are
  # written anyway so the failure is inspectable, but downstream stages must not
  # consume this corpus"). That happened: 12 model jobs ran for ~14 minutes on an
  # invalid corpus before it was caught.
  #
  # `afterany:PREV,afterok:GATE` keeps the resume behaviour for a TIMEOUT while
  # restoring the gate. Slurm treats a comma-separated list as AND.
  local dep="$1"; shift
  local gate="$1"; shift
  local exp="$1"; shift
  local extra="$1"; shift
  local depflag=""
  if [ -n "$dep" ] && [ -n "$gate" ]; then
    depflag="--dependency=afterany:${dep},afterok:${gate}"
  elif [ -n "$dep" ]; then
    depflag="--dependency=afterany:${dep}"
  fi
  if [ -n "$DRY" ]; then
    echo "[dry] sbatch --parsable $depflag $LOGFLAGS $extra --export=ALL,$exp $*" >&2
    echo "DRYID"
  else
    sbatch --parsable $depflag $LOGFLAGS $extra --export="ALL,$exp" "$@"
  fi
}

sub() {  # sub <dependency-or-empty> <exports> <extra-sbatch-args> <script> <args...>
  local dep="$1"; shift
  local exp="$1"; shift
  local extra="$1"; shift
  local depflag=""
  [ -n "$dep" ] && depflag="--dependency=afterok:${dep}"
  if [ -n "$DRY" ]; then
    echo "[dry] sbatch --parsable $depflag $LOGFLAGS $extra --export=ALL,$exp $*" >&2
    echo "DRYID"
  else
    sbatch --parsable $depflag $LOGFLAGS $extra --export="ALL,$exp" "$@"
  fi
}

# ------------------------------------------------------------ preflight
# Cheap checks first: each guards a failure mode that is SILENT downstream, and
# each costs about two minutes against a multi-hour run.
J_NAM=$(sub "" "PREFLIGHT=1" "" $CPU tests/check_names.py)
echo "static: every name resolves             : $J_NAM"
# catches what name resolution cannot: pandas method/accessor shadowing
# (`g.tail`, `df.style`) and module attributes that do not exist
J_SHD=$(sub "" "PREFLIGHT=1" "" $CPU tests/check_shadowing.py)
echo "static: no attribute shadowing          : $J_SHD"
J_INV=$(sub "" "PREFLIGHT=1" "" $CPU tests/test_invariants.py)
echo "invariants (hooks, padding, dtypes)     : $J_INV"
# Roles are the variable the roster expansion touches: a template that drops a
# role message, or renders it as a tag the model never saw, fails NOWHERE else.
J_ROL=$(sub "" "PREFLIGHT=1" "" $CPU tests/smoke_roles.py)
echo "roles: all 4 classes on all 5 models    : $J_ROL"
# The estimators, against cases with known answers. Every other check in this
# project verifies structure or consistency; this is the only one that verifies
# a NUMBER. It found the AUC tie bug.
J_EST=$(sub "" "PREFLIGHT=1" "" $CPU tests/test_estimators.py)
echo "estimators reproduce known answers      : $J_EST"
# Mutation tests: prove the verifier's checks fail when they should. A check
# that cannot fail reports green forever; this found three that could not.
J_TEE=$(sub "" "PREFLIGHT=1" "" $CPU tests/test_verifier_teeth.py)
echo "verifier checks have teeth (mutations)  : $J_TEE"
J_RP=$(sub "" "PREFLIGHT=1" "" $GPU tests/smoke_read_positions.py)
echo "readout: per-item positions + all layers: $J_RP"
J_SP=$(sub "" "PREFLIGHT=1" "" $GPU tests/smoke_steer_positions.py)
echo "steering: position masks reach the hook : $J_SP"
# `generate_steered` is called at six sites in rq1.py, including the causal stage
# that produces GATE 1. Its smoke test existed but was never run by the chain — an
# unrun test of a live function is the worst of both worlds.
J_GS=$(sub "" "PREFLIGHT=1" "" $GPU tests/smoke_generate_steered.py)
echo "steering survives incremental decoding : $J_GS"
PRE="${J_NAM}:${J_SHD}:${J_INV}:${J_ROL}:${J_EST}:${J_TEE}:${J_RP}:${J_SP}:${J_GS}"
echo

# ------------------------------------------------------------ shakedown
# All 14 stages end-to-end on the 0.5B model at FAST_DEV scale, into a throwaway
# root. Its numbers are meaningless and are never read; what it establishes is
# that every stage RUNS under the current code. Without it, a crash in a late
# stage is discovered hours into a 24h job, after the expensive extraction is
# already done — which is exactly how the MoE arm was lost once before.
SHAKE_ROOT=./results_shakedown
J_SK1=$(sub "$PRE" "RESULTS_ROOT=$SHAKE_ROOT,FAST_DEV=1,MODELS=qwen2.5-0.5b" "" \
        $CPU experiments/e1_0_corpus.py)
# MIN_COMPLETE_BASES is lowered ONLY here. The real floor of 40 exists because a
# register contrast cannot be fitted on fewer; on the throwaway root the point is
# to execute `style_level2`, not to measure anything, and generating 100+ bases
# would make the shakedown as expensive as the thing it is guarding.
J_SK2=$(sub "$J_SK1" "RESULTS_ROOT=$SHAKE_ROOT,FAST_DEV=1,N_STYLE_BASE=8,MIN_COMPLETE_BASES=3,STYLE_MAX_NEW_TOKENS=120" \
        "--time=01:00:00" $GPU experiments/e1_7_style_corpus.py)
J_SK3=$(sub "$J_SK2" "RESULTS_ROOT=$SHAKE_ROOT,FAST_DEV=1,MODELS=qwen2.5-0.5b,CONTROL_VARIANT=auto" \
        "--time=02:00:00" $GPU experiments/rq1.py)
echo "shakedown: 14 stages end-to-end (0.5B)  : $J_SK1 -> $J_SK2 -> $J_SK3"
PRE="${PRE}:${J_SK3}"
echo

GATE_DEPS=""
for ROOT in ./results ./results_verify; do
  TAG=$([ "$ROOT" = "./results" ] && echo primary || echo verify)

  J_CORP=$(sub "$PRE" "RESULTS_ROOT=$ROOT,MODELS=$(echo $MODELS_LIST | tr ' ' '+')" "" \
           $CPU experiments/e1_0_corpus.py)
  echo "E1.0 corpus                -> $TAG : $J_CORP"
  J_STY=$(sub "$J_CORP" "RESULTS_ROOT=$ROOT,N_STYLE_BASE=168,STYLE_MAX_NEW_TOKENS=200" "" \
          $GPU experiments/e1_7_style_corpus.py)
  echo "E1.7 Level 2 style corpus  -> $TAG : $J_STY"

  for M in $MODELS_LIST; do
    NSEG=${SEGMENTS[$M]:-2}
    PREV="$J_STY"; DEPKIND="afterok"; SEGIDS=""
    for ((k = 1; k <= NSEG; k++)); do
      if [ "$DEPKIND" = "afterok" ]; then
        J=$(sub "$PREV" "RESULTS_ROOT=$ROOT,MODELS=$M,CONTROL_VARIANT=$MATCHED" \
            "--time=${SEG_HOURS}:00:00" $GPU experiments/rq1.py)
      else
        # afterany: a segment that hits the wall must still be resumed
        J=$(sub_any "$PREV" "$J_STY" "RESULTS_ROOT=$ROOT,MODELS=$M,CONTROL_VARIANT=$MATCHED" \
            "--time=${SEG_HOURS}:00:00" $GPU experiments/rq1.py)
      fi
      SEGIDS="${SEGIDS} ${J}"
      PREV="$J"; DEPKIND="afterany"
    done
    echo "RQ1 full  $M  -> $TAG : ${SEGIDS# }  ($MATCHED, ${NSEG}x${SEG_HOURS}h, resumed)"
    GATE_DEPS="${GATE_DEPS}:${PREV}"

    # The opportunistic stronger contrast — BOTH roots. Running it only in the
    # primary root made the two roots asymmetric: the `under` gate would have had
    # no independent repetition, while every other quantity did. It is cheap
    # (causal-only) and records a skip where the cell is too thin, so there is no
    # reason to measure fewer arms in one root than the other.
    if true; then
      JU=$(sub "$PREV" "RESULTS_ROOT=$ROOT,MODELS=$M,CONTROL_VARIANT=under,CONTROL_VARIANT_OPTIONAL=1" \
           "--time=${SEG_HOURS}:00:00" $GPU experiments/rq1.py --only causal)
      echo "  E1.6 under arm (optional)         : $JU"
      GATE_DEPS="${GATE_DEPS}:${JU}"
    fi
  done
  echo
done

GATE_DEPS="${GATE_DEPS#:}"
echo "================================================================"
echo "Watch:  squeue -u \$USER"
echo "Then:   sbatch --dependency=afterok:$GATE_DEPS \\"
echo "               --export=ALL slurm/scripts/blank_slate_post.sh"
