#!/bin/bash
# Watch whatever this user currently has queued/running; exit on failure or completion.
# Retries transient squeue socket timeouts (a failed call is not an empty queue).
cd /home/b6aj/jtelintelo.b6aj/SecurityControl
SINCE="${SINCE:-$(date -d '2 hours ago' -Iseconds | cut -c1-16)}"
# Known-handled, so the watcher does not re-report jobs already dealt with:
#   6462544            fetch invoked with the model as --export instead of argv
#   6463663-687        nemotron pre_mlp fidelity crash (fixed: has_site guard)
#   6463706            post pass Slurm-cancelled when those became unsatisfiable
#   6485666            post pass exits with the COUNT of non-zero steps by design
#                      ("NOT -e: a failing check must not hide the checks after
#                      it"), so exit 8 means 8 steps reported problems, not a
#                      crash. Slurm records that as FAILED and the watcher cannot
#                      tell the difference from the state alone.
HANDLED="${HANDLED:-6463663|6463664|6463665|6463685|6463686|6463687|6463706|6462544|6485666}"
f=0
while true; do
  out=$(squeue -u "$USER" -h -o "%i" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then
    f=$((f+1)); [ $f -ge 10 ] && { echo "squeue unreachable 10x at $(date -Is) — UNKNOWN"; exit 2; }
    sleep 60; continue
  fi
  f=0
  bad=$(sacct -S "$SINCE" -X -n -o JobID,JobName%14,State 2>/dev/null \
        | grep -E "FAILED|OUT_OF_ME|NODE_FAIL" | grep -vE "$HANDLED" | grep -v fetch)
  [ -n "$bad" ] && { echo "=== PROBLEM at $(date -Is) ==="; echo "$bad"; exit 1; }
  [ -z "$out" ] && { echo "=== ALL JOBS DONE at $(date -Is) ==="
                     sacct -S "$SINCE" -X -o JobID,JobName%14,State,Elapsed 2>/dev/null | tail -25; exit 0; }
  sleep 300
done
