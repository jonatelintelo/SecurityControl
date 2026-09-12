#!/bin/bash
# Watches the RQ1 campaign. Exits 1 on a real failure, 0 when everything is done.
#
# Lessons baked in:
#  * `squeue` intermittently returns "Socket timed out" with a non-zero exit on
#    this cluster. A failed call is NOT an empty queue — treating it as one
#    would mean resubmitting a chain that is still running. Retried; ten
#    consecutive failures are reported as UNKNOWN, never as complete.
#  * TIMEOUT on a resume segment is EXPECTED (24h QOS cap; the next segment
#    resumes via the stage skip-if-complete check), so it is not a failure.
#  * Failures are matched against a KNOWN-HANDLED list so the watcher does not
#    re-report the same dead jobs forever after they have been dealt with.
cd /home/b6aj/jtelintelo.b6aj/SecurityControl
SINCE="${SINCE:-2026-09-11T14:22}"
# already diagnosed and resubmitted: nemotron's pre_mlp fidelity crash, and the
# post pass that Slurm cancelled when their afterok became unsatisfiable
HANDLED="${HANDLED:-6463663|6463664|6463665|6463685|6463686|6463687|6463706|6462544}"
fails=0
while true; do
  out=$(squeue -u "$USER" -h -o "%i" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then
    fails=$((fails+1))
    [ $fails -ge 10 ] && { echo "=== squeue unreachable 10x at $(date -Is) — state UNKNOWN ==="; exit 2; }
    sleep 60; continue
  fi
  fails=0
  bad=$(sacct -S "$SINCE" -X -n -o JobID,JobName%14,State 2>/dev/null \
        | grep -E "FAILED|OUT_OF_ME|NODE_FAIL" | grep -vE "$HANDLED" | grep -v fetch)
  if [ -n "$bad" ]; then
    echo "=== CHAIN PROBLEM at $(date -Is) ==="; echo "$bad"; exit 1
  fi
  if [ -z "$out" ]; then
    echo "=== CHAIN COMPLETE at $(date -Is) ==="
    sacct -S "$SINCE" -X -o JobID,JobName%14,State,Elapsed 2>/dev/null | tail -60
    exit 0
  fi
  sleep 600
done
