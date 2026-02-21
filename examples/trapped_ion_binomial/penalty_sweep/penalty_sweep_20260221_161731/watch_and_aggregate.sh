#!/usr/bin/env bash
set -euo pipefail
QSTAT_BIN="/opt/pbs/default/bin/qstat"
if [[ ! -x "$QSTAT_BIN" ]]; then
  QSTAT_BIN="qstat"
fi
SWEEP_ROOT="/scratch/mu61/yl8164/quantum_control_rl_server/examples/trapped_ion_binomial/penalty_sweep/penalty_sweep_20260221_161731"
JOBS=(161275274.gadi-pbs 161275275.gadi-pbs 161275276.gadi-pbs 161275277.gadi-pbs 161275279.gadi-pbs)
LOG="$SWEEP_ROOT/aggregate_watch.log"

echo "[$(date)] watch start" >> "$LOG"
while true; do
  active=0
  for jid in "${JOBS[@]}"; do
    if "$QSTAT_BIN" "$jid" >/dev/null 2>&1; then
      active=1
      break
    fi
  done
  if [[ "$active" -eq 0 ]]; then
    break
  fi
  echo "[$(date)] jobs still active" >> "$LOG"
  sleep 180
done

echo "[$(date)] all jobs finished; running aggregate" >> "$LOG"
source /scratch/mu61/yl8164/qcrl_envs/venv_dq/bin/activate
python /scratch/mu61/yl8164/quantum_control_rl_server/gadi/aggregate_binomial_penalty_sweep.py --sweep-root "$SWEEP_ROOT" >> "$LOG" 2>&1
echo "[$(date)] aggregate done" >> "$LOG"
