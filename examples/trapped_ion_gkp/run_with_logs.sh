#!/usr/bin/env bash
set -euo pipefail

# One-click launcher: starts training server + client, logs output,
# and auto-generates plots when training finishes.
# Run this inside an activated environment (venv/conda) on a compute node.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/outputs/logs"
mkdir -p "${LOG_DIR}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5555}"

# Optional split interpreters:
#   SERVER_PYTHON=/path/to/tf/python
#   CLIENT_PYTHON=/path/to/dq/python
#   POST_PYTHON=/path/to/python_with_matplotlib
SERVER_PYTHON="${SERVER_PYTHON:-python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-python}"
POST_PYTHON="${POST_PYTHON:-${CLIENT_PYTHON}}"

TS="$(date +"%Y%m%d_%H%M%S")"
SERVER_LOG="${LOG_DIR}/server_${TS}.log"
CLIENT_LOG="${LOG_DIR}/client_${TS}.log"
RUNNER_LOG="${LOG_DIR}/runner_${TS}.log"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "Logs:"
echo "  server -> ${SERVER_LOG}"
echo "  client -> ${CLIENT_LOG}"
echo "Plots will be generated into ${ROOT_DIR}/outputs/"

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Training server (background)
"${SERVER_PYTHON}" -u "${ROOT_DIR}/trapped_ion_gkp_training_server.py" 2>&1 | tee -a "${SERVER_LOG}" &
SERVER_PID=$!

SERVER_STARTUP_WAIT="${SERVER_STARTUP_WAIT:-90}"
start_ts="$(date +%s)"
while true; do
  # If server process died, fail fast.
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Server process exited before client start. Check ${SERVER_LOG}" >&2
    exit 1
  fi
  # Wait until socket is listening without sending protocol bytes.
  if command -v ss >/dev/null 2>&1; then
    if ss -lnt | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
      break
    fi
  elif command -v netstat >/dev/null 2>&1; then
    if netstat -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)${PORT}$"; then
      break
    fi
  else
    # Fallback: no listener-inspection tool available.
    sleep 1
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= SERVER_STARTUP_WAIT )); then
      break
    fi
    continue
  fi
  if [[ "${SERVER_STARTUP_WAIT}" -eq 0 ]]; then
    break
  fi
  now_ts="$(date +%s)"
  if (( now_ts - start_ts >= SERVER_STARTUP_WAIT )); then
    echo "Server did not open ${HOST}:${PORT} within ${SERVER_STARTUP_WAIT}s. Check ${SERVER_LOG}" >&2
    exit 1
  fi
  sleep 1
done
echo "Server is ready on ${HOST}:${PORT}" | tee -a "${RUNNER_LOG}"

# Client (foreground)
"${CLIENT_PYTHON}" -u "${ROOT_DIR}/trapped_ion_gkp_client.py" 2>&1 | tee -a "${CLIENT_LOG}"

# Post-processing
if [[ "${SKIP_PLOTS:-0}" != "1" ]]; then
  "${POST_PYTHON}" -u "${ROOT_DIR}/parse_trapped_ion_gkp_data.py" 2>&1 | tee -a "${RUNNER_LOG}"
  "${POST_PYTHON}" -u "${ROOT_DIR}/plot_trapped_ion_gkp_pulses.py" 2>&1 | tee -a "${RUNNER_LOG}"
else
  echo "SKIP_PLOTS=1, skip parse/plot scripts" | tee -a "${RUNNER_LOG}"
fi
