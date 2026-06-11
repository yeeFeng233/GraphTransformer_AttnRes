#!/usr/bin/env bash
set -euo pipefail

# Sequential launcher for GRIT+AttnRes on all ECHO-Synth tasks.
# On 2xA800 servers, each task launcher defaults to two concurrent 1-GPU trials.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
TASKS="${TASKS:-diam ecc sssp}"
DETACH=0
ARGS=()

for arg in "$@"; do
  if [ "$arg" = "--detach" ]; then
    DETACH=1
  else
    ARGS+=("$arg")
  fi
done

PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/grit_attnres_synth_search}"
mkdir -p "${LOG_DIR}"

if [ "${DETACH}" -eq 1 ]; then
  timestamp="$(date +%Y%m%d_%H%M%S)"
  launcher_log="${LOG_DIR}/launcher_${timestamp}.log"
  pid_file="${LOG_DIR}/launcher_${timestamp}.pid"
  echo "Starting GRIT+AttnRes ECHO-Synth sequential search in detached mode..."
  nohup bash "${SCRIPT_PATH}" "${ARGS[@]}" > "${launcher_log}" 2>&1 &
  bg_pid=$!
  echo "${bg_pid}" > "${pid_file}"
  echo "Detached successfully."
  echo "PID: ${bg_pid}"
  echo "Launcher log: ${launcher_log}"
  echo "PID file: ${pid_file}"
  exit 0
fi

for task in ${TASKS}; do
  echo "============================================================"
  echo "[$(date '+%F %T')] Launching GRIT+AttnRes ECHO-Synth search: ${task}"
  echo "============================================================"
  TASK_NAME="${task}" bash "${SCRIPT_DIR}/run_GRITAttnRes_diam_search.sh" "${ARGS[@]}"
done

echo "[$(date '+%F %T')] GRIT+AttnRes ECHO-Synth searches completed for: ${TASKS}"
