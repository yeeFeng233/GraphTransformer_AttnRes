#!/usr/bin/env bash
set -euo pipefail

DETACH="${DETACH:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BEST_CONFIGS="${BEST_CONFIGS:-configs/best_attnres.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/multiseed}"
SEEDS="${SEEDS:-1 2 3 4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S_%N')}"

if [[ "${1:-}" != "--worker" && "${DETACH}" == "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  log_file="${OUTPUT_ROOT}/logs/${RUN_TAG}.log"
  pid_file="${OUTPUT_ROOT}/logs/${RUN_TAG}.pid"
  DETACH=0 nohup bash "$0" --worker >"${log_file}" 2>&1 &
  echo "$!" >"${pid_file}"
  echo "Started AttnRes four-seed runs."
  echo "PID: $(cat "${pid_file}")"
  echo "Log: ${log_file}"
  exit 0
fi

read -r -a seed_args <<<"${SEEDS}"
read -r -a gpu_args <<<"${GPU_IDS}"

"${PYTHON_BIN}" scripts/run_attnres_multiseed.py \
  --best_configs "${BEST_CONFIGS}" \
  --output_root "${OUTPUT_ROOT}" \
  --seeds "${seed_args[@]}" \
  --gpus "${gpu_args[@]}" \
  --max_epochs "${MAX_EPOCHS}" \
  --patience "${PATIENCE}" \
  --num_workers "${NUM_WORKERS}" \
  --monitor_metric val_mae \
  --skip_existing

"${PYTHON_BIN}" scripts/summarize_multiseed.py \
  --input_root "${OUTPUT_ROOT}" \
  --output_csv "${OUTPUT_ROOT}/summary.csv"
