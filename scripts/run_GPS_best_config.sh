#!/usr/bin/env bash
# kill process: kill $(cat logs/run_GPS_best_config/launcher_*.pid)
set -euo pipefail

DETACH=0
ARGS=()

for arg in "$@"; do
  if [ "$arg" = "--detach" ]; then
    DETACH=1
  else
    ARGS+=("$arg")
  fi
done

LOG_DIR="logs/gps_table12"
mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
launcher_log="${LOG_DIR}/launcher_${timestamp}.log"
pid_file="${LOG_DIR}/launcher_${timestamp}.pid"

if [ "$DETACH" -eq 1 ]; then
  echo "Starting in detached mode..."
  nohup bash "$0" "${ARGS[@]}" > "$launcher_log" 2>&1 &
  bg_pid=$!
  echo "$bg_pid" > "$pid_file"
  echo "Detached successfully."
  echo "PID: $bg_pid"
  echo "Launcher log: $launcher_log"
  echo "PID file: $pid_file"
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot determine available GPUs." >&2
  exit 1
fi

GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)"

if [ -z "${GPU_LIST}" ]; then
  echo "No GPUs detected by nvidia-smi." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}"

CONDA_ROOT="${CONDA_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lyf_echo}"
PYTHON_BIN="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/python"
TRAIN_SCRIPT="scripts/train.py"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ROOT or CONDA_ENV_NAME before running this script." >&2
  exit 1
fi

run_task () {
  local task="$1"
  local num_layers="$2"
  local hidden_dim="$3"
  local lr="$4"
  local weight_decay="$5"
  local batch_size="$6"

  echo "============================================================"
  echo "[$(date '+%F %T')] Running GPS on task=${task}"
  echo "num_layers=${num_layers}, hidden_dim=${hidden_dim}, lr=${lr}, weight_decay=${weight_decay}, batch_size=${batch_size}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"

  "${PYTHON_BIN}" ${TRAIN_SCRIPT} \
    --task "${task}" \
    --gnn_type GNN \
    --conv_layer GPSConv \
    --hidden_dim "${hidden_dim}" \
    --num_layers "${num_layers}" \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --batch_size "${batch_size}" \
    2>&1 | tee "${LOG_DIR}/${task}.log"
}

# GPS Table 12 best hyperparameters
# batch_size uses conservative values to reduce OOM risk
# run_task diam   17  40   0.00004     0.00015    128
# run_task sssp   26  56   0.00031     0.00029    128
run_task ecc    17  162  0.00034     0.00007    64
run_task charge 36  216  0.00005     0.00005    32
# run_task energy 26  192  0.000024067 0.00038179 32

echo "[$(date '+%F %T')] All GPS Table 12 runs completed."
