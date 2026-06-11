#!/usr/bin/env bash
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

PROJECT_DIR="$(pwd -P)"
LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/gps_energy_reproduce}"
mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
launcher_log="${LOG_DIR}/launcher_${timestamp}.log"
pid_file="${LOG_DIR}/launcher_${timestamp}.pid"
run_log="${LOG_DIR}/energy_gps_${timestamp}.log"

if [ "$DETACH" -eq 1 ]; then
  echo "Starting GPS energy reproduction in detached mode..."
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

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST%%,*}}"

CONDA_ROOT="${CONDA_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lyf_echo}"
PYTHON_BIN="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/python"
TRAIN_SCRIPT="scripts/train.py"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ROOT or CONDA_ENV_NAME before running this script." >&2
  exit 1
fi

TASK="${TASK:-energy}"
NUM_LAYERS="${NUM_LAYERS:-26}"
HIDDEN_DIM="${HIDDEN_DIM:-192}"
LR="${LR:-0.000024067}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.00038179}"
BATCH_SIZE="${BATCH_SIZE:-256}"
GPS_NUM_HEADS="${GPS_NUM_HEADS:-2}"
GPS_ATTN_DROPOUT="${GPS_ATTN_DROPOUT:-0.0}"
GPS_ATTN_TYPE="${GPS_ATTN_TYPE:-multihead}"
DROPOUT_PROB="${DROPOUT_PROB:-0.0}"

echo "============================================================" | tee "$run_log"
echo "[$(date '+%F %T')] Reproducing GPS Table 12 energy configuration with unified batch size" | tee -a "$run_log"
echo "task=${TASK}" | tee -a "$run_log"
echo "num_layers=${NUM_LAYERS}, hidden_dim=${HIDDEN_DIM}, batch_size=${BATCH_SIZE}" | tee -a "$run_log"
echo "lr=${LR}, weight_decay=${WEIGHT_DECAY}" | tee -a "$run_log"
echo "gps_num_heads=${GPS_NUM_HEADS}, gps_attn_dropout=${GPS_ATTN_DROPOUT}, dropout_prob=${DROPOUT_PROB}" | tee -a "$run_log"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "$run_log"
echo "Log: ${run_log}" | tee -a "$run_log"
echo "============================================================" | tee -a "$run_log"

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
  --task "${TASK}" \
  --gnn_type GNN \
  --conv_layer GPSConv \
  --hidden_dim "${HIDDEN_DIM}" \
  --num_layers "${NUM_LAYERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --batch_size "${BATCH_SIZE}" \
  --dropout_prob "${DROPOUT_PROB}" \
  --gps_num_heads "${GPS_NUM_HEADS}" \
  --gps_attn_dropout "${GPS_ATTN_DROPOUT}" \
  --gps_attn_type "${GPS_ATTN_TYPE}" \
  --quiet \
  2>&1 | tee -a "$run_log"

echo "[$(date '+%F %T')] GPS energy reproduction completed." | tee -a "$run_log"
