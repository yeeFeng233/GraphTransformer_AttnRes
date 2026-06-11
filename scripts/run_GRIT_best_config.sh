#!/usr/bin/env bash
# kill process: kill $(cat logs/run_GRIT_best_config/launcher_*.pid)
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

LOG_DIR="logs/grit_table14"
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

export CUDA_VISIBLE_DEVICES=4

PYTHON_BIN="/root/.local/bin/uv run python"
TRAIN_SCRIPT="scripts/train.py"

run_task () {
  local task="$1"
  local num_layers="$2"
  local hidden_dim="$3"
  local lr="$4"
  local weight_decay="$5"
  local batch_size="$6"
  local num_heads=2
  local attn_drop="$7"

  echo "============================================================"
  echo "[$(date '+%F %T')] Running GRIT on task=${task}"
  echo "num_layers=${num_layers}, hidden_dim=${hidden_dim}, lr=${lr}, weight_decay=${weight_decay}, batch_size=${batch_size}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"

  ${PYTHON_BIN} ${TRAIN_SCRIPT} \
    --task "${task}" \
    --gnn_type GNN \
    --conv_layer GRIT \
    --hidden_dim "${hidden_dim}" \
    --num_layers "${num_layers}" \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --batch_size "${batch_size}" \
    --grit_num_heads "${num_heads}" \
    --grit_attn_dropout "${attn_drop}" \
    2>&1 | tee "${LOG_DIR}/${task}.log"
}

# GPS Table 12 best hyperparameters
# batch_size uses conservative values to reduce OOM risk
# run_task diam   17  40   0.00004     0.00015    128
# run_task sssp   26  56   0.00031     0.00029    128
# run_task ecc    17  162  0.00034     0.00007    64
# run_task charge 36  216  0.00005     0.00005    32
# run_task energy 26  192  0.000024067 0.00038179 32

# GRIT Table 14 best hyperparameters
# batch_size uses conservative values to reduce OOM risk
#run_task diam   32  256   0.00082     0.00032    256    0.433
#run_task sssp   40  128   0.00048     0.00047    256    0.008
#run_task charge 32  128  0.00034     0.00034    256    0.351
run_task energy 8   64   0.00076     0.00094    256    0.178


echo "[$(date '+%F %T')] All GRIT Table 14 runs completed."
