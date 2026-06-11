#!/usr/bin/env bash
# kill process: kill $(cat logs/gps_attnres_best_config/launcher_*.pid)
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

LOG_DIR="logs/gps_attnres_best_config"
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
GPU_COUNT="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | awk 'NF {count++} END {print count+0}')"

if [ "${GPU_COUNT}" -ne 1 ]; then
  echo "This runner is tuned for single-GPU throughput checks." >&2
  echo "Visible GPUs: ${CUDA_VISIBLE_DEVICES}" >&2
fi

CONDA_ROOT="${CONDA_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lyf_echo}"
PYTHON_BIN="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/python"
TRAIN_SCRIPT="scripts/train.py"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ROOT or CONDA_ENV_NAME before running this script." >&2
  exit 1
fi

AUTO_TUNE_BATCH="${AUTO_TUNE_BATCH:-1}"
BATCH_MARGIN_NUM="${BATCH_MARGIN_NUM:-7}"
BATCH_MARGIN_DEN="${BATCH_MARGIN_DEN:-8}"
CHARGE_START_BATCH="${CHARGE_START_BATCH:-256}"
ENERGY_START_BATCH="${ENERGY_START_BATCH:-512}"
CHARGE_MAX_PROBE_BATCH="${CHARGE_MAX_PROBE_BATCH:-4096}"
ENERGY_MAX_PROBE_BATCH="${ENERGY_MAX_PROBE_BATCH:-8192}"
CHARGE_BATCH_STEP="${CHARGE_BATCH_STEP:-32}"
ENERGY_BATCH_STEP="${ENERGY_BATCH_STEP:-64}"
CHARGE_BATCH_SIZE="${CHARGE_BATCH_SIZE:-128}"
ENERGY_BATCH_SIZE="${ENERGY_BATCH_SIZE:-256}"

probe_batch_size () {
  local task="$1"
  local num_layers="$2"
  local hidden_dim="$3"
  local batch_size="$4"
  local gps_num_heads="$5"
  local gps_attn_dropout="$6"
  local dropout_prob="$7"
  local attnres_history_stride="$8"

  TASK="${task}" \
  NUM_LAYERS="${num_layers}" \
  HIDDEN_DIM="${hidden_dim}" \
  BATCH_SIZE="${batch_size}" \
  GPS_NUM_HEADS="${gps_num_heads}" \
  GPS_ATTN_DROPOUT="${gps_attn_dropout}" \
  DROPOUT_PROB="${dropout_prob}" \
  ATTNRES_HISTORY_STRIDE="${attnres_history_stride}" \
  "${PYTHON_BIN}" - <<'PY'
import os
import sys

import torch
from torch_geometric.loader import DataLoader

from models.gnn import GNN
from utils import get_dataset


def cast_batch(batch):
    if batch.x.dtype == torch.float64:
        batch.x = batch.x.float()
    if getattr(batch, "edge_attr", None) is not None and batch.edge_attr.dtype == torch.float64:
        batch.edge_attr = batch.edge_attr.float()
    if batch.y.dtype == torch.float64:
        batch.y = batch.y.float()
    return batch


task = os.environ["TASK"]
num_layers = int(os.environ["NUM_LAYERS"])
hidden_dim = int(os.environ["HIDDEN_DIM"])
batch_size = int(os.environ["BATCH_SIZE"])
gps_num_heads = int(os.environ["GPS_NUM_HEADS"])
gps_attn_dropout = float(os.environ["GPS_ATTN_DROPOUT"])
dropout_prob = float(os.environ["DROPOUT_PROB"])
attnres_history_stride = int(os.environ["ATTNRES_HISTORY_STRIDE"])

try:
    data_train, _, _, num_feat, num_class = get_dataset(root="./data/", task=task, constant_feature=None)
    loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    batch = cast_batch(next(iter(loader)))

    model = GNN(
        input_dim=num_feat,
        output_dim=num_class,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        node_level_task=task not in ["diam", "energy"],
        conv_layer="GPSAttnRes",
        dropout_prob=dropout_prob,
        edge_dim=2,
        gps_num_heads=gps_num_heads,
        gps_attn_dropout=gps_attn_dropout,
        gps_attn_type="multihead",
        attnres_history_stride=attnres_history_stride,
    ).cuda()
    model.train()

    batch = batch.cuda()
    torch.cuda.reset_peak_memory_stats()
    out = model(batch).squeeze(-1)
    loss = torch.nn.functional.mse_loss(out, batch.y)
    loss.backward()
    torch.cuda.synchronize()
    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"BATCH_OK {batch_size} {peak_gb:.2f}")
except RuntimeError as exc:
    if "out of memory" in str(exc).lower():
        print(f"BATCH_OOM {batch_size}")
        sys.exit(88)
    raise
PY
}

find_batch_size () {
  local task="$1"
  local num_layers="$2"
  local hidden_dim="$3"
  local gps_num_heads="$4"
  local gps_attn_dropout="$5"
  local dropout_prob="$6"
  local attnres_history_stride="$7"
  local start_batch="$8"
  local max_probe_batch="$9"
  local step="${10}"

  local low=0
  local high="${start_batch}"

  echo "[$(date '+%F %T')] Probing batch size for ${task} (start=${start_batch}, max=${max_probe_batch}, step=${step})"

  while [ "${high}" -le "${max_probe_batch}" ]; do
    if probe_batch_size "${task}" "${num_layers}" "${hidden_dim}" "${high}" "${gps_num_heads}" "${gps_attn_dropout}" "${dropout_prob}" "${attnres_history_stride}"; then
      low="${high}"
      if [ "${high}" -eq "${max_probe_batch}" ]; then
        break
      fi
      high=$(( high * 2 ))
      if [ "${high}" -gt "${max_probe_batch}" ]; then
        high="${max_probe_batch}"
      fi
    else
      status=$?
      if [ "${status}" -ne 88 ]; then
        echo "Batch probe failed unexpectedly for ${task} at batch_size=${high}." >&2
        exit "${status}"
      fi
      break
    fi
  done

  if [ "${low}" -eq 0 ]; then
    echo "Even the starting batch size ${start_batch} does not fit for ${task}." >&2
    exit 1
  fi

  local left="${low}"
  local right="${high}"

  if [ "${left}" -lt "${right}" ]; then
    while [ $(( right - left )) -gt "${step}" ]; do
      local mid=$(( (left + right) / 2 ))
      mid=$(( (mid / step) * step ))
      if [ "${mid}" -le "${left}" ]; then
        mid=$(( left + step ))
      fi
      if [ "${mid}" -ge "${right}" ]; then
        mid=$(( right - step ))
      fi
      if [ "${mid}" -le "${left}" ] || [ "${mid}" -ge "${right}" ]; then
        break
      fi

      if probe_batch_size "${task}" "${num_layers}" "${hidden_dim}" "${mid}" "${gps_num_heads}" "${gps_attn_dropout}" "${dropout_prob}" "${attnres_history_stride}"; then
        left="${mid}"
      else
        status=$?
        if [ "${status}" -ne 88 ]; then
          echo "Binary-search probe failed unexpectedly for ${task} at batch_size=${mid}." >&2
          exit "${status}"
        fi
        right="${mid}"
      fi
    done
  fi

  local usable=$(( left * BATCH_MARGIN_NUM / BATCH_MARGIN_DEN ))
  usable=$(( (usable / step) * step ))
  if [ "${usable}" -lt "${step}" ]; then
    usable="${step}"
  fi
  if [ "${usable}" -gt "${left}" ]; then
    usable="${left}"
  fi

  echo "[$(date '+%F %T')] Batch probe result for ${task}: max_fit=${left}, chosen=${usable}"
  echo "${usable}"
}

run_task () {
  local task="$1"
  local num_layers="$2"
  local hidden_dim="$3"
  local lr="$4"
  local weight_decay="$5"
  local batch_size="$6"
  local gps_num_heads="$7"
  local gps_attn_dropout="$8"
  local dropout_prob="$9"
  local attnres_history_stride="${10}"
  local start_batch="${11}"
  local max_probe_batch="${12}"
  local batch_step="${13}"

  if [ "${AUTO_TUNE_BATCH}" = "1" ]; then
    batch_size="$(find_batch_size "${task}" "${num_layers}" "${hidden_dim}" "${gps_num_heads}" "${gps_attn_dropout}" "${dropout_prob}" "${attnres_history_stride}" "${start_batch}" "${max_probe_batch}" "${batch_step}" | tail -n 1)"
  fi

  echo "============================================================"
  echo "[$(date '+%F %T')] Running GPSAttnRes on task=${task}"
  echo "num_layers=${num_layers}, hidden_dim=${hidden_dim}, lr=${lr}, weight_decay=${weight_decay}, batch_size=${batch_size}"
  echo "gps_num_heads=${gps_num_heads}, gps_attn_dropout=${gps_attn_dropout}, dropout_prob=${dropout_prob}, attnres_history_stride=${attnres_history_stride}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "============================================================"

  "${PYTHON_BIN}" ${TRAIN_SCRIPT} \
    --task "${task}" \
    --gnn_type GNN \
    --conv_layer GPSAttnRes \
    --hidden_dim "${hidden_dim}" \
    --num_layers "${num_layers}" \
    --lr "${lr}" \
    --weight_decay "${weight_decay}" \
    --batch_size "${batch_size}" \
    --gps_num_heads "${gps_num_heads}" \
    --gps_attn_dropout "${gps_attn_dropout}" \
    --gps_attn_type multihead \
    --dropout_prob "${dropout_prob}" \
    --attnres_history_stride "${attnres_history_stride}" \
    2>&1 | tee "${LOG_DIR}/${task}_GPSAttnRes.log"
}

# Throughput-oriented bootstrap configs for a single A800.
# Batch size is auto-tuned by default so the card is used much more aggressively.
# run_task charge 24 160 0.00005 0.00005 "${CHARGE_BATCH_SIZE}" 4 0.05 0.05 2 "${CHARGE_START_BATCH}" "${CHARGE_MAX_PROBE_BATCH}" "${CHARGE_BATCH_STEP}"
run_task energy 16 128 0.00004 0.00020 "${ENERGY_BATCH_SIZE}" 4 0.05 0.05 2 "${ENERGY_START_BATCH}" "${ENERGY_MAX_PROBE_BATCH}" "${ENERGY_BATCH_STEP}"

# Useful overrides:
# AUTO_TUNE_BATCH=0 CHARGE_BATCH_SIZE=512 ENERGY_BATCH_SIZE=1024 bash scripts/run_GPSAttnRes_best_config.sh
# CHARGE_START_BATCH=512 CHARGE_MAX_PROBE_BATCH=8192 ENERGY_START_BATCH=1024 ENERGY_MAX_PROBE_BATCH=16384 bash scripts/run_GPSAttnRes_best_config.sh
# BATCH_MARGIN_NUM=15 BATCH_MARGIN_DEN=16 bash scripts/run_GPSAttnRes_best_config.sh

echo "[$(date '+%F %T')] All GPSAttnRes best-config runs completed."
