#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ABLATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
PROJECT_ROOT="$(cd "${ABLATION_DIR}/../.." && pwd -P)"
cd "${PROJECT_ROOT}"

SEEDS="${SEEDS:-1 2 3 4}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
RUNS="${RUNS:-gps_ecc gps_diam grit_sssp}"
VARIANTS="${VARIANTS:-none pre_gt pre_ffn both}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ABLATION_DIR}/results}"
LOG_ROOT="${LOG_ROOT:-${ABLATION_DIR}/logs}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${ABLATION_DIR}/checkpoints}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-80}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_one () {
  local run_name="$1"
  local variant="$2"
  local seed="$3"
  local gpu_id="$4"
  local base_model task layers hidden batch lr wd dropout gps_heads gps_attn_dropout stride grit_heads grit_attn_dropout

  gps_heads=""
  gps_attn_dropout=""
  stride=""
  grit_heads=""
  grit_attn_dropout=""

  case "${run_name}" in
    gps_ecc)
      base_model=GPS; task=ecc; layers=19; hidden=192; batch=256
      lr=0.00028926866413819184; wd=5.4850931995345765e-05; dropout=0.05
      gps_heads=2; gps_attn_dropout=0.05; stride=4
      ;;
    gps_diam)
      base_model=GPS; task=diam; layers=21; hidden=80; batch=256
      lr=7.027506842162453e-05; wd=0.00018881921338068878; dropout=0.0
      gps_heads=2; gps_attn_dropout=0.03; stride=8
      ;;
    grit_sssp)
      base_model=GRIT; task=sssp; layers=44; hidden=96; batch=224
      lr=0.00028; wd=0.00015; dropout=0.05
      grit_heads=4; grit_attn_dropout=0.1
      ;;
    *)
      echo "Unknown run: ${run_name}" >&2
      exit 2
      ;;
  esac

  local output_dir="${OUTPUT_ROOT}/${run_name}/${variant}"
  local log_dir="${LOG_ROOT}/${run_name}/${variant}"
  local ckpt_dir="${CHECKPOINT_ROOT}/${run_name}/${variant}/seed_${seed}"
  local result_json="${output_dir}/seed_${seed}.json"
  local log_file="${log_dir}/seed_${seed}.log"
  mkdir -p "${output_dir}" "${log_dir}" "${ckpt_dir}"

  if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${result_json}" ]; then
    echo "[$(date '+%F %T')] Skip existing ${run_name}/${variant}/seed_${seed}"
    return 0
  fi

  local model_args=()
  if [ "${base_model}" = "GPS" ]; then
    model_args=(
      --gps_num_heads "${gps_heads}"
      --gps_attn_dropout "${gps_attn_dropout}"
      --gps_attn_type multihead
      --attnres_history_stride "${stride}"
    )
  else
    model_args=(
      --grit_num_heads "${grit_heads}"
      --grit_attn_dropout "${grit_attn_dropout}"
      --grit_act relu
    )
  fi

  echo "[$(date '+%F %T')] run=${run_name} variant=${variant} seed=${seed} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" "${ABLATION_DIR}/train/train_position.py" \
    --task "${task}" \
    --base_model "${base_model}" \
    --position_variant "${variant}" \
    --num_layers "${layers}" \
    --hidden_dim "${hidden}" \
    --batch_size "${batch}" \
    --lr "${lr}" \
    --weight_decay "${wd}" \
    --dropout_prob "${dropout}" \
    --seed "${seed}" \
    --monitor_metric val_mae \
    --max_epochs "${MAX_EPOCHS}" \
    --early_stopping_patience "${PATIENCE}" \
    --num_workers "${NUM_WORKERS}" \
    --quiet \
    --checkpoint_dir "${ckpt_dir}" \
    --default_root_dir "${ABLATION_DIR}" \
    --result_json "${result_json}" \
    "${model_args[@]}" \
    2>&1 | tee "${log_file}"
}

read -r -a seed_array <<< "${SEEDS}"
read -r -a gpu_array <<< "${GPU_IDS}"

if [ "${#seed_array[@]}" -gt "${#gpu_array[@]}" ]; then
  echo "Need at least one GPU ID per concurrently executed seed." >&2
  echo "Seeds: ${SEEDS}" >&2
  echo "GPU IDs: ${GPU_IDS}" >&2
  exit 2
fi

for run_name in ${RUNS}; do
  for variant in ${VARIANTS}; do
    echo "============================================================"
    echo "Starting position ablation: run=${run_name}, variant=${variant}"
    echo "============================================================"

    pids=()
    labels=()
    for index in "${!seed_array[@]}"; do
      seed="${seed_array[$index]}"
      gpu_id="${gpu_array[$index]}"
      run_one "${run_name}" "${variant}" "${seed}" "${gpu_id}" &
      pids+=("$!")
      labels+=("${run_name}/${variant}/seed_${seed}/gpu_${gpu_id}")
    done

    failed=0
    for index in "${!pids[@]}"; do
      if ! wait "${pids[$index]}"; then
        echo "FAILED: ${labels[$index]}" >&2
        failed=1
      fi
    done

    if [ "${failed}" -ne 0 ]; then
      echo "Stopping because at least one seed failed for ${run_name}/${variant}." >&2
      exit 1
    fi
  done
done

"${PYTHON_BIN}" "${ABLATION_DIR}/scripts/collect_results.py" \
  --input_dir "${OUTPUT_ROOT}" \
  --output_csv "${OUTPUT_ROOT}/summary.csv"

