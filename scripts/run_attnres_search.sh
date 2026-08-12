#!/usr/bin/env bash
set -euo pipefail

TASKS="${TASKS:-diam ecc sssp charge energy}"
MODELS="${MODELS:-gps_attnres grit_attnres}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_CPUS="${NUM_CPUS:-32}"
GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-1}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-6}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_SAMPLES="${N_SAMPLES:-24}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
REPORT_EVERY="${REPORT_EVERY:-5}"
PATIENCE="${PATIENCE:-40}"
ASHA_GRACE_EPOCHS="${ASHA_GRACE_EPOCHS:-200}"
MONITOR_METRIC="${MONITOR_METRIC:-val_mae}"
DETACH="${DETACH:-1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/search}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S_%N')}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ar_${UID:-user}}"
AUTO_MULTI_SEED="${AUTO_MULTI_SEED:-0}"
BEST_CONFIGS_OUTPUT="${BEST_CONFIGS_OUTPUT:-${OUTPUT_ROOT}/best_configs.csv}"
MULTISEED_OUTPUT_ROOT="${MULTISEED_OUTPUT_ROOT:-${OUTPUT_ROOT}/multiseed}"
FINAL_SEEDS="${FINAL_SEEDS:-1 2 3 4}"
FINAL_GPU_IDS="${FINAL_GPU_IDS:-0 1 2 3}"
FINAL_MAX_EPOCHS="${FINAL_MAX_EPOCHS:-1000}"
FINAL_PATIENCE="${FINAL_PATIENCE:-100}"
FINAL_NUM_WORKERS="${FINAL_NUM_WORKERS:-8}"

if [[ "${1:-}" != "--worker" && "${DETACH}" == "1" ]]; then
  mkdir -p "${OUTPUT_ROOT}/logs"
  log_file="${OUTPUT_ROOT}/logs/${RUN_TAG}.log"
  pid_file="${OUTPUT_ROOT}/logs/${RUN_TAG}.pid"
  DETACH=0 nohup bash "$0" --worker >"${log_file}" 2>&1 &
  echo "$!" >"${pid_file}"
  if [[ "${AUTO_MULTI_SEED}" == "1" ]]; then
    echo "Started AttnRes search-to-four-seed pipeline."
  else
    echo "Started AttnRes search."
  fi
  echo "PID: $(cat "${pid_file}")"
  echo "Log: ${log_file}"
  exit 0
fi

mkdir -p \
  "${OUTPUT_ROOT}/csv" \
  "${OUTPUT_ROOT}/trial_logs" \
  "${OUTPUT_ROOT}/ray_results" \
  "${RAY_TEMP_DIR}"

read -r -a task_args <<<"${TASKS}"
read -r -a model_args <<<"${MODELS}"
ASHA_GRACE_REPORTS=$(( (ASHA_GRACE_EPOCHS + REPORT_EVERY - 1) / REPORT_EVERY ))

if [[ "${AUTO_MULTI_SEED}" == "1" ]]; then
  if [[ "${MONITOR_METRIC}" != "val_mae" ]]; then
    echo "AUTO_MULTI_SEED requires MONITOR_METRIC=val_mae." >&2
    exit 2
  fi
  read -r -a final_seed_args <<<"${FINAL_SEEDS}"
  read -r -a final_gpu_args <<<"${FINAL_GPU_IDS}"
  if [[ "${#final_seed_args[@]}" -ne 4 ]]; then
    echo "AUTO_MULTI_SEED requires exactly four FINAL_SEEDS." >&2
    exit 2
  fi
  if [[ "${#final_gpu_args[@]}" -lt 4 ]]; then
    echo "AUTO_MULTI_SEED requires at least four FINAL_GPU_IDS." >&2
    exit 2
  fi
fi

expected_models=()
for model in "${model_args[@]}"; do
  case "${model}" in
    gps_attnres)
      expected_models+=("GPSAttnRes")
      ;;
    grit_attnres)
      expected_models+=("GRITAttnRes")
      ;;
    *)
      echo "Unsupported model selector: ${model}" >&2
      exit 2
      ;;
  esac
done

"${PYTHON_BIN}" scripts/search.py \
  --tasks "${task_args[@]}" \
  --models "${model_args[@]}" \
  --n_samples "${N_SAMPLES}" \
  --num_gpus "${NUM_GPUS}" \
  --num_cpus "${NUM_CPUS}" \
  --gpus_per_trial "${GPUS_PER_TRIAL}" \
  --cpus_per_trial "${CPUS_PER_TRIAL}" \
  --num_workers "${NUM_WORKERS}" \
  --max_epochs "${MAX_EPOCHS}" \
  --report_every_n_epochs "${REPORT_EVERY}" \
  --early_stopping_patience "${PATIENCE}" \
  --asha_grace_period "${ASHA_GRACE_REPORTS}" \
  --monitor_metric "${MONITOR_METRIC}" \
  --logger csv \
  --scheduler asha \
  --search_alg optuna \
  --output_dir "${OUTPUT_ROOT}/csv" \
  --training_log_dir "${OUTPUT_ROOT}/trial_logs" \
  --storage_path "$(pwd)/${OUTPUT_ROOT}/ray_results" \
  --ray_temp_dir "${RAY_TEMP_DIR}"

if [[ "${AUTO_MULTI_SEED}" == "1" ]]; then
  echo "[pipeline] Search complete; selecting configurations by validation MAE."
  "${PYTHON_BIN}" scripts/select_best_configs.py \
    --input_dir "${OUTPUT_ROOT}/csv" \
    --output_csv "${BEST_CONFIGS_OUTPUT}" \
    --metric val_mae \
    --require_tasks "${task_args[@]}" \
    --require_models "${expected_models[@]}"

  echo "[pipeline] Best configurations: ${BEST_CONFIGS_OUTPUT}"
  echo "[pipeline] Starting fixed-configuration four-seed evaluation."
  DETACH=0 \
  PYTHON_BIN="${PYTHON_BIN}" \
  BEST_CONFIGS="${BEST_CONFIGS_OUTPUT}" \
  OUTPUT_ROOT="${MULTISEED_OUTPUT_ROOT}" \
  SEEDS="${FINAL_SEEDS}" \
  GPU_IDS="${FINAL_GPU_IDS}" \
  MAX_EPOCHS="${FINAL_MAX_EPOCHS}" \
  PATIENCE="${FINAL_PATIENCE}" \
  NUM_WORKERS="${FINAL_NUM_WORKERS}" \
    bash scripts/run_attnres_multiseed.sh

  echo "[pipeline] Complete."
  echo "[pipeline] Summary: ${MULTISEED_OUTPUT_ROOT}/summary.csv"
fi
