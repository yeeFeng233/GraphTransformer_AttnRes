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

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/gps_attnres_energy_search_part1}"
mkdir -p "$LOG_DIR"

timestamp="$(date +%Y%m%d_%H%M%S)"
launcher_log="${LOG_DIR}/launcher_${timestamp}.log"
pid_file="${LOG_DIR}/launcher_${timestamp}.pid"
run_log="${LOG_DIR}/search_${timestamp}.log"

if [ "$DETACH" -eq 1 ]; then
  echo "Starting GPSAttnRes energy search in detached mode..."
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
HOST_CPU_COUNT="$(nproc)"

CONDA_ROOT="${CONDA_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lyf_echo}"
PYTHON_BIN="${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/python"
SEARCH_SCRIPT="scripts/search.py"
DOWNLOAD_SCRIPT="scripts/download-all.py"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set CONDA_ROOT or CONDA_ENV_NAME before running this script." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N_SAMPLES="${N_SAMPLES:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
REPORT_EVERY_N_EPOCHS="${REPORT_EVERY_N_EPOCHS:-1}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-4}"
GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/gps_attnres_energy_part1}"
STORAGE_PATH="${STORAGE_PATH:-${PROJECT_DIR}/ray_results/gps_attnres_energy_part1}"
RAY_TEMP_TARGET="${RAY_TEMP_TARGET:-${PROJECT_DIR}/ray_tmp/gps_attnres_energy_part1}"
# Ray creates UNIX sockets under the temp dir; this visible path must stay short.
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/echo_ray_energy_part1_${USER:-user}}"
TASKS=(energy)
TRAINING_LOG_DIR="${TRAINING_LOG_DIR:-${LOG_DIR}/training_logs}"
MODELS="${MODELS:-gps_attnres_gp_part1}"
SCHEDULER="${SCHEDULER:-none}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-100}"
ASHA_GRACE_PERIOD="${ASHA_GRACE_PERIOD:-60}"

if [ "${GPUS_PER_TRIAL}" -lt 1 ]; then
  echo "GPUS_PER_TRIAL must be at least 1." >&2
  exit 1
fi

if [ "${GPU_COUNT}" -lt "${GPUS_PER_TRIAL}" ]; then
  echo "Visible GPUs (${GPU_COUNT}) are fewer than GPUS_PER_TRIAL (${GPUS_PER_TRIAL})." >&2
  exit 1
fi

GPU_PARALLEL_TRIALS=$(( GPU_COUNT / GPUS_PER_TRIAL ))
CPU_PARALLEL_TRIALS=$(( HOST_CPU_COUNT / CPUS_PER_TRIAL ))
if [ "${CPU_PARALLEL_TRIALS}" -lt 1 ]; then
  echo "Computed CPU-limited trial count is zero. Lower CPUS_PER_TRIAL." >&2
  exit 1
fi

if [ "${GPU_PARALLEL_TRIALS}" -lt "${CPU_PARALLEL_TRIALS}" ]; then
  EFFECTIVE_PARALLEL_TRIALS="${GPU_PARALLEL_TRIALS}"
else
  EFFECTIVE_PARALLEL_TRIALS="${CPU_PARALLEL_TRIALS}"
fi

if [ "${EFFECTIVE_PARALLEL_TRIALS}" -lt 1 ]; then
  echo "Computed parallel trial count is zero. Check GPU/CPU settings." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${STORAGE_PATH}"
mkdir -p "${RAY_TEMP_TARGET}"
if [ -L "${RAY_TEMP_DIR}" ]; then
  ln -sfn "${RAY_TEMP_TARGET}" "${RAY_TEMP_DIR}"
elif [ -e "${RAY_TEMP_DIR}" ]; then
  echo "Ray temp path exists and is not a symlink; using it directly: ${RAY_TEMP_DIR}" | tee -a "${run_log}"
else
  ln -s "${RAY_TEMP_TARGET}" "${RAY_TEMP_DIR}"
fi

echo "============================================================" | tee "${run_log}"
echo "[$(date '+%F %T')] Running GPSAttnRes energy-only hyperparameter search" | tee -a "${run_log}"
echo "Tasks: ${TASKS[*]}" | tee -a "${run_log}"
echo "Models: ${MODELS}" | tee -a "${run_log}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${run_log}"
echo "N_SAMPLES=${N_SAMPLES}, MAX_EPOCHS=${MAX_EPOCHS}, REPORT_EVERY_N_EPOCHS=${REPORT_EVERY_N_EPOCHS}" | tee -a "${run_log}"
echo "SCHEDULER=${SCHEDULER}, EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}, ASHA_GRACE_PERIOD=${ASHA_GRACE_PERIOD}" | tee -a "${run_log}"
echo "Host CPUs=${HOST_CPU_COUNT}, GPUs=${GPU_COUNT}, CPUs/trial=${CPUS_PER_TRIAL}, GPUs/trial=${GPUS_PER_TRIAL}" | tee -a "${run_log}"
echo "Effective concurrent trials=${EFFECTIVE_PARALLEL_TRIALS}" | tee -a "${run_log}"
echo "PyTorch CUDA alloc conf: ${PYTORCH_CUDA_ALLOC_CONF}" | tee -a "${run_log}"
echo "Log dir: ${LOG_DIR}" | tee -a "${run_log}"
echo "Training log dir: ${TRAINING_LOG_DIR}" | tee -a "${run_log}"
echo "Output dir: ${OUTPUT_DIR}" | tee -a "${run_log}"
echo "Ray Tune storage: ${STORAGE_PATH}" | tee -a "${run_log}"
echo "Ray temp dir: ${RAY_TEMP_DIR}" | tee -a "${run_log}"
echo "Ray temp target: ${RAY_TEMP_TARGET}" | tee -a "${run_log}"
echo "Energy search part1: patience-driven long training; batch=512; see search-space/energy/gps_attnres_gp_part1.yaml" | tee -a "${run_log}"
echo "Initial Optuna points are ordered from the largest parameter settings first." | tee -a "${run_log}"
echo "Results: ${OUTPUT_DIR}/live_search_energy.csv updates during the run; search_energy.csv is written after the task finishes" | tee -a "${run_log}"
echo "============================================================" | tee -a "${run_log}"

for task in "${TASKS[@]}"; do
  echo "[$(date '+%F %T')] Ensuring dataset is ready for task=${task}" | tee -a "${run_log}"
  "${PYTHON_BIN}" "${DOWNLOAD_SCRIPT}" --root ./data --task "${task}" 2>&1 | tee -a "${run_log}"
done

"${PYTHON_BIN}" "${SEARCH_SCRIPT}" \
  --tasks "${TASKS[@]}" \
  --models ${MODELS} \
  --n_samples "${N_SAMPLES}" \
  --scheduler "${SCHEDULER}" \
  --search_alg optuna \
  --logger csv \
  --max_epochs "${MAX_EPOCHS}" \
  --report_every_n_epochs "${REPORT_EVERY_N_EPOCHS}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --asha_grace_period "${ASHA_GRACE_PERIOD}" \
  --num_cpus "${HOST_CPU_COUNT}" \
  --num_gpus "${GPU_COUNT}" \
  --gpus_per_trial "${GPUS_PER_TRIAL}" \
  --cpus_per_trial "${CPUS_PER_TRIAL}" \
  --num_workers "${NUM_WORKERS}" \
  --output_dir "${OUTPUT_DIR}" \
  --training_log_dir "${TRAINING_LOG_DIR}" \
  --storage_path "${STORAGE_PATH}" \
  --ray_temp_dir "${RAY_TEMP_DIR}" \
  2>&1 | tee -a "${run_log}"

echo "[$(date '+%F %T')] GPSAttnRes energy-only hyperparameter search completed." | tee -a "${run_log}"
