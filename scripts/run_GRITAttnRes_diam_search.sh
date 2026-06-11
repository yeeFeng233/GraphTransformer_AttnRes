#!/usr/bin/env bash
set -euo pipefail

# Generic ECHO-Synth GRIT+AttnRes search launcher.
# Task-specific wrappers set TASK_NAME and call this script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
cd "${PROJECT_DIR}"

TASK_NAME="${TASK_NAME:-diam}"
MODEL_NAME="${MODEL_NAME:-grit_attnres_gp}"
MAX_VISIBLE_GPUS="${MAX_VISIBLE_GPUS:-2}"
DETACH=0
ARGS=()

for arg in "$@"; do
  if [ "$arg" = "--detach" ]; then
    DETACH=1
  else
    ARGS+=("$arg")
  fi
done

case "${TASK_NAME}" in
  diam|ecc|sssp) ;;
  *)
    echo "Unsupported ECHO-Synth task: ${TASK_NAME}. Expected one of: diam, ecc, sssp." >&2
    exit 1
    ;;
esac

CONFIG_FILE="${PROJECT_DIR}/search-space/${TASK_NAME}/${MODEL_NAME}.yaml"
if [ ! -f "${CONFIG_FILE}" ]; then
  echo "Search-space config not found: ${CONFIG_FILE}" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-${PROJECT_DIR}/logs/grit_attnres_${TASK_NAME}_search}"
mkdir -p "${LOG_DIR}"

timestamp="$(date +%Y%m%d_%H%M%S)"
launcher_log="${LOG_DIR}/launcher_${timestamp}.log"
pid_file="${LOG_DIR}/launcher_${timestamp}.pid"
run_log="${LOG_DIR}/search_${timestamp}.log"

if [ "${DETACH}" -eq 1 ]; then
  echo "Starting GRIT+AttnRes ${TASK_NAME} search in detached mode..."
  nohup bash "${SCRIPT_PATH}" "${ARGS[@]}" > "${launcher_log}" 2>&1 &
  bg_pid=$!
  echo "${bg_pid}" > "${pid_file}"
  echo "Detached successfully."
  echo "PID: ${bg_pid}"
  echo "Launcher log: ${launcher_log}"
  echo "PID file: ${pid_file}"
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; cannot determine available GPUs." >&2
  exit 1
fi

GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader | head -n "${MAX_VISIBLE_GPUS}" | paste -sd, -)"
if [ -z "${GPU_LIST}" ]; then
  echo "No GPUs detected by nvidia-smi." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_LIST}}"
GPU_COUNT="$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | awk 'NF {count++} END {print count+0}')"
HOST_CPU_COUNT="$(nproc)"

CONDA_ROOT="${CONDA_ROOT:-/HOME/nsccgz_ywang/nsccgz_ywang_wzh/HDD_POOL/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-lyf_echo}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ROOT}/envs/${CONDA_ENV_NAME}/bin/python}"
SEARCH_SCRIPT="${PROJECT_DIR}/scripts/search.py"
DOWNLOAD_SCRIPT="${PROJECT_DIR}/scripts/download-all.py"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "Python interpreter not found: ${PYTHON_BIN}" >&2
  echo "Set PYTHON_BIN directly, or set CONDA_ROOT/CONDA_ENV_NAME before running this script." >&2
  exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

N_SAMPLES="${N_SAMPLES:-32}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
REPORT_EVERY_N_EPOCHS="${REPORT_EVERY_N_EPOCHS:-1}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-4}"
GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/results/grit_attnres_${TASK_NAME}}"
STORAGE_PATH="${STORAGE_PATH:-${PROJECT_DIR}/ray_results/grit_attnres_${TASK_NAME}}"
RAY_TEMP_TARGET="${RAY_TEMP_TARGET:-${PROJECT_DIR}/ray_tmp/grit_attnres_${TASK_NAME}}"
# Ray creates UNIX sockets under the visible temp path; keep it short.
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/echo_ray_gritar_${TASK_NAME}_${USER:-user}}"
TRAINING_LOG_DIR="${TRAINING_LOG_DIR:-${LOG_DIR}/training_logs}"
SCHEDULER="${SCHEDULER:-none}"
SEARCH_ALG="${SEARCH_ALG:-optuna}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-80}"
ASHA_GRACE_PERIOD="${ASHA_GRACE_PERIOD:-60}"
MONITOR_METRIC="${MONITOR_METRIC:-val_mae}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-1}"
DRY_RUN="${DRY_RUN:-0}"

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

mkdir -p "${OUTPUT_DIR}" "${STORAGE_PATH}" "${RAY_TEMP_TARGET}" "${TRAINING_LOG_DIR}"
if [ -L "${RAY_TEMP_DIR}" ]; then
  ln -sfn "${RAY_TEMP_TARGET}" "${RAY_TEMP_DIR}"
elif [ -e "${RAY_TEMP_DIR}" ]; then
  echo "Ray temp path exists and is not a symlink; using it directly: ${RAY_TEMP_DIR}" | tee -a "${run_log}"
else
  ln -s "${RAY_TEMP_TARGET}" "${RAY_TEMP_DIR}"
fi

echo "============================================================" | tee "${run_log}"
echo "[$(date '+%F %T')] Running GRIT+AttnRes ${TASK_NAME} hyperparameter search" | tee -a "${run_log}"
echo "Project dir: ${PROJECT_DIR}" | tee -a "${run_log}"
echo "Task: ${TASK_NAME}" | tee -a "${run_log}"
echo "Model search-space: ${MODEL_NAME}" | tee -a "${run_log}"
echo "Config file: ${CONFIG_FILE}" | tee -a "${run_log}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}" | tee -a "${run_log}"
echo "MAX_VISIBLE_GPUS=${MAX_VISIBLE_GPUS}" | tee -a "${run_log}"
echo "N_SAMPLES=${N_SAMPLES}, MAX_EPOCHS=${MAX_EPOCHS}, REPORT_EVERY_N_EPOCHS=${REPORT_EVERY_N_EPOCHS}" | tee -a "${run_log}"
echo "SCHEDULER=${SCHEDULER}, SEARCH_ALG=${SEARCH_ALG}, EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}, ASHA_GRACE_PERIOD=${ASHA_GRACE_PERIOD}" | tee -a "${run_log}"
echo "MONITOR_METRIC=${MONITOR_METRIC}" | tee -a "${run_log}"
echo "Host CPUs=${HOST_CPU_COUNT}, GPUs=${GPU_COUNT}, CPUs/trial=${CPUS_PER_TRIAL}, GPUs/trial=${GPUS_PER_TRIAL}" | tee -a "${run_log}"
echo "Effective concurrent trials=${EFFECTIVE_PARALLEL_TRIALS}" | tee -a "${run_log}"
echo "PyTorch CUDA alloc conf: ${PYTORCH_CUDA_ALLOC_CONF}" | tee -a "${run_log}"
echo "Python: ${PYTHON_BIN}" | tee -a "${run_log}"
echo "Log dir: ${LOG_DIR}" | tee -a "${run_log}"
echo "Training log dir: ${TRAINING_LOG_DIR}" | tee -a "${run_log}"
echo "Output dir: ${OUTPUT_DIR}" | tee -a "${run_log}"
echo "Ray Tune storage: ${STORAGE_PATH}" | tee -a "${run_log}"
echo "Ray temp dir: ${RAY_TEMP_DIR}" | tee -a "${run_log}"
echo "Ray temp target: ${RAY_TEMP_TARGET}" | tee -a "${run_log}"
echo "Results: live_search_${TASK_NAME}.csv updates during the run; search_${TASK_NAME}.csv is written after completion." | tee -a "${run_log}"
echo "============================================================" | tee -a "${run_log}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "DRY_RUN=1; command validation completed without launching search." | tee -a "${run_log}"
  exit 0
fi

if [ "${DOWNLOAD_DATASET}" = "1" ]; then
  echo "[$(date '+%F %T')] Ensuring dataset is ready for task=${TASK_NAME}" | tee -a "${run_log}"
  "${PYTHON_BIN}" "${DOWNLOAD_SCRIPT}" --root ./data --task "${TASK_NAME}" 2>&1 | tee -a "${run_log}"
else
  echo "DOWNLOAD_DATASET=0; skipping dataset preparation." | tee -a "${run_log}"
fi

"${PYTHON_BIN}" "${SEARCH_SCRIPT}" \
  --tasks "${TASK_NAME}" \
  --models "${MODEL_NAME}" \
  --n_samples "${N_SAMPLES}" \
  --scheduler "${SCHEDULER}" \
  --search_alg "${SEARCH_ALG}" \
  --logger csv \
  --max_epochs "${MAX_EPOCHS}" \
  --report_every_n_epochs "${REPORT_EVERY_N_EPOCHS}" \
  --early_stopping_patience "${EARLY_STOPPING_PATIENCE}" \
  --monitor_metric "${MONITOR_METRIC}" \
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

echo "[$(date '+%F %T')] GRIT+AttnRes ${TASK_NAME} search completed." | tee -a "${run_log}"
