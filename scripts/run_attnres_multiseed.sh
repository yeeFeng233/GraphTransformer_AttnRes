#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-43 44 45 46 47}"
GPU_IDS="${GPU_IDS:-0 1 2 3 4}"
RUNS="${RUNS:-gps_diam gps_ecc gps_sssp gps_charge gps_energy grit_diam grit_ecc grit_sssp grit_charge grit_energy}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/multiseed}"
MAX_EPOCHS="${MAX_EPOCHS:-1000}"
PATIENCE="${PATIENCE:-80}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

run_one () {
  local run_name="$1"
  local seed="$2"
  local gpu_id="$3"
  local model task layers hidden batch lr wd dropout heads attn_dropout stride

  case "${run_name}" in
    gps_charge)
      model=GPSAttnRes; task=charge; layers=28; hidden=192; batch=256
      lr=9.286961387808238e-05; wd=2.623473114072005e-05; dropout=0.0
      heads=8; attn_dropout=0.05; stride=2
      ;;
    gps_energy)
      model=GPSAttnRes; task=energy; layers=26; hidden=192; batch=512
      lr=4e-05; wd=0.0003; dropout=0.0
      heads=4; attn_dropout=0.0; stride=4
      ;;
    gps_diam)
      model=GPSAttnRes; task=diam; layers=21; hidden=80; batch=256
      lr=7.027506842162453e-05; wd=0.00018881921338068878; dropout=0.0
      heads=2; attn_dropout=0.03; stride=8
      ;;
    gps_ecc)
      model=GPSAttnRes; task=ecc; layers=19; hidden=192; batch=256
      lr=0.00028926866413819184; wd=5.4850931995345765e-05; dropout=0.05
      heads=2; attn_dropout=0.05; stride=4
      ;;
    gps_sssp)
      model=GPSAttnRes; task=sssp; layers=28; hidden=64; batch=256
      lr=0.0001308642441676819; wd=0.00015241115488340732; dropout=0.03
      heads=2; attn_dropout=0.03; stride=4
      ;;
    grit_diam)
      model=GRITAttnRes; task=diam; layers=24; hidden=192; batch=128
      lr=0.0006787077591940332; wd=0.0002445340977989499; dropout=0.1
      heads=4; attn_dropout=0.433; stride=
      ;;
    grit_ecc)
      model=GRITAttnRes; task=ecc; layers=24; hidden=192; batch=128
      lr=0.00025; wd=8e-05; dropout=0.0
      heads=2; attn_dropout=0.05; stride=
      ;;
    grit_sssp)
      model=GRITAttnRes; task=sssp; layers=44; hidden=96; batch=224
      lr=0.00028; wd=0.00015; dropout=0.05
      heads=4; attn_dropout=0.1; stride=
      ;;
    grit_charge)
      model=GRITAttnRes; task=charge; layers=32; hidden=128; batch=128
      lr=0.00034; wd=0.00034; dropout=0.0
      heads=2; attn_dropout=0.351; stride=
      ;;
    grit_energy)
      model=GRITAttnRes; task=energy; layers=12; hidden=64; batch=128
      lr=0.00076; wd=0.00094; dropout=0.0
      heads=2; attn_dropout=0.178; stride=
      ;;
    *)
      echo "Unknown run: ${run_name}" >&2
      exit 2
      ;;
  esac

  local output_dir="${OUTPUT_ROOT}/${model}/${task}"
  local result_json="${output_dir}/seed_${seed}.json"
  local log_file="${output_dir}/seed_${seed}.log"
  mkdir -p "${output_dir}"

  if [ "${SKIP_EXISTING}" = "1" ] && [ -s "${result_json}" ]; then
    echo "[$(date '+%F %T')] Skip existing ${model}/${task} seed=${seed}"
    return 0
  fi

  local model_args=()
  if [ "${model}" = "GPSAttnRes" ]; then
    model_args=(
      --gps_num_heads "${heads}"
      --gps_attn_dropout "${attn_dropout}"
      --gps_attn_type multihead
      --attnres_history_stride "${stride}"
    )
  else
    model_args=(
      --grit_num_heads "${heads}"
      --grit_attn_dropout "${attn_dropout}"
      --grit_act relu
    )
  fi

  echo "[$(date '+%F %T')] ${model}/${task} seed=${seed} gpu=${gpu_id}"
  CUDA_VISIBLE_DEVICES="${gpu_id}" "${PYTHON_BIN}" scripts/train.py \
    --task "${task}" \
    --gnn_type GNN \
    --conv_layer "${model}" \
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
  echo "============================================================"
  echo "Starting five-seed group: ${run_name}"
  echo "============================================================"

  pids=()
  labels=()
  for index in "${!seed_array[@]}"; do
    seed="${seed_array[$index]}"
    gpu_id="${gpu_array[$index]}"
    run_one "${run_name}" "${seed}" "${gpu_id}" &
    pids+=("$!")
    labels+=("${run_name}/seed_${seed}/gpu_${gpu_id}")
  done

  failed=0
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      echo "FAILED: ${labels[$index]}" >&2
      failed=1
    fi
  done

  if [ "${failed}" -ne 0 ]; then
    echo "Stopping because at least one seed failed for ${run_name}." >&2
    exit 1
  fi
done

"${PYTHON_BIN}" scripts/summarize_multiseed.py \
  --input_dir "${OUTPUT_ROOT}" \
  --output_csv "${OUTPUT_ROOT}/summary.csv"
