#!/usr/bin/env bash

set -uo pipefail

GPU_ID="${1:?usage: $0 GPU_ID [QUEUE_ID]}"
QUEUE_ID="${2:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python"
MIN_AVAILABLE_KB=$((80 * 1024 * 1024))

CONFIGS=(
  "configs/boundary_globalba25_stride4_margin0_abd6213_norefiner.yaml"
  "configs/boundary_globalba100_localba8_pose2e4_umeyama5_abd6213_norefiner.yaml"
)
MAX_FRAMES=(25 100)
RUN_PREFIXES=(
  "boundary_globalba25_stride4_margin0_abd6213_norefiner_gpu${GPU_ID}_force_seq"
  "boundary_globalba100_localba8_pose2e4_umeyama5_abd6213_norefiner_gpu${GPU_ID}_force_seq"
)

QUEUE_DIR="${PROJECT_ROOT}/outputs/boundary_abd6213_force_queue_${QUEUE_ID}"
QUEUE_LOG="${QUEUE_DIR}/queue.log"

mkdir -p "${QUEUE_DIR}"

log_queue() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${QUEUE_LOG}"
}

read_swap_counters() {
  awk '
    $1 == "pswpin" { input = $2 }
    $1 == "pswpout" { output = $2 }
    END { print input + 0, output + 0 }
  ' /proc/vmstat
}

stop_project_group() {
  local pid="$1"
  local pgid="$2"
  log_queue "event=resource_stop_signal signal=SIGINT pid=${pid} pgid=${pgid}"
  kill -INT -- "-${pgid}" 2>/dev/null || true
  for _ in $(seq 1 10); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  log_queue "event=resource_stop_signal signal=SIGTERM pid=${pid} pgid=${pgid}"
  kill -TERM -- "-${pgid}" 2>/dev/null || true
}

prepare_resolved_config() {
  local source_config="$1"
  local output_dir="$2"
  local run_id="$3"
  local wandb_mode="$4"

  if [[ -e "${output_dir}" ]]; then
    log_queue "event=refuse_overwrite output_dir=${output_dir}"
    return 1
  fi
  mkdir -p "${output_dir}"

  "${PYTHON_BIN}" - "${PROJECT_ROOT}" "${source_config}" "${output_dir}" "${run_id}" "${wandb_mode}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from system.pano_droid_gs_slam import load_config

source = (root / sys.argv[2]).resolve()
output_dir = Path(sys.argv[3]).resolve()
run_id = sys.argv[4]
wandb_mode = sys.argv[5]

config = load_config(source)
config.setdefault("Results", {})["save_dir"] = str(output_dir)
config.setdefault("WeightsAndBiases", {}).update(
    {
        "enabled": True,
        "mode": wandb_mode,
        "run_name": run_id,
    }
)
config.setdefault("Visualization", {})["enabled"] = True

with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY
}

run_one() {
  local source_config="$1"
  local max_frames="$2"
  local run_id="$3"
  local output_dir="${PROJECT_ROOT}/outputs/${run_id}"
  local resource_log="${output_dir}/resource_monitor.log"
  local run_log="${output_dir}/run.log"
  local resource_stop=0
  local swap_growth_streak=0
  local previous_swap_in previous_swap_out
  local pid pgid status

  prepare_resolved_config "${source_config}" "${output_dir}" "${run_id}" "${WANDB_MODE}" || return 1
  read -r previous_swap_in previous_swap_out < <(read_swap_counters)

  log_queue "event=run_start run_id=${run_id} gpu=${GPU_ID} max_frames=${max_frames}"
  {
    printf '%s event=launcher_start run_id=%s gpu=%s commit=%s wandb_mode=%s\n' \
      "$(date --iso-8601=seconds)" "${run_id}" "${GPU_ID}" \
      "$(git -C "${PROJECT_ROOT}" rev-parse HEAD)" "${WANDB_MODE}"
  } >> "${resource_log}"

  (
    cd "${PROJECT_ROOT}" || exit 1
    exec setsid env \
      CUDA_DEVICE_ORDER=PCI_BUS_ID \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      OMP_NUM_THREADS=2 \
      MKL_NUM_THREADS=2 \
      OPENBLAS_NUM_THREADS=2 \
      NUMEXPR_NUM_THREADS=2 \
      PYTHONUNBUFFERED=1 \
      "${PYTHON_BIN}" -m system.pano_droid_gs_slam \
      --config "${output_dir}/resolved_config.json" \
      --max-frames "${max_frames}" \
      --wandb \
      --wandb-mode "${WANDB_MODE}" \
      --run-name "${run_id}"
  ) > "${run_log}" 2>&1 &
  pid=$!
  printf '%s\n' "${pid}" > "${output_dir}/pid"

  pgid=""
  for _ in $(seq 1 20); do
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ')"
    [[ -n "${pgid}" ]] && break
    sleep 0.1
  done
  if [[ -z "${pgid}" ]]; then
    pgid="${pid}"
  fi
  printf '%s\n' "${pgid}" > "${output_dir}/pgid"
  log_queue "event=process_start run_id=${run_id} pid=${pid} pgid=${pgid}"

  while kill -0 "${pid}" 2>/dev/null; do
    sleep 15
    kill -0 "${pid}" 2>/dev/null || break

    available_kb="$(awk '$1 == "MemAvailable:" { print $2 }' /proc/meminfo)"
    read -r current_swap_in current_swap_out < <(read_swap_counters)
    if (( current_swap_in > previous_swap_in || current_swap_out > previous_swap_out )); then
      swap_growth_streak=$((swap_growth_streak + 1))
    else
      swap_growth_streak=0
    fi
    previous_swap_in="${current_swap_in}"
    previous_swap_out="${current_swap_out}"

    gpu_sample="$(
      nvidia-smi -i "${GPU_ID}" \
        --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
        --format=csv,noheader,nounits 2>/dev/null || printf 'n/a'
    )"
    printf '%s pid=%s available_kb=%s pswpin=%s pswpout=%s swap_growth_streak=%s gpu=%s sample="%s"\n' \
      "$(date --iso-8601=seconds)" "${pid}" "${available_kb}" \
      "${current_swap_in}" "${current_swap_out}" "${swap_growth_streak}" \
      "${GPU_ID}" "${gpu_sample}" >> "${resource_log}"

    if (( available_kb < MIN_AVAILABLE_KB || swap_growth_streak >= 2 )); then
      resource_stop=1
      stop_project_group "${pid}" "${pgid}"
      break
    fi
  done

  wait "${pid}"
  status=$?
  printf '%s\n' "${status}" > "${output_dir}/exit_status"
  printf '%s\n' "${resource_stop}" > "${output_dir}/resource_stop"
  printf '%s event=process_exit status=%s resource_stop=%s\n' \
    "$(date --iso-8601=seconds)" "${status}" "${resource_stop}" >> "${resource_log}"
  log_queue "event=run_exit run_id=${run_id} status=${status} resource_stop=${resource_stop}"

  if (( resource_stop != 0 )); then
    return 125
  fi
  return "${status}"
}

if curl --head --silent --max-time 5 https://api.wandb.ai/ >/dev/null 2>&1; then
  WANDB_MODE=online
else
  WANDB_MODE=offline
fi

log_queue "event=queue_start queue_id=${QUEUE_ID} gpu=${GPU_ID} wandb_mode=${WANDB_MODE}"
free -h >> "${QUEUE_LOG}"
nvidia-smi -i "${GPU_ID}" \
  --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader >> "${QUEUE_LOG}" 2>&1 || true
nvidia-smi \
  --query-compute-apps=gpu_uuid,pid,used_memory,process_name \
  --format=csv,noheader >> "${QUEUE_LOG}" 2>&1 || true

for index in "${!CONFIGS[@]}"; do
  run_id="${RUN_PREFIXES[$index]}_${QUEUE_ID}"
  run_one "${CONFIGS[$index]}" "${MAX_FRAMES[$index]}" "${run_id}"
  status=$?
  if (( status != 0 )); then
    log_queue "event=queue_abort failed_run=${run_id} status=${status}"
    exit "${status}"
  fi
done

log_queue "event=queue_complete queue_id=${QUEUE_ID}"
