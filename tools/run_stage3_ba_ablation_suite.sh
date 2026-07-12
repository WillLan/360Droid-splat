#!/usr/bin/env bash
set -uo pipefail

ROOT="${STAGE3_ROOT:-/mnt/disk1/lanboyang/Project/360Droid-splat}"
PYTHON="${STAGE3_PYTHON:-/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python}"
SUITE_DIR="${1:?usage: run_stage3_ba_ablation_suite.sh SUITE_DIR [GPU_LIST] [MASTER_PORT]}"
GPU_LIST="${2:-6,7}"
MASTER_PORT="${3:-29667}"

if [[ ! -f "$SUITE_DIR/manifest.json" ]]; then
  echo "Missing ablation manifest: $SUITE_DIR/manifest.json" >&2
  exit 2
fi

mapfile -t CONFIGS < <(find "$SUITE_DIR/configs" -maxdepth 1 -type f -name '*.yaml' -print | sort)
if [[ "${#CONFIGS[@]}" -eq 0 ]]; then
  echo "No generated YAML configurations found in $SUITE_DIR/configs." >&2
  exit 2
fi

cd "$ROOT" || exit 2
for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .yaml)"
  log="$SUITE_DIR/${name}.launch.log"
  echo "[$(date --iso-8601=seconds)] starting $name on physical GPUs $GPU_LIST"
  CUDA_VISIBLE_DEVICES="$GPU_LIST" PYTHONPATH="$ROOT" \
    "$PYTHON" -m torch.distributed.run \
      --nproc_per_node=2 \
      --master_addr=127.0.0.1 \
      --master_port="$MASTER_PORT" \
      training/train_spherical_ba_recurrent_refiner.py \
      --config "$config" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  echo "[$(date --iso-8601=seconds)] finished $name with exit code $status"
  if [[ "$status" -ne 0 ]]; then
    exit "$status"
  fi
done
