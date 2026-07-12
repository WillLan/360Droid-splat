#!/usr/bin/env bash
set -uo pipefail

ROOT="${STAGE3_ROOT:-/mnt/disk1/lanboyang/Project/360Droid-splat}"
PYTHON="${STAGE3_PYTHON:-/mnt/disk1/lanboyang/miniconda3/envs/pfgs360/bin/python}"
SUITE_DIR="${1:?usage: run_stage3_ba_eval_sweep.sh SUITE_DIR [GPU] [MAX_BATCHES] [START_BATCH]}"
GPU="${2:-6}"
MAX_BATCHES="${3:-32}"
START_BATCH="${4:-0}"

if [[ ! -f "$SUITE_DIR/manifest.json" ]]; then
  echo "Missing sweep manifest: $SUITE_DIR/manifest.json" >&2
  exit 2
fi

mapfile -t CONFIGS < <(find "$SUITE_DIR/configs" -maxdepth 1 -type f -name '*.yaml' -print | sort)
for config in "${CONFIGS[@]}"; do
  name="$(basename "$config" .yaml)"
  output="$SUITE_DIR/results/$name.json"
  log="$SUITE_DIR/results/$name.log"
  echo "[$(date --iso-8601=seconds)] evaluating $name on physical GPU $GPU"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT" \
    "$PYTHON" -u "$ROOT/tools/evaluate_stage3_ba.py" \
      --config "$config" \
      --max-batches "$MAX_BATCHES" \
      --start-batch "$START_BATCH" \
      --output "$output" 2>&1 | tee "$log"
  status=${PIPESTATUS[0]}
  echo "[$(date --iso-8601=seconds)] finished $name with exit code $status"
  if [[ "$status" -ne 0 ]]; then
    exit "$status"
  fi
done
