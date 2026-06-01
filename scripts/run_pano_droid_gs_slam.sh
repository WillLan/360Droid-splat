#!/usr/bin/env bash
set -euo pipefail

python -m system.pano_droid_gs_slam --config "${1:-configs/pano_droid_gs_slam.yaml}"

