#!/usr/bin/env bash
set -euo pipefail

python -m frontend.pano_droid.train --config "${1:-configs/pano_droid_train.yaml}"

