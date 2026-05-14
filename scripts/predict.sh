#!/usr/bin/env bash
# Run a trained detector on an image or directory of images.
#
# Usage:
#   scripts/predict.sh <source-image-or-dir> [run-name]
set -euo pipefail

cd "$(dirname "$0")/.."

SOURCE="${1:?usage: scripts/predict.sh <source> [run-name]}"
RUN_NAME="${2:-chars_v1}"

export YOLO_OFFLINE=1
export YOLO_AUTOINSTALL=false

python -m src.infer.predict \
    --weights   "runs/train/${RUN_NAME}/weights/best.pt" \
    --class-map configs/dataset/class_map.json \
    --source    "${SOURCE}" \
    --out       "runs/predict/${RUN_NAME}" \
    --viz
