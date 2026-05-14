#!/usr/bin/env bash
# Train a YOLOv8n character detector from scratch (offline-safe).
set -euo pipefail

cd "$(dirname "$0")/.."

export YOLO_OFFLINE=1
export YOLO_AUTOINSTALL=false

python -m src.train.train \
    --mode    train \
    --model   configs/model/yolov8n.yaml \
    --data    configs/dataset/chars.yaml \
    --epochs  "${EPOCHS:-300}" \
    --imgsz   "${IMGSZ:-640}" \
    --batch   "${BATCH:-16}" \
    --project runs/train \
    --name    "${RUN_NAME:-chars_v1}"
