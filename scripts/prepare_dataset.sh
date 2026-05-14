#!/usr/bin/env bash
# Build the YOLO dataset from the raw annotations under data/raw/.
set -euo pipefail

cd "$(dirname "$0")/.."

python -m src.dataset.build_yolo_dataset \
    --raw         data/raw \
    --out         data/yolo \
    --class-map   configs/dataset/class_map.json \
    --dataset-yaml configs/dataset/chars.yaml \
    --split       0.8 0.1 0.1

python -m src.dataset.verify \
    --root         data/yolo \
    --dataset-yaml configs/dataset/chars.yaml
