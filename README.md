# 1_detect_image_yolo

A YOLO-based image-detection pipeline that extracts **character regions** from text images
and, for each region, recovers two attributes:

1. The **Unicode codepoint** of the character.
2. The **font type** (font family / style) the character is rendered in.

The project is designed to run **fully offline**: the host machine has the YOLO Python
framework installed but does **not** have access to pre-trained YOLO weights. Training
must therefore start from a randomly initialised model defined by a YAML architecture
file (e.g. `yolov8n.yaml`) and never from a `.pt` checkpoint downloaded from the internet.

---

## 1. Project Goals

- Localise every printed character in an image with a tight bounding box.
- Assign each box a class label that uniquely encodes `(unicode_codepoint, font_type)`.
- Provide a reproducible dataset-preparation step that converts raw annotations
  into the YOLO label format.
- Train, evaluate, and run inference without any network access.

---

## 2. Build Guide

End-to-end checklist for going from a fresh clone to a trained detector and a
working inference run. All steps are designed to succeed with no network access.

### 2.1. Prerequisites

| Requirement       | Notes                                                         |
|-------------------|---------------------------------------------------------------|
| Python 3.10+      | Tested against 3.10 – 3.12.                                   |
| pip               | With access to an **offline wheel mirror** (no PyPI required).|
| GPU (optional)    | Training on CPU works but is much slower.                     |

The project does not require any pre-downloaded YOLO `.pt` weights — see §3 below.

### 2.2. Install

```bash
cd 1_detect_image_yolo
python -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links=<local-wheel-dir> -r requirements.txt
```

`--no-index --find-links=<dir>` guarantees pip never reaches the internet. Drop the
flag only on a host where outbound traffic is acceptable.

### 2.3. Verify the install

```bash
python -m pytest -q
```

You should see all unit tests pass. These exercise the label/class-map logic, the
schema validator, the split assignment, and the dataset builder — no GPU or
network needed.

### 2.4. Drop in raw data

Place your inputs under:

```
data/raw/images/         # *.png / *.jpg
data/raw/annotations/    # one *.json per image — schema in §4.4
```

### 2.5. Build the YOLO dataset

```bash
bash scripts/prepare_dataset.sh
```

This will:

1. Discover every annotation and validate it.
2. Update `configs/dataset/class_map.json` (append-only — existing ids preserved).
3. Convert pixel bboxes to YOLO normalised format and write
   `data/yolo/{images,labels}/{train,val,test}/`.
4. Regenerate the `names:` block of `configs/dataset/chars.yaml`.
5. Run `src/dataset/verify.py` to sanity-check every label.

### 2.6. Train

```bash
bash scripts/train.sh
# or with overrides:
EPOCHS=500 IMGSZ=768 BATCH=8 RUN_NAME=chars_v2 bash scripts/train.sh
```

Outputs land in `runs/train/<RUN_NAME>/`. The best checkpoint is
`runs/train/<RUN_NAME>/weights/best.pt`.

### 2.7. Predict

```bash
bash scripts/predict.sh path/to/page.png chars_v1
```

Writes one `<stem>.json` (decoded detections) and one `<stem>.png`
(visualisation) per source image to `runs/predict/<RUN_NAME>/`.

### 2.8. Build checklist

- [ ] `pip install -r requirements.txt` succeeded offline.
- [ ] `pytest -q` is green.
- [ ] `data/raw/images/` and `data/raw/annotations/` are populated.
- [ ] `bash scripts/prepare_dataset.sh` printed `Dataset OK.`.
- [ ] `configs/dataset/class_map.json` and `configs/dataset/chars.yaml`'s
      `names:` block both reflect the expected classes.
- [ ] `bash scripts/train.sh` produced `runs/train/<run>/weights/best.pt`.
- [ ] `bash scripts/predict.sh` produced JSON + PNG outputs.

---

## 3. Project Architecture

```
1_detect_image_yolo/
├── README.md
├── requirements.txt
├── configs/
│   ├── model/
│   │   └── yolov8n.yaml          # model architecture (NO pretrained weights)
│   └── dataset/
│       └── chars.yaml            # Ultralytics dataset descriptor
├── data/
│   ├── raw/                      # source images + raw annotations (JSON/CSV)
│   │   ├── images/
│   │   └── annotations/
│   ├── interim/                  # cleaned / validated annotations
│   └── yolo/                     # final YOLO-format dataset
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── src/
│   ├── __init__.py
│   ├── labels/
│   │   ├── class_map.py          # (unicode, font) <-> class_id mapping
│   │   └── schema.py             # raw annotation schema + validators
│   ├── dataset/
│   │   ├── build_yolo_dataset.py # raw -> YOLO format converter
│   │   ├── split.py              # train/val/test split
│   │   └── verify.py             # sanity-check labels (bbox bounds, classes)
│   ├── train/
│   │   └── train.py              # training entry point (from-scratch only)
│   ├── infer/
│   │   ├── predict.py            # run trained model on new images
│   │   └── decode.py             # class_id -> (unicode, font) postprocess
│   └── utils/
│       ├── io.py
│       └── viz.py                # draw predictions for debugging
├── scripts/
│   ├── prepare_dataset.sh
│   ├── train.sh
│   └── predict.sh
├── runs/                         # Ultralytics training/inference outputs
└── tests/
    ├── test_class_map.py
    └── test_build_yolo_dataset.py
```

### Data flow

```
raw images + raw annotations
         │
         ▼
   src/dataset/build_yolo_dataset.py
   (uses src/labels/class_map.py to map (unicode, font) -> class_id)
         │
         ▼
   data/yolo/{images,labels}/{train,val,test}/
   configs/dataset/chars.yaml
         │
         ▼
   src/train/train.py  ──►  runs/train/<exp>/weights/best.pt
         │
         ▼
   src/infer/predict.py + src/infer/decode.py
         │
         ▼
   list of detections: [{bbox, unicode, font, confidence}, ...]
```

### Class encoding

Each YOLO class id corresponds to a pair `(unicode_codepoint, font_type)`. The mapping is
generated deterministically by `src/labels/class_map.py` and persisted as
`configs/dataset/class_map.json` so that:

- Training is reproducible (the same `(unicode, font)` always gets the same class id).
- Inference can decode `class_id` back to the human-meaningful tuple without ambiguity.

`class_map.json` schema:

```json
{
  "version": 1,
  "classes": [
    {"id": 0, "unicode": 65,  "char": "A", "font": "NotoSans-Regular"},
    {"id": 1, "unicode": 65,  "char": "A", "font": "NotoSans-Bold"},
    {"id": 2, "unicode": 97,  "char": "a", "font": "NotoSans-Regular"}
  ]
}
```

### YOLO label format

For every image `data/yolo/images/<split>/<name>.png` there is a sibling text file
`data/yolo/labels/<split>/<name>.txt`. Each line is one character instance:

```
<class_id> <x_center> <y_center> <width> <height>
```

All four box values are normalised to `[0, 1]` relative to image width/height.

### Dataset descriptor (`configs/dataset/chars.yaml`)

```yaml
path: ../../data/yolo        # root, relative to this file
train: images/train
val:   images/val
test:  images/test

# 'names' is generated from configs/dataset/class_map.json
# and written here by scripts/prepare_dataset.sh
names:
  0: "U+0041 NotoSans-Regular"
  1: "U+0041 NotoSans-Bold"
  2: "U+0061 NotoSans-Regular"
```

---

## 4. Library Usage

### Required libraries

| Library        | Purpose                                                   |
|----------------|-----------------------------------------------------------|
| `ultralytics`  | YOLO framework (model definition, training loop, predict) |
| `torch`        | Backend used by Ultralytics                               |
| `Pillow`       | Image I/O, conversion, size queries                       |
| `numpy`        | Array math for bbox normalisation                         |
| `PyYAML`       | Read/write dataset and model YAML configs                 |
| `pytest`       | Tests                                                     |

`requirements.txt` pins these to versions that exist in the offline package mirror.
No package may be installed from PyPI at runtime.

### Offline constraint — important

The standard Ultralytics tutorial uses code like:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")   # downloads weights from the internet — NOT ALLOWED HERE
```

In this project, training **must** start from a YAML model definition shipped with
Ultralytics, which initialises weights randomly:

```python
from ultralytics import YOLO
model = YOLO("configs/model/yolov8n.yaml")   # offline-safe, random init
model.train(
    data="configs/dataset/chars.yaml",
    epochs=300,
    imgsz=640,
    pretrained=False,                        # explicit: never fetch weights
    project="runs/train",
    name="chars_v1",
)
```

Because no transfer learning is possible, expect:

- More epochs (300+ instead of the typical 50–100).
- Stronger reliance on data augmentation (mosaic, mixup, hsv jitter — all built into
  Ultralytics, no extra downloads needed).
- A larger and more balanced dataset, especially across font types.

To keep Ultralytics from attempting network calls (font downloads for plots, update
checks), set these environment variables before running anything:

```bash
export YOLO_OFFLINE=1
export YOLO_AUTOINSTALL=false
```

### Inference

```python
from ultralytics import YOLO
from src.infer.decode import load_class_map, decode

model     = YOLO("runs/train/chars_v1/weights/best.pt")
class_map = load_class_map("configs/dataset/class_map.json")

results = model.predict(source="path/to/image.png", imgsz=640, conf=0.25)
for box in results[0].boxes:
    cls_id     = int(box.cls)
    unicode_cp, font = decode(class_map, cls_id)
    xyxy       = box.xyxy[0].tolist()
    print(unicode_cp, chr(unicode_cp), font, xyxy)
```

---

## 5. Developer Guide

### 4.1. Set up the environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --no-index --find-links=<local-wheel-dir> -r requirements.txt
```

`--no-index --find-links=<dir>` guarantees pip never reaches the internet.

### 4.2. Prepare raw annotations

Drop your data under `data/raw/`:

- `data/raw/images/*.png` — input images.
- `data/raw/annotations/*.json` — one JSON per image, schema:

```json
{
  "image": "0001.png",
  "width": 1280,
  "height": 720,
  "instances": [
    {"unicode": 65, "font": "NotoSans-Regular",
     "bbox_xyxy": [102, 48, 138, 96]},
    {"unicode": 97, "font": "NotoSans-Bold",
     "bbox_xyxy": [140, 50, 170, 95]}
  ]
}
```

`bbox_xyxy` is in absolute pixel coordinates `[x_min, y_min, x_max, y_max]`.

### 4.3. Build the YOLO dataset

```bash
bash scripts/prepare_dataset.sh
# or, equivalently:
python -m src.dataset.build_yolo_dataset \
    --raw data/raw \
    --out data/yolo \
    --class-map configs/dataset/class_map.json \
    --split 0.8 0.1 0.1
python -m src.dataset.verify --root data/yolo
```

This step:

1. Scans every raw annotation and registers every `(unicode, font)` pair encountered
   into `class_map.json` (existing entries are preserved — class ids are append-only).
2. Converts `bbox_xyxy` (pixels) to YOLO `(x_center, y_center, w, h)` normalised to
   `[0, 1]`.
3. Copies images and writes label `.txt` files into the train/val/test splits.
4. Regenerates the `names:` block of `configs/dataset/chars.yaml`.

> ⚠️ **Never delete `class_map.json` after training has started.** Doing so would
> renumber classes and silently invalidate the trained weights.

### 4.4. Train

```bash
YOLO_OFFLINE=1 YOLO_AUTOINSTALL=false \
python -m src.train.train \
    --model  configs/model/yolov8n.yaml \
    --data   configs/dataset/chars.yaml \
    --epochs 300 \
    --imgsz  640 \
    --name   chars_v1
```

Outputs land in `runs/train/chars_v1/`:

- `weights/best.pt`  — best mAP checkpoint
- `weights/last.pt`  — most recent checkpoint
- `results.csv`, plots, confusion matrix

### 4.5. Evaluate

```bash
python -m src.train.train \
    --mode   val \
    --weights runs/train/chars_v1/weights/best.pt \
    --data   configs/dataset/chars.yaml
```

Watch:

- **Per-class precision/recall** — strongly imbalanced classes (rare fonts) will
  show up here first.
- **mAP@50** and **mAP@50-95**.

### 4.6. Predict

```bash
python -m src.infer.predict \
    --weights   runs/train/chars_v1/weights/best.pt \
    --class-map configs/dataset/class_map.json \
    --source    data/samples/page_01.png \
    --out       runs/predict/page_01/
```

Produces:

- `page_01.json` — list of `{bbox, unicode, char, font, confidence}` records.
- `page_01.png`  — visualisation with boxes drawn (via `src/utils/viz.py`).

### 4.7. Testing

```bash
pytest -q
```

Tests should cover at minimum:

- `class_map.py`: append-only behaviour, idempotency, round-trip encode/decode.
- `build_yolo_dataset.py`: correct normalisation, bbox clipping, split ratios,
  rejection of malformed annotations.

### 4.8. Common pitfalls

| Symptom                                              | Likely cause                                                                 |
|------------------------------------------------------|------------------------------------------------------------------------------|
| Ultralytics tries to download `yolov8n.pt`           | Passed a `.pt` name instead of the `configs/model/*.yaml` path.              |
| Network error on first epoch                         | `YOLO_OFFLINE`/`YOLO_AUTOINSTALL` not set; fonts or update check attempted.  |
| Training loss does not decrease                      | Random init + too few epochs; raise epochs, check augmentations and lr.      |
| Class count mismatch between `chars.yaml` and model  | Stale `class_map.json` — rerun `prepare_dataset` and retrain from scratch.   |
| Per-class recall near zero for some classes          | Class imbalance; oversample rare `(unicode, font)` pairs or add synthetic.   |

---

## 6. Roadmap

- Synthetic data generator that renders characters in arbitrary fonts to bootstrap
  the dataset without manual annotation.
- Two-stage variant: a font-agnostic detector + a small font classifier crop-head,
  to decouple localisation from font recognition.
- Export to ONNX for offline deployment on CPU-only targets.
