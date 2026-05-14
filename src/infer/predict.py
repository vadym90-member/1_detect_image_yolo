"""Run a trained YOLO model on one or more images and emit decoded detections.

Each detection is reported as::

    {
        "bbox_xyxy": [x_min, y_min, x_max, y_max],   # pixel coordinates
        "unicode":   <int codepoint>,
        "char":      "<single character>",
        "font":      "<font family / style>",
        "confidence": <float in [0,1]>
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List


def _force_offline_env() -> None:
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


@dataclass
class Detection:
    bbox_xyxy: List[float]
    unicode: int
    char: str
    font: str
    confidence: float


def _run(args: argparse.Namespace) -> int:
    _force_offline_env()
    from ultralytics import YOLO

    from src.infer.decode import decode, load_class_map
    from src.utils.viz import draw_detections

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    class_map = load_class_map(args.class_map)
    model = YOLO(str(weights))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    for r in results:
        src_path = Path(r.path)
        detections: List[Detection] = []
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                unicode_cp, font = decode(class_map, cls_id)
                xyxy = [float(v) for v in box.xyxy[0].tolist()]
                detections.append(
                    Detection(
                        bbox_xyxy=xyxy,
                        unicode=unicode_cp,
                        char=chr(unicode_cp),
                        font=font,
                        confidence=float(box.conf.item()),
                    )
                )

        json_path = out_dir / f"{src_path.stem}.json"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump([asdict(d) for d in detections], fh, ensure_ascii=False, indent=2)
            fh.write("\n")

        if args.viz:
            draw_detections(
                src_path,
                detections,
                out_dir / f"{src_path.stem}.png",
            )

    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument(
        "--class-map",
        type=Path,
        default=Path("configs/dataset/class_map.json"),
    )
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, default=Path("runs/predict"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--viz", action="store_true", help="also save visualisations")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
