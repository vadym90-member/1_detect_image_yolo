"""Train or validate a YOLO detector.

Offline-only: training is bootstrapped from a YAML architecture file (random
init), never from a ``.pt`` checkpoint that would otherwise be downloaded.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List


def _force_offline_env() -> None:
    """Set environment flags that prevent Ultralytics from reaching the network."""
    os.environ.setdefault("YOLO_OFFLINE", "1")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def _train(args: argparse.Namespace) -> int:
    _force_offline_env()
    from ultralytics import YOLO  # imported lazily so env vars take effect first

    model_path = Path(args.model)
    if model_path.suffix.lower() != ".yaml":
        raise ValueError(
            f"--model must be a .yaml architecture file (got {model_path}). "
            "Loading .pt weights is disabled because this project runs offline."
        )
    if not model_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_path}")

    model = YOLO(str(model_path))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        pretrained=False,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        verbose=True,
    )
    return 0


def _val(args: argparse.Namespace) -> int:
    _force_offline_env()
    from ultralytics import YOLO

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    model = YOLO(str(weights))
    model.val(data=str(args.data), imgsz=args.imgsz)
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--mode", choices=("train", "val"), default="train",
    )
    p.add_argument("--model", type=Path, default=Path("configs/model/yolov8n.yaml"))
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--data", type=Path, default=Path("configs/dataset/chars.yaml"))
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--project", type=Path, default=Path("runs/train"))
    p.add_argument("--name", type=str, default="chars_v1")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.mode == "train":
        return _train(args)
    if args.weights is None:
        raise SystemExit("--weights is required for --mode val")
    return _val(args)


if __name__ == "__main__":
    raise SystemExit(main())
