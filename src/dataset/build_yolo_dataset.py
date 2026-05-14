"""Convert raw annotations + images into the YOLO dataset format.

Pipeline:

1. Discover every ``*.json`` annotation under ``--raw/annotations/``.
2. Parse and validate each annotation (``src.labels.schema``).
3. Update the persistent class map (``src.labels.class_map``) — append-only.
4. Assign each image to a train/val/test split (``src.dataset.split``).
5. Copy/symlink the image and write its YOLO ``.txt`` label file under
   ``--out/{images,labels}/<split>/``.
6. Regenerate the ``names:`` block of the dataset YAML.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

from src.dataset.split import SplitRatios, assign_splits
from src.labels.class_map import ClassMap
from src.labels.schema import Annotation, AnnotationError


def _bbox_xyxy_to_yolo(
    bbox: Tuple[float, float, float, float], img_w: int, img_h: int
) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0 / img_w
    cy = (y0 + y1) / 2.0 / img_h
    w = (x1 - x0) / img_w
    h = (y1 - y0) / img_h
    return cx, cy, w, h


def _format_label_line(
    class_id: int, cx: float, cy: float, w: float, h: float
) -> str:
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def _copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _write_label(dst: Path, lines: List[str]) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        if lines:
            fh.write("\n")


def _update_dataset_yaml(yaml_path: Path, class_map: ClassMap) -> None:
    with yaml_path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    doc["names"] = class_map.names()
    with yaml_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False, allow_unicode=True)


def build(
    raw_dir: Path,
    out_dir: Path,
    class_map_path: Path,
    dataset_yaml: Path,
    ratios: SplitRatios,
) -> None:
    raw_images = raw_dir / "images"
    raw_annotations = raw_dir / "annotations"
    if not raw_annotations.is_dir():
        raise FileNotFoundError(f"No annotations directory: {raw_annotations}")
    if not raw_images.is_dir():
        raise FileNotFoundError(f"No images directory: {raw_images}")

    class_map = ClassMap.load(class_map_path)

    # Wipe previous split dirs but keep the parent skeleton (so .gitkeep stays).
    for sub in ("images", "labels"):
        for split in ("train", "val", "test"):
            d = out_dir / sub / split
            if d.exists():
                for f in d.iterdir():
                    if f.name == ".gitkeep":
                        continue
                    if f.is_dir():
                        shutil.rmtree(f)
                    else:
                        f.unlink()
            d.mkdir(parents=True, exist_ok=True)

    annotations: List[Tuple[Path, Annotation]] = []
    for ann_path in sorted(raw_annotations.glob("*.json")):
        try:
            ann = Annotation.load(ann_path)
        except AnnotationError as exc:
            raise AnnotationError(f"{ann_path}: {exc}") from exc
        annotations.append((ann_path, ann))

    if not annotations:
        raise RuntimeError(f"No annotations found in {raw_annotations}")

    # First pass: register all (unicode, font) pairs to keep ids stable across runs.
    for _, ann in annotations:
        for inst in ann.instances:
            class_map.get_or_create_id(inst.unicode, inst.font)

    image_stems = [Path(ann.image).stem for _, ann in annotations]
    if len(set(image_stems)) != len(image_stems):
        raise RuntimeError("Duplicate image stems in raw annotations")
    splits = assign_splits(image_stems, ratios)

    for ann_path, ann in annotations:
        image_path = raw_images / ann.image
        if not image_path.exists():
            raise FileNotFoundError(
                f"Annotation {ann_path} references missing image {image_path}"
            )
        stem = Path(ann.image).stem
        split = splits[stem]
        dst_image = out_dir / "images" / split / ann.image
        dst_label = out_dir / "labels" / split / f"{stem}.txt"

        _copy_image(image_path, dst_image)

        lines: List[str] = []
        for inst in ann.instances:
            cid = class_map.get_id(inst.unicode, inst.font)
            cx, cy, w, h = _bbox_xyxy_to_yolo(inst.bbox_xyxy, ann.width, ann.height)
            lines.append(_format_label_line(cid, cx, cy, w, h))
        _write_label(dst_label, lines)

    class_map.save(class_map_path)
    _update_dataset_yaml(dataset_yaml, class_map)


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--raw", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("data/yolo"))
    p.add_argument(
        "--class-map",
        type=Path,
        default=Path("configs/dataset/class_map.json"),
    )
    p.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("configs/dataset/chars.yaml"),
    )
    p.add_argument(
        "--split",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=[0.8, 0.1, 0.1],
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    ratios = SplitRatios(*args.split)
    build(
        raw_dir=args.raw,
        out_dir=args.out,
        class_map_path=args.class_map,
        dataset_yaml=args.dataset_yaml,
        ratios=ratios,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
