"""Sanity-check a built YOLO dataset.

Verifies, per split:
- Every image has a sibling label file (and vice versa).
- Every label line parses, with class id in range and bbox coords in ``(0, 1]``.
- The class id space is contiguous starting at 0 across the whole dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Set

import yaml

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _iter_images(images_dir: Path) -> List[Path]:
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def _read_names(dataset_yaml: Path) -> int:
    with dataset_yaml.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    names = doc.get("names") or {}
    return len(names)


def verify(root: Path, dataset_yaml: Path) -> List[str]:
    errors: List[str] = []
    n_classes = _read_names(dataset_yaml)
    seen_class_ids: Set[int] = set()

    for split in ("train", "val", "test"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            errors.append(f"missing split dir: {split}")
            continue

        images = _iter_images(img_dir)
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in lbl_dir.glob("*.txt")}

        for missing in sorted(image_stems - label_stems):
            errors.append(f"{split}: image without label: {missing}")
        for missing in sorted(label_stems - image_stems):
            errors.append(f"{split}: label without image: {missing}")

        for lbl in sorted(lbl_dir.glob("*.txt")):
            with lbl.open("r", encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) != 5:
                        errors.append(
                            f"{lbl}:{lineno}: expected 5 fields, got {len(parts)}"
                        )
                        continue
                    try:
                        cid = int(parts[0])
                        cx, cy, w, h = (float(v) for v in parts[1:])
                    except ValueError as exc:
                        errors.append(f"{lbl}:{lineno}: parse error: {exc}")
                        continue
                    if not (0 <= cid < n_classes):
                        errors.append(
                            f"{lbl}:{lineno}: class id {cid} out of range "
                            f"[0,{n_classes})"
                        )
                    for name, v in (("cx", cx), ("cy", cy), ("w", w), ("h", h)):
                        if not (0.0 <= v <= 1.0):
                            errors.append(
                                f"{lbl}:{lineno}: {name}={v} outside [0,1]"
                            )
                    if w <= 0 or h <= 0:
                        errors.append(f"{lbl}:{lineno}: non-positive w/h")
                    seen_class_ids.add(cid)

    if seen_class_ids:
        expected = set(range(max(seen_class_ids) + 1))
        gaps = sorted(expected - seen_class_ids)
        if gaps:
            errors.append(f"class ids never observed in labels: {gaps}")

    return errors


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("data/yolo"))
    p.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("configs/dataset/chars.yaml"),
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    errors = verify(args.root, args.dataset_yaml)
    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(f"\n{len(errors)} error(s).", file=sys.stderr)
        return 1
    print("Dataset OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
