"""Visualise predicted detections by drawing labelled boxes over the source image."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def draw_detections(
    image_path: str | Path,
    detections: Iterable,  # iterable of src.infer.predict.Detection
    out_path: str | Path,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        x0, y0, x1, y1 = det.bbox_xyxy
        draw.rectangle([(x0, y0), (x1, y1)], outline=(255, 0, 0), width=2)
        label = f"U+{det.unicode:04X} {det.font} {det.confidence:.2f}"
        draw.text((x0, max(0.0, y0 - 10)), label, fill=(255, 0, 0), font=font)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
