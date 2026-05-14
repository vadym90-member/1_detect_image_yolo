"""Raw annotation schema and validators.

A raw annotation describes one source image and the list of character instances
it contains, in *absolute pixel coordinates*. The dataset builder converts these
into YOLO-format normalised boxes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


class AnnotationError(ValueError):
    """Raised when a raw annotation fails validation."""


@dataclass(frozen=True)
class Instance:
    unicode: int
    font: str
    bbox_xyxy: Tuple[float, float, float, float]  # x_min, y_min, x_max, y_max

    def validate(self, img_w: int, img_h: int) -> None:
        if self.unicode < 0:
            raise AnnotationError(f"Negative unicode codepoint: {self.unicode}")
        if not self.font:
            raise AnnotationError("font must be a non-empty string")
        x0, y0, x1, y1 = self.bbox_xyxy
        if not (x1 > x0 and y1 > y0):
            raise AnnotationError(
                f"Degenerate bbox: {self.bbox_xyxy} (x1<=x0 or y1<=y0)"
            )
        if x0 < 0 or y0 < 0 or x1 > img_w or y1 > img_h:
            raise AnnotationError(
                f"Bbox {self.bbox_xyxy} exceeds image bounds {img_w}x{img_h}"
            )


@dataclass(frozen=True)
class Annotation:
    image: str
    width: int
    height: int
    instances: List[Instance]

    @classmethod
    def from_dict(cls, blob: dict) -> "Annotation":
        try:
            image = str(blob["image"])
            width = int(blob["width"])
            height = int(blob["height"])
            raw_instances = blob["instances"]
        except (KeyError, TypeError, ValueError) as exc:
            raise AnnotationError(f"Malformed annotation header: {exc}") from exc

        if width <= 0 or height <= 0:
            raise AnnotationError(f"Non-positive image size: {width}x{height}")
        if not isinstance(raw_instances, list):
            raise AnnotationError("'instances' must be a list")

        instances: List[Instance] = []
        for idx, item in enumerate(raw_instances):
            try:
                bbox = item["bbox_xyxy"]
                inst = Instance(
                    unicode=int(item["unicode"]),
                    font=str(item["font"]),
                    bbox_xyxy=(
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    ),
                )
            except (KeyError, TypeError, ValueError, IndexError) as exc:
                raise AnnotationError(
                    f"Malformed instance #{idx} in {image!r}: {exc}"
                ) from exc
            inst.validate(width, height)
            instances.append(inst)

        return cls(image=image, width=width, height=height, instances=instances)

    @classmethod
    def load(cls, path: str | Path) -> "Annotation":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))
