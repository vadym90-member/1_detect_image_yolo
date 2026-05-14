"""Decode YOLO class ids back into ``(unicode_codepoint, font_type)`` pairs."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from src.labels.class_map import ClassMap


def load_class_map(path: str | Path) -> ClassMap:
    return ClassMap.load(path)


def decode(class_map: ClassMap, class_id: int) -> Tuple[int, str]:
    """Return ``(unicode_codepoint, font_type)`` for a predicted class id."""
    return class_map.decode(class_id)
