"""Deterministic train/val/test split of an iterable of items."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

SplitName = str  # "train" | "val" | "test"


@dataclass(frozen=True)
class SplitRatios:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split ratios must sum to 1.0 (got {total:.6f})"
            )
        for name, v in (("train", self.train), ("val", self.val), ("test", self.test)):
            if v < 0:
                raise ValueError(f"Negative ratio for split {name!r}: {v}")


def _bucket(key: str, salt: str) -> float:
    """Hash a key to a stable float in ``[0, 1)``."""
    h = hashlib.sha256(f"{salt}:{key}".encode("utf-8")).digest()
    n = int.from_bytes(h[:8], "big")
    return (n & ((1 << 53) - 1)) / float(1 << 53)


def assign_splits(
    keys: Sequence[str],
    ratios: SplitRatios,
    salt: str = "1_detect_image_yolo",
) -> Dict[str, SplitName]:
    """Map each key to a split deterministically based on its hash."""
    out: Dict[str, SplitName] = {}
    for k in keys:
        r = _bucket(k, salt)
        if r < ratios.train:
            out[k] = "train"
        elif r < ratios.train + ratios.val:
            out[k] = "val"
        else:
            out[k] = "test"
    return out


def group_by_split(
    assignments: Dict[str, SplitName],
) -> Dict[SplitName, List[str]]:
    grouped: Dict[SplitName, List[str]] = {"train": [], "val": [], "test": []}
    for k, s in assignments.items():
        grouped[s].append(k)
    for v in grouped.values():
        v.sort()
    return grouped
