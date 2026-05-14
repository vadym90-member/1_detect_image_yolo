"""Persistent, append-only mapping between ``(unicode_codepoint, font_type)`` pairs
and integer YOLO class ids.

Class ids are assigned in the order pairs are first observed and never reused, so
that re-running the dataset builder against an extended raw corpus keeps every
previously trained class id stable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClassEntry:
    id: int
    unicode: int
    char: str
    font: str

    @staticmethod
    def from_pair(class_id: int, unicode_cp: int, font: str) -> "ClassEntry":
        return ClassEntry(
            id=class_id,
            unicode=unicode_cp,
            char=chr(unicode_cp),
            font=font,
        )


class ClassMap:
    """Bidirectional ``(unicode, font) <-> class_id`` table with on-disk persistence."""

    def __init__(self, entries: Optional[Iterable[ClassEntry]] = None) -> None:
        self._by_pair: Dict[Tuple[int, str], ClassEntry] = {}
        self._by_id: Dict[int, ClassEntry] = {}
        for e in entries or ():
            self._insert(e)

    # ------------------------------------------------------------------ I/O

    @classmethod
    def load(cls, path: str | Path) -> "ClassMap":
        path = Path(path)
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as fh:
            blob = json.load(fh)
        if blob.get("version") != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported class_map version: {blob.get('version')!r} "
                f"(expected {SCHEMA_VERSION})"
            )
        entries = [
            ClassEntry(
                id=int(item["id"]),
                unicode=int(item["unicode"]),
                char=item["char"],
                font=str(item["font"]),
            )
            for item in blob["classes"]
        ]
        return cls(entries)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "version": SCHEMA_VERSION,
            "classes": [asdict(e) for e in self.entries()],
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(blob, fh, ensure_ascii=False, indent=2)
            fh.write("\n")

    # -------------------------------------------------------------- accessors

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, pair: Tuple[int, str]) -> bool:
        return pair in self._by_pair

    def entries(self) -> List[ClassEntry]:
        return [self._by_id[i] for i in sorted(self._by_id)]

    def get_id(self, unicode_cp: int, font: str) -> int:
        return self._by_pair[(unicode_cp, font)].id

    def decode(self, class_id: int) -> Tuple[int, str]:
        entry = self._by_id[class_id]
        return entry.unicode, entry.font

    def names(self) -> Dict[int, str]:
        """Render the human-readable ``names`` block for the dataset YAML."""
        return {
            e.id: f"U+{e.unicode:04X} {e.font}" for e in self.entries()
        }

    # --------------------------------------------------------------- mutators

    def get_or_create_id(self, unicode_cp: int, font: str) -> int:
        """Return an existing class id or append a new one."""
        key = (unicode_cp, font)
        existing = self._by_pair.get(key)
        if existing is not None:
            return existing.id
        new_id = (max(self._by_id) + 1) if self._by_id else 0
        entry = ClassEntry.from_pair(new_id, unicode_cp, font)
        self._insert(entry)
        return new_id

    def _insert(self, entry: ClassEntry) -> None:
        if entry.id in self._by_id:
            raise ValueError(f"Duplicate class id: {entry.id}")
        key = (entry.unicode, entry.font)
        if key in self._by_pair:
            raise ValueError(f"Duplicate (unicode, font) pair: {key}")
        self._by_id[entry.id] = entry
        self._by_pair[key] = entry
