import json

import pytest

from src.labels.class_map import ClassMap, SCHEMA_VERSION


def test_get_or_create_assigns_sequential_ids():
    cm = ClassMap()
    assert cm.get_or_create_id(65, "NotoSans-Regular") == 0
    assert cm.get_or_create_id(65, "NotoSans-Bold") == 1
    assert cm.get_or_create_id(97, "NotoSans-Regular") == 2
    # Re-requesting an existing pair returns the same id.
    assert cm.get_or_create_id(65, "NotoSans-Regular") == 0


def test_round_trip_save_load(tmp_path):
    cm = ClassMap()
    cm.get_or_create_id(65, "NotoSans-Regular")
    cm.get_or_create_id(97, "NotoSans-Bold")

    path = tmp_path / "class_map.json"
    cm.save(path)

    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["version"] == SCHEMA_VERSION
    assert [c["id"] for c in blob["classes"]] == [0, 1]
    assert blob["classes"][0]["char"] == "A"

    cm2 = ClassMap.load(path)
    assert len(cm2) == 2
    assert cm2.get_id(65, "NotoSans-Regular") == 0
    assert cm2.decode(1) == (97, "NotoSans-Bold")


def test_load_missing_returns_empty(tmp_path):
    cm = ClassMap.load(tmp_path / "does_not_exist.json")
    assert len(cm) == 0


def test_append_only_preserves_existing_ids(tmp_path):
    path = tmp_path / "class_map.json"
    cm = ClassMap()
    cm.get_or_create_id(65, "Sans")
    cm.get_or_create_id(66, "Sans")
    cm.save(path)

    cm2 = ClassMap.load(path)
    # New pair appears with a new id; previous ids are stable.
    assert cm2.get_or_create_id(67, "Sans") == 2
    assert cm2.get_id(65, "Sans") == 0
    assert cm2.get_id(66, "Sans") == 1


def test_names_format():
    cm = ClassMap()
    cm.get_or_create_id(0x41, "NotoSans-Regular")
    names = cm.names()
    assert names[0] == "U+0041 NotoSans-Regular"


def test_rejects_unknown_version(tmp_path):
    path = tmp_path / "class_map.json"
    path.write_text(json.dumps({"version": 999, "classes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported class_map version"):
        ClassMap.load(path)
