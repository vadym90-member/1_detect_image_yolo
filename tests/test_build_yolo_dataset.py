import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from src.dataset.build_yolo_dataset import build
from src.dataset.split import SplitRatios
from src.dataset.verify import verify
from src.labels.class_map import ClassMap


def _write_image(path: Path, size=(120, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(255, 255, 255)).save(path)


def _write_annotation(path: Path, image_name: str, instances) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "image": image_name,
        "width": 120,
        "height": 80,
        "instances": instances,
    }
    path.write_text(json.dumps(blob), encoding="utf-8")


def _make_dataset_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "path": "../../data/yolo",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


@pytest.fixture
def project(tmp_path: Path):
    raw = tmp_path / "data" / "raw"
    out = tmp_path / "data" / "yolo"
    class_map = tmp_path / "configs" / "dataset" / "class_map.json"
    dataset_yaml = tmp_path / "configs" / "dataset" / "chars.yaml"
    _make_dataset_yaml(dataset_yaml)

    # 30 small images is enough to populate all three splits with 0.8/0.1/0.1.
    for i in range(30):
        name = f"img_{i:03d}.png"
        _write_image(raw / "images" / name)
        _write_annotation(
            raw / "annotations" / f"img_{i:03d}.json",
            name,
            [
                {"unicode": 65, "font": "Sans", "bbox_xyxy": [10, 10, 50, 70]},
                {"unicode": 66, "font": "Sans", "bbox_xyxy": [60, 10, 110, 70]},
            ],
        )

    return {
        "raw": raw,
        "out": out,
        "class_map": class_map,
        "dataset_yaml": dataset_yaml,
    }


def test_build_produces_yolo_layout(project):
    build(
        raw_dir=project["raw"],
        out_dir=project["out"],
        class_map_path=project["class_map"],
        dataset_yaml=project["dataset_yaml"],
        ratios=SplitRatios(0.8, 0.1, 0.1),
    )

    out = project["out"]
    total_images = sum(
        len(list((out / "images" / s).glob("*.png")))
        for s in ("train", "val", "test")
    )
    total_labels = sum(
        len(list((out / "labels" / s).glob("*.txt")))
        for s in ("train", "val", "test")
    )
    assert total_images == 30
    assert total_labels == 30

    cm = ClassMap.load(project["class_map"])
    assert len(cm) == 2
    assert cm.get_id(65, "Sans") == 0
    assert cm.get_id(66, "Sans") == 1


def test_label_lines_are_normalised(project):
    build(
        raw_dir=project["raw"],
        out_dir=project["out"],
        class_map_path=project["class_map"],
        dataset_yaml=project["dataset_yaml"],
        ratios=SplitRatios(0.8, 0.1, 0.1),
    )

    label_files = []
    for split in ("train", "val", "test"):
        label_files.extend((project["out"] / "labels" / split).glob("*.txt"))
    assert label_files

    for lbl in label_files:
        for line in lbl.read_text(encoding="utf-8").strip().splitlines():
            parts = line.split()
            assert len(parts) == 5
            cid = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:])
            assert cid in (0, 1)
            for v in (cx, cy, w, h):
                assert 0.0 < v < 1.0


def test_verify_passes_on_built_dataset(project):
    build(
        raw_dir=project["raw"],
        out_dir=project["out"],
        class_map_path=project["class_map"],
        dataset_yaml=project["dataset_yaml"],
        ratios=SplitRatios(0.8, 0.1, 0.1),
    )
    errors = verify(project["out"], project["dataset_yaml"])
    assert errors == []


def test_dataset_yaml_names_block_populated(project):
    build(
        raw_dir=project["raw"],
        out_dir=project["out"],
        class_map_path=project["class_map"],
        dataset_yaml=project["dataset_yaml"],
        ratios=SplitRatios(0.8, 0.1, 0.1),
    )
    doc = yaml.safe_load(project["dataset_yaml"].read_text(encoding="utf-8"))
    assert doc["names"] == {
        0: "U+0041 Sans",
        1: "U+0042 Sans",
    }


def test_rerun_is_append_only(project, tmp_path):
    build(
        raw_dir=project["raw"],
        out_dir=project["out"],
        class_map_path=project["class_map"],
        dataset_yaml=project["dataset_yaml"],
        ratios=SplitRatios(0.8, 0.1, 0.1),
    )

    # Add a new image introducing a brand new (unicode, font) pair.
    new_name = "img_extra.png"
    _write_image(project["raw"] / "images" / new_name)
    _write_annotation(
        project["raw"] / "annotations" / "img_extra.json",
        new_name,
        [{"unicode": 67, "font": "Sans", "bbox_xyxy": [10, 10, 50, 70]}],
    )

    build(
        raw_dir=project["raw"],
        out_dir=project["out"],
        class_map_path=project["class_map"],
        dataset_yaml=project["dataset_yaml"],
        ratios=SplitRatios(0.8, 0.1, 0.1),
    )

    cm = ClassMap.load(project["class_map"])
    assert cm.get_id(65, "Sans") == 0
    assert cm.get_id(66, "Sans") == 1
    assert cm.get_id(67, "Sans") == 2
