import pytest

from src.labels.schema import Annotation, AnnotationError


def _ann(**overrides):
    base = {
        "image": "0001.png",
        "width": 100,
        "height": 50,
        "instances": [
            {"unicode": 65, "font": "Sans", "bbox_xyxy": [10, 5, 30, 25]},
        ],
    }
    base.update(overrides)
    return base


def test_valid_annotation():
    ann = Annotation.from_dict(_ann())
    assert ann.image == "0001.png"
    assert len(ann.instances) == 1
    assert ann.instances[0].unicode == 65


def test_rejects_degenerate_bbox():
    blob = _ann(instances=[
        {"unicode": 65, "font": "Sans", "bbox_xyxy": [10, 5, 10, 25]},
    ])
    with pytest.raises(AnnotationError, match="Degenerate bbox"):
        Annotation.from_dict(blob)


def test_rejects_out_of_bounds():
    blob = _ann(instances=[
        {"unicode": 65, "font": "Sans", "bbox_xyxy": [10, 5, 30, 100]},
    ])
    with pytest.raises(AnnotationError, match="exceeds image bounds"):
        Annotation.from_dict(blob)


def test_rejects_empty_font():
    blob = _ann(instances=[
        {"unicode": 65, "font": "", "bbox_xyxy": [10, 5, 30, 25]},
    ])
    with pytest.raises(AnnotationError, match="font"):
        Annotation.from_dict(blob)


def test_rejects_negative_unicode():
    blob = _ann(instances=[
        {"unicode": -1, "font": "Sans", "bbox_xyxy": [10, 5, 30, 25]},
    ])
    with pytest.raises(AnnotationError, match="Negative unicode"):
        Annotation.from_dict(blob)


def test_rejects_non_positive_image_size():
    with pytest.raises(AnnotationError, match="Non-positive image size"):
        Annotation.from_dict(_ann(width=0))
