"""Tests for Roboflow Universe model integration."""

from __future__ import annotations

import sys

import pytest

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image

pytestmark = [
    pytest.mark.skipif(
        sys.version_info[:2] < (3, 12) or sys.platform in ("darwin", "win32"),
        reason="Requires Python 3.12 or higher, skipped on macOS and Windows",
    ),
    pytest.mark.flaky(reruns=3, reruns_delay=2),
]


def test_roboflow_universe() -> None:
    """Test the Roboflow Universe model for object detection."""
    model = AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model="rfdetr-base",
        confidence_threshold=0.5,
        device="cpu",
    )

    image_path = "tests/data/small-vehicles1.jpeg"
    image = read_image(image_path)

    result = get_prediction(image, model)
    predictions = result.object_prediction_list

    assert len(predictions) > 0

    sliced_results = get_sliced_prediction(
        image,
        model,
        slice_height=224,
        slice_width=224,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    sliced_predictions = sliced_results.object_prediction_list
    assert len(sliced_predictions) > len(predictions)


def test_roboflow_universe_segmentation() -> None:
    """Test the Roboflow Universe model for instance segmentation."""
    model = AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model="coco-dataset-vdnr1/37",
        confidence_threshold=0.5,
        device="cpu",
    )

    assert model.has_mask

    image_path = "tests/data/small-vehicles1.jpeg"
    image = read_image(image_path)

    result = get_prediction(image, model)
    predictions = result.object_prediction_list

    assert len(predictions) > 0
    assert predictions[0].mask

    sliced_results = get_sliced_prediction(
        image,
        model,
        slice_height=224,
        slice_width=224,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    sliced_predictions = sliced_results.object_prediction_list
    assert len(sliced_predictions) > len(predictions)


def test_rfdetr() -> None:
    """Test the RFDETR model classes and instances for object detection."""
    from rfdetr.assets.coco_classes import COCO_CLASSES
    from rfdetr.detr import RFDETRBase

    models = [
        RFDETRBase,
        RFDETRBase(),
    ]
    for model_variant in models:
        model = AutoDetectionModel.from_pretrained(
            model_type="roboflow",
            model=model_variant,
            confidence_threshold=0.5,
            category_mapping=COCO_CLASSES,
            device="cpu",
        )

        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        result = get_prediction(image, model)
        predictions = result.object_prediction_list

        assert len(predictions) > 0

        sliced_results = get_sliced_prediction(
            image,
            model,
            slice_height=224,
            slice_width=224,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        sliced_predictions = sliced_results.object_prediction_list
        assert len(sliced_predictions) > len(predictions)


def test_rfdetr_seg() -> None:
    """Test the RFDETR model classes and instances for instance segmentation."""
    from rfdetr.assets.coco_classes import COCO_CLASSES
    from rfdetr.detr import RFDETRSegMedium

    models = [
        RFDETRSegMedium,
        RFDETRSegMedium(),
    ]
    for model_variant in models:
        model = AutoDetectionModel.from_pretrained(
            model_type="roboflow",
            model=model_variant,
            confidence_threshold=0.5,
            category_mapping=COCO_CLASSES,
            image_size=432,
            device="cpu",
        )

        assert model.has_mask

        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        result = get_prediction(image, model)
        predictions = result.object_prediction_list

        assert len(predictions) > 0
        assert predictions[0].mask

        sliced_results = get_sliced_prediction(
            image,
            model,
            slice_height=224,
            slice_width=224,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        sliced_predictions = sliced_results.object_prediction_list
        assert len(sliced_predictions) > len(predictions)


def test_rfdetr_seg_by_class_name() -> None:
    """An RF-DETR class name string selects a local model instead of Roboflow Universe."""
    from rfdetr.assets.coco_classes import COCO_CLASSES

    model = AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model="RFDETRSegMedium",
        confidence_threshold=0.5,
        category_mapping=COCO_CLASSES,
        image_size=432,
        device="cpu",
    )

    assert model.has_mask

    result = get_prediction(read_image("tests/data/small-vehicles1.jpeg"), model)
    assert len(result.object_prediction_list) > 0


def test_rfdetr_class_name_does_not_use_universe() -> None:
    """Class-name strings must not be routed to the API, plain strings must be."""
    from sahi.models.roboflow import RoboflowDetectionModel

    local = RoboflowDetectionModel(model="RFDETRSegMedium", load_at_init=False)
    assert local._use_universe is False

    universe = RoboflowDetectionModel(model="rfdetr-base", load_at_init=False)
    assert universe._use_universe is True


def test_rfdetr_unresolvable_model_raises() -> None:
    """An unusable local model reports how to pass a local RF-DETR model."""
    from sahi.models.roboflow import RoboflowDetectionModel

    detection_model = RoboflowDetectionModel(model=None, load_at_init=False)
    with pytest.raises(ValueError, match="Could not resolve a local RF-DETR model"):
        detection_model.load_model()
