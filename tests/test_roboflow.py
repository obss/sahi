import sys

import pytest

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] < (3, 12) or sys.platform == "darwin", reason="Requires Python 3.12 or higher and not macOS"
)


def test_roboflow_universe():
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


def test_rfdetr():
    """Test the RFDETR model classes and instances for object detection."""

    from rfdetr.detr import RFDETRBase
    from rfdetr.util.coco_classes import COCO_CLASSES

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
