"""Tests for LibreYOLO detection model integration."""

from __future__ import annotations

import pytest

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image

pytest.importorskip("libreyolo")

# LibreYOLO downloads the weights to this path when they are missing
MODEL_PATH = "tests/data/models/libreyolo/LibreYOLO9t.pt"
CONFIDENCE_THRESHOLD = 0.3


def test_libreyolo_inference() -> None:
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="libreyolo",
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device="cpu",
        image_size=640,
    )
    assert len(detection_model.category_names) == 80

    detection_model.perform_inference(read_image("tests/data/small-vehicles1.jpeg"))
    detection_model.convert_original_predictions()
    object_prediction_list = detection_model.object_prediction_list

    assert len(object_prediction_list) > 0
    assert all(pred.score.value >= CONFIDENCE_THRESHOLD for pred in object_prediction_list)
    assert any(pred.category.name == "car" for pred in object_prediction_list)
