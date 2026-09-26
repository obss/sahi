"""Tests for LibreYOLO detection model integration."""

from __future__ import annotations

import importlib

import pytest

from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image
from sahi.utils.libreyolo import LibreYoloTestConstants, download_libreyolo9t_model

pytestmark = pytest.mark.skipif(importlib.util.find_spec("libreyolo") is None, reason="libreyolo is not installed")

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640


class TestLibreYoloDetectionModel:
    """Test LibreYOLO detection model functionality."""

    def test_load_model(self) -> None:
        """Test loading a LibreYOLO detection model."""
        from sahi.models.libreyolo import LibreYoloDetectionModel

        download_libreyolo9t_model()

        libreyolo_detection_model = LibreYoloDetectionModel(
            model_path=LibreYoloTestConstants.LIBREYOLO9T_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        assert libreyolo_detection_model.model is not None

    def test_set_model(self) -> None:
        """Test setting a pre-loaded LibreYOLO model."""
        from libreyolo import LibreYOLO

        from sahi.models.libreyolo import LibreYoloDetectionModel

        download_libreyolo9t_model()

        libreyolo_model = LibreYOLO(LibreYoloTestConstants.LIBREYOLO9T_MODEL_PATH)

        libreyolo_detection_model = LibreYoloDetectionModel(
            model=libreyolo_model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        assert libreyolo_detection_model.model is not None

    def test_perform_inference(self) -> None:
        """Test inference with LibreYOLO model."""
        from sahi.models.libreyolo import LibreYoloDetectionModel

        download_libreyolo9t_model()

        libreyolo_detection_model = LibreYoloDetectionModel(
            model_path=LibreYoloTestConstants.LIBREYOLO9T_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        libreyolo_detection_model.perform_inference(image)
        original_predictions = libreyolo_detection_model.original_predictions

        boxes = original_predictions
        assert boxes is not None

        # verify confidence threshold is respected
        for box in boxes[0]:
            assert box[4].item() >= CONFIDENCE_THRESHOLD

        # verify category names are loaded
        assert len(libreyolo_detection_model.category_names) == 80

    def test_convert_original_predictions(self) -> None:
        """Test converting LibreYOLO predictions to ObjectPrediction."""
        from sahi.models.libreyolo import LibreYoloDetectionModel

        download_libreyolo9t_model()

        libreyolo_detection_model = LibreYoloDetectionModel(
            model_path=LibreYoloTestConstants.LIBREYOLO9T_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        libreyolo_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        libreyolo_detection_model.convert_original_predictions()
        object_prediction_list = libreyolo_detection_model.object_prediction_list

        # verify predictions are ObjectPrediction instances
        assert len(object_prediction_list) > 0
        for object_prediction in object_prediction_list:
            assert isinstance(object_prediction, ObjectPrediction)
            assert object_prediction.score.value >= CONFIDENCE_THRESHOLD

    def test_auto_model_type(self) -> None:
        """Test that LibreYOLO can be loaded via AutoDetectionModel."""
        from sahi import AutoDetectionModel

        download_libreyolo9t_model()

        detection_model = AutoDetectionModel.from_pretrained(
            model_type="libreyolo",
            model_path=LibreYoloTestConstants.LIBREYOLO9T_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
        )

        assert detection_model.model is not None
