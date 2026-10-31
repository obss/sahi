"""Tests for class names when an Ultralytics model renames its classes after loading."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sahi.models.ultralytics import UltralyticsDetectionModel


class FakeTensor:
    """Minimal stand-in for the torch tensor the wrapper reads predictions from."""

    def __init__(self, values: list[list[float]]) -> None:
        self.values = np.asarray(values, dtype=np.float32)

    def cpu(self) -> FakeTensor:
        return self

    def detach(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.values


class FakePromptableModel:
    """Detection model that renames its classes with set_classes, like YOLOE and YOLO-World."""

    def __init__(self) -> None:
        self.names = {0: "0", 1: "1"}
        self.overrides = {"task": "detect"}

    def set_classes(self, names: list[str]) -> None:
        self.names = dict(enumerate(names))


def predict_names(detection_model: UltralyticsDetectionModel) -> list[str]:
    """Convert two fake boxes, one per class, and return the predicted class names."""
    detection_model._original_predictions = [
        SimpleNamespace(data=FakeTensor([[10, 10, 50, 50, 0.9, 0], [60, 60, 90, 90, 0.8, 1]]))
    ]
    detection_model._original_shapes = [(100, 100, 3)]
    detection_model.convert_original_predictions()
    return [prediction.category.name for prediction in detection_model.object_prediction_list]


def test_set_classes_after_loading_names_the_predictions() -> None:
    """Classes set after the model is loaded show up in the predictions."""
    fake_model = FakePromptableModel()
    detection_model = UltralyticsDetectionModel(confidence_threshold=0.3, device="cpu", load_at_init=False)
    detection_model.set_model(fake_model)

    fake_model.set_classes(["car", "truck"])

    assert predict_names(detection_model) == ["car", "truck"]


def test_category_mapping_passed_in_is_kept() -> None:
    """A category_mapping given by the user wins over the model's own class names."""
    fake_model = FakePromptableModel()
    detection_model = UltralyticsDetectionModel(
        confidence_threshold=0.3, device="cpu", load_at_init=False, category_mapping={"0": "person", "1": "bike"}
    )
    detection_model.set_model(fake_model)

    fake_model.set_classes(["car", "truck"])

    assert predict_names(detection_model) == ["person", "bike"]
