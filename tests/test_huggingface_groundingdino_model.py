"""Tests for HuggingFace GroundingDINO zero-shot detection integration."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sahi.models.huggingface import HuggingfaceDetectionModel

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] < (3, 9), reason="transformers>=4.49.0 requires Python 3.9 or higher"
)


class GroundingDinoForObjectDetection:
    """Minimal GroundingDINO-like model for zero-shot adapter tests."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={0: "LABEL_0"}, num_labels=1)

    def to(self, device: str) -> GroundingDinoForObjectDetection:
        return self

    def __call__(self, **inputs: object) -> SimpleNamespace:
        return SimpleNamespace()


class GroundingDinoProcessor:
    """Minimal GroundingDINO-like processor for zero-shot adapter tests."""

    def __init__(self) -> None:
        self.calls = []
        self.postprocess_calls = []

    def __call__(self, images: object, text: object, return_tensors: str) -> dict:
        import torch

        batch_size = len(images) if isinstance(images, list) else 1
        self.calls.append({"text": text})
        return {"input_ids": torch.ones((batch_size, 3), dtype=torch.long)}

    def post_process_grounded_object_detection(
        self,
        outputs: object,
        input_ids: object,
        threshold: float,
        text_threshold: float,
        target_sizes: list[tuple[int, int]],
    ) -> list[dict]:
        import torch

        self.postprocess_calls.append(
            {"threshold": threshold, "text_threshold": text_threshold, "target_sizes": target_sizes}
        )
        # third box is a combined phrase that must be filtered out when text_labels are fixed
        return [
            {
                "scores": torch.tensor([0.9, 0.8, 0.7]),
                "boxes": torch.tensor([[1.0, 2.0, 12.0, 14.0], [-5.0, 3.0, 40.0, 30.0], [2.0, 3.0, 10.0, 12.0]]),
                "text_labels": ["car", "truck", "car truck"],
            }
            for _ in target_sizes
        ]


def _build_model(processor: object, **kwargs: object) -> HuggingfaceDetectionModel:
    return HuggingfaceDetectionModel(
        model=GroundingDinoForObjectDetection(),
        processor=processor,
        device="cpu",
        load_at_init=True,
        **kwargs,
    )


def test_groundingdino_zero_shot_conversion_with_text_labels() -> None:
    """Fixed text_labels yield stable ids and combined phrases are dropped."""
    processor = GroundingDinoProcessor()
    model = _build_model(processor, confidence_threshold=0.4, text_threshold=0.2, text_labels=["car", "truck"])

    model.perform_inference(np.zeros((20, 30, 3), dtype=np.uint8))
    model.convert_original_predictions(shift_amount=[[5, 6]], full_shape=[[100, 120]])

    predictions = model.object_prediction_list
    assert processor.calls[0]["text"] == [["car", "truck"]]
    assert processor.postprocess_calls[0]["threshold"] == 0.4
    assert processor.postprocess_calls[0]["text_threshold"] == 0.2
    assert processor.postprocess_calls[0]["target_sizes"] == [(20, 30)]
    assert len(predictions) == 2  # "car truck" combined phrase filtered out
    assert (predictions[0].category.id, predictions[0].category.name) == (0, "car")
    assert predictions[0].bbox.to_xyxy() == [1.0, 2.0, 12.0, 14.0]
    assert predictions[0].score.value == pytest.approx(0.9)
    assert predictions[0].bbox.shift_amount == (5, 6)
    assert (predictions[1].category.id, predictions[1].category.name) == (1, "truck")
    assert predictions[1].bbox.to_xyxy() == [0.0, 3.0, 30.0, 20.0]  # negative x clamped to 0


def test_groundingdino_zero_shot_batch_repeats_text_labels() -> None:
    """text_labels are repeated per image for batch inference."""
    processor = GroundingDinoProcessor()
    model = _build_model(processor, confidence_threshold=0.4, text_labels=["car", "truck"])

    model.perform_batch_inference([np.zeros((20, 30, 3), np.uint8), np.zeros((40, 50, 3), np.uint8)])
    model.convert_original_predictions(shift_amount=[[0, 0], [10, 20]], full_shape=None)

    assert processor.calls[0]["text"] == [["car", "truck"], ["car", "truck"]]
    assert processor.postprocess_calls[0]["target_sizes"] == [(20, 30), (40, 50)]
    per_image = model.object_prediction_list_per_image
    assert [len(p) for p in per_image] == [2, 2]
    assert per_image[1][0].bbox.shift_amount == (10, 20)
