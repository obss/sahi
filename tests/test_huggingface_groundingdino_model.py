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

MODEL_DEVICE = "cpu"


class GroundingDinoForObjectDetection:
    """Minimal GroundingDINO-like model for zero-shot adapter tests."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(id2label={0: "LABEL_0"}, num_labels=1)
        self.calls = []

    def to(self, device: str) -> GroundingDinoForObjectDetection:
        self.device = device
        return self

    def __call__(self, **inputs: object) -> SimpleNamespace:
        import torch

        self.calls.append(inputs)
        return SimpleNamespace(
            logits=torch.zeros((len(inputs["input_ids"]), 2, 4)),
            pred_boxes=torch.zeros((len(inputs["input_ids"]), 2, 4)),
        )


class GroundingDinoProcessor:
    """Minimal GroundingDINO-like processor for zero-shot adapter tests."""

    def __init__(self) -> None:
        self.calls = []
        self.postprocess_calls = []

    def __call__(self, images: object, text: object, return_tensors: str) -> dict:
        import torch

        batch_size = len(images) if isinstance(images, list) else 1
        self.calls.append({"images": images, "text": text, "return_tensors": return_tensors})
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
            {
                "outputs": outputs,
                "input_ids": input_ids,
                "threshold": threshold,
                "text_threshold": text_threshold,
                "target_sizes": target_sizes,
            }
        )
        return [
            {
                "scores": torch.tensor([0.9, 0.8, 0.7]),
                "boxes": torch.tensor([[1.0, 2.0, 12.0, 14.0], [-5.0, 3.0, 40.0, 30.0], [2.0, 3.0, 10.0, 12.0]]),
                "text_labels": ["car", "truck", "car truck"],
            }
            for _ in target_sizes
        ]


def test_groundingdino_image_size_uses_valid_processor_size() -> None:
    """Test GroundingDINO image_size maps to a valid HuggingFace size dict."""
    assert HuggingfaceDetectionModel._get_processor_size(
        GroundingDinoForObjectDetection(),
        640,
    ) == {"shortest_edge": 640, "longest_edge": 640}


def test_groundingdino_zero_shot_conversion_with_text_labels() -> None:
    """Test GroundingDINO-style zero-shot prediction conversion."""
    processor = GroundingDinoProcessor()
    huggingface_detection_model = HuggingfaceDetectionModel(
        model=GroundingDinoForObjectDetection(),
        processor=processor,
        confidence_threshold=0.4,
        text_threshold=0.2,
        text_labels=["car", "truck"],
        device=MODEL_DEVICE,
        load_at_init=True,
    )

    image = np.zeros((20, 30, 3), dtype=np.uint8)
    huggingface_detection_model.perform_inference(image)
    huggingface_detection_model.convert_original_predictions(
        shift_amount=[[5, 6]],
        full_shape=[[100, 120]],
    )

    object_prediction_list = huggingface_detection_model.object_prediction_list
    assert processor.calls[0]["text"] == [["car", "truck"]]
    assert processor.postprocess_calls[0]["threshold"] == 0.4
    assert processor.postprocess_calls[0]["text_threshold"] == 0.2
    assert processor.postprocess_calls[0]["target_sizes"] == [(20, 30)]
    assert len(object_prediction_list) == 2
    assert object_prediction_list[0].category.id == 0
    assert object_prediction_list[0].category.name == "car"
    assert object_prediction_list[0].bbox.to_xyxy() == [1.0, 2.0, 12.0, 14.0]
    assert object_prediction_list[0].score.value == pytest.approx(0.9)
    assert object_prediction_list[0].bbox.shift_amount == (5, 6)
    assert object_prediction_list[1].category.id == 1
    assert object_prediction_list[1].category.name == "truck"
    assert object_prediction_list[1].bbox.to_xyxy() == [0.0, 3.0, 30.0, 20.0]


def test_groundingdino_zero_shot_batch_repeats_text_labels() -> None:
    """Test GroundingDINO-style text labels are repeated for batch inference."""
    processor = GroundingDinoProcessor()
    huggingface_detection_model = HuggingfaceDetectionModel(
        model=GroundingDinoForObjectDetection(),
        processor=processor,
        confidence_threshold=0.4,
        text_labels=["car", "truck"],
        device=MODEL_DEVICE,
        load_at_init=True,
    )

    images = [
        np.zeros((20, 30, 3), dtype=np.uint8),
        np.zeros((40, 50, 3), dtype=np.uint8),
    ]
    huggingface_detection_model.perform_batch_inference(images)
    huggingface_detection_model.convert_original_predictions(
        shift_amount=[[0, 0], [10, 20]],
        full_shape=None,
    )

    assert processor.calls[0]["text"] == [["car", "truck"], ["car", "truck"]]
    assert processor.postprocess_calls[0]["target_sizes"] == [(20, 30), (40, 50)]
    assert len(huggingface_detection_model.object_prediction_list_per_image) == 2
    assert len(huggingface_detection_model.object_prediction_list_per_image[0]) == 2
    assert len(huggingface_detection_model.object_prediction_list_per_image[1]) == 2
    assert huggingface_detection_model.object_prediction_list_per_image[1][0].bbox.shift_amount == (10, 20)
