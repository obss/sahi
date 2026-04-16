"""Tests for Gemma-4 multimodal detection model integration.

These tests exercise ``Gemma4DetectionModel`` without downloading the real
Gemma-4 weights. The model and processor are replaced with lightweight stubs
that emulate ``AutoModelForMultimodalLM`` + ``AutoProcessor`` just enough to
drive the SAHI inference path.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from sahi.models.gemma4 import Gemma4DetectionModel
from sahi.prediction import ObjectPrediction

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] < (3, 9), reason="transformers>=4.49.0 requires Python 3.9 or higher"
)

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3


class _StubTensor:
    """Minimal tensor-like object supporting ``.shape`` and slicing used by the model."""

    def __init__(self, data: list[list[int]]) -> None:
        self._data = data
        rows = len(data)
        cols = len(data[0]) if data else 0
        self.shape = (rows, cols)

    def __getitem__(self, item: Any) -> Any:
        return self._data[0]


class _StubInputs(dict):
    """Dict-like container mirroring the real processor output."""

    def to(self, _device: Any) -> _StubInputs:
        return self


class _StubProcessor:
    """Returns a canned JSON response regardless of the input."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[dict] = []

    def apply_chat_template(self, messages: list[dict], **kwargs: Any) -> _StubInputs:
        self.calls.append({"messages": messages, "kwargs": kwargs})
        return _StubInputs(input_ids=_StubTensor([[0, 1, 2, 3]]))

    def decode(self, _tokens: Any, **_kwargs: Any) -> str:
        return self._response


class _StubModel:
    """Returns the stub token sequence so ``perform_inference`` can run end-to-end."""

    def __init__(self) -> None:
        self.device = "cpu"
        self.calls: list[dict] = []

    def generate(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return _StubTensor([[0, 1, 2, 3, 4, 5, 6, 7]])


def _make_model(response: str, classes: list[str] | None = None) -> Gemma4DetectionModel:
    detection_model = Gemma4DetectionModel(
        classes=classes or ["bike", "person"],
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=MODEL_DEVICE,
        load_at_init=False,
    )
    detection_model.set_model(_StubModel(), _StubProcessor(response))
    return detection_model


def test_set_classes_builds_category_mapping() -> None:
    model = _make_model("[]")
    model.set_classes(["car", "truck", "bus"])
    assert model.classes == ["car", "truck", "bus"]
    assert model.category_mapping == {0: "car", 1: "truck", 2: "bus"}
    assert model.num_categories == 3


def test_set_classes_rejects_empty_list() -> None:
    model = _make_model("[]")
    with pytest.raises(ValueError):
        model.set_classes([])


def test_extract_json_variants() -> None:
    direct = '[{"box_2d":[10,20,30,40],"label":"bike"}]'
    fenced = "```json\n" + direct + "\n```"
    noisy = "sure, here: " + direct + " — done"
    single = '{"box_2d":[10,20,30,40],"label":"bike"}'

    assert Gemma4DetectionModel._extract_json(direct)[0]["label"] == "bike"
    assert Gemma4DetectionModel._extract_json(fenced)[0]["box_2d"] == [10, 20, 30, 40]
    assert Gemma4DetectionModel._extract_json(noisy)[0]["label"] == "bike"
    assert Gemma4DetectionModel._extract_json(single) == [{"box_2d": [10, 20, 30, 40], "label": "bike"}]
    assert Gemma4DetectionModel._extract_json("no json at all") == []


def test_resize_to_patch_multiple() -> None:
    from PIL import Image

    image = Image.new("RGB", (100, 77))
    resized = Gemma4DetectionModel._resize_to_patch_multiple(image)
    assert resized.size == (96, 48)  # nearest multiples of 48 <= original dims


def test_perform_inference_without_classes_raises() -> None:
    model = _make_model("[]", classes=["bike"])
    model._classes = []  # simulate forgotten set_classes call
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        model.perform_inference(image)


def test_perform_inference_and_convert_predictions() -> None:
    response = '[{"box_2d":[100,200,400,600],"label":"bike"}, {"box_2d":[50,60,150,200],"label":"person","score":0.9}]'
    model = _make_model(response, classes=["bike", "person"])

    image = np.zeros((500, 1000, 3), dtype=np.uint8)
    model.perform_inference(image)
    model.convert_original_predictions()

    preds = model.object_prediction_list
    assert len(preds) == 2
    assert all(isinstance(p, ObjectPrediction) for p in preds)

    bike = preds[0]
    assert bike.category.id == 0
    assert bike.category.name == "bike"
    # box_2d [ymin=100, xmin=200, ymax=400, xmax=600] with 1000x500 image
    # → xmin=200, ymin=50, xmax=600, ymax=200
    bbox = bike.bbox.to_xyxy()
    assert bbox == [200.0, 50.0, 600.0, 200.0]

    person = preds[1]
    assert person.category.id == 1
    assert person.category.name == "person"
    assert person.score.value == pytest.approx(0.9)


def test_filters_unknown_labels_and_low_scores() -> None:
    response = '[{"box_2d":[0,0,100,100],"label":"unicorn"}, {"box_2d":[0,0,100,100],"label":"bike","score":0.1}]'
    model = _make_model(response, classes=["bike"])

    image = np.zeros((500, 500, 3), dtype=np.uint8)
    model.perform_inference(image)
    model.convert_original_predictions()

    # unicorn is not in classes; bike score is below threshold — both dropped.
    assert model.object_prediction_list == []


def test_perform_batch_inference_runs_per_image() -> None:
    response = '[{"box_2d":[0,0,1000,1000],"label":"bike"}]'
    model = _make_model(response, classes=["bike"])

    images = [
        np.zeros((200, 200, 3), dtype=np.uint8),
        np.zeros((400, 400, 3), dtype=np.uint8),
    ]
    model.perform_batch_inference(images)
    model.convert_original_predictions(
        shift_amount=[[0, 0], [0, 0]],
        full_shape=[[200, 200], [400, 400]],
    )

    per_image = model.object_prediction_list_per_image
    assert len(per_image) == 2
    assert per_image[0][0].bbox.to_xyxy() == [0.0, 0.0, 200.0, 200.0]
    assert per_image[1][0].bbox.to_xyxy() == [0.0, 0.0, 400.0, 400.0]


def test_set_instruction_overrides_prompt() -> None:
    model = _make_model("[]", classes=["car"])
    default_messages = model._build_messages(object())
    default_text = default_messages[0]["content"][1]["text"]
    assert "Detect every car" in default_text

    model.set_instruction("This is an aerial parking lot. Find every car, including partially occluded ones")
    custom_messages = model._build_messages(object())
    custom_text = custom_messages[0]["content"][1]["text"]
    assert "aerial parking lot" in custom_text
    # JSON schema suffix must still be present so we can parse the response.
    assert "box_2d" in custom_text and "Return only the JSON" in custom_text

    # Clearing restores default.
    model.set_instruction(None)
    assert "Detect every car" in model._build_messages(object())[0]["content"][1]["text"]


def test_auto_detection_model_registration() -> None:
    from sahi.auto_model import MODEL_TYPE_TO_MODEL_CLASS_NAME
    from sahi.utils.file import import_model_class

    assert MODEL_TYPE_TO_MODEL_CLASS_NAME["gemma4"] == "Gemma4DetectionModel"
    cls = import_model_class("gemma4", "Gemma4DetectionModel")
    assert cls is Gemma4DetectionModel
