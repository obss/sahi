"""Tests for HuggingFace segmentation model integration.

One parametrized suite covers MaskFormer, Mask2Former, and OneFormer across the
instance, semantic, and panoptic heads. A single checkpoint serves all three
heads for MaskFormer/OneFormer, while Mask2Former uses one checkpoint per head.
"""

from __future__ import annotations

import sys
from importlib.util import find_spec

import pytest

from sahi.auto_model import AutoDetectionModel
from sahi.models.huggingface_segmentation import HuggingfaceSegmentationModel, SegmentationType
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image
from tests.utils.huggingface import HuggingfaceConstants as C

pytestmark = [
    pytest.mark.skipif(sys.version_info[:2] < (3, 9), reason="transformers>=4.49.0 requires Python 3.9 or higher"),
    pytest.mark.skipif(find_spec("transformers") is None, reason="transformers not installed"),
]

DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_PATH = "tests/data/small-vehicles1.jpeg"
IMAGE = read_image(IMAGE_PATH)
IMAGE_H, IMAGE_W = IMAGE.shape[:2]

INSTANCE = SegmentationType.INSTANCE_SEGMENTATION
SEMANTIC = SegmentationType.SEMANTIC_SEGMENTATION
PANOPTIC = SegmentationType.PANOPTIC_SEGMENTATION

CASES = [
    pytest.param(C.MASK2FORMER_INSTANCE_MODEL_PATH, INSTANCE, id="mask2former-instance"),
    pytest.param(C.MASK2FORMER_SEMANTIC_MODEL_PATH, SEMANTIC, id="mask2former-semantic"),
    pytest.param(C.MASK2FORMER_PANOPTIC_MODEL_PATH, PANOPTIC, id="mask2former-panoptic"),
    pytest.param(C.MASKFORMER_MODEL_PATH, INSTANCE, id="maskformer-instance"),
    pytest.param(C.MASKFORMER_MODEL_PATH, SEMANTIC, id="maskformer-semantic"),
    pytest.param(C.MASKFORMER_MODEL_PATH, PANOPTIC, id="maskformer-panoptic"),
    pytest.param(C.ONEFORMER_MODEL_PATH, INSTANCE, id="oneformer-instance"),
    pytest.param(C.ONEFORMER_MODEL_PATH, SEMANTIC, id="oneformer-semantic"),
    pytest.param(C.ONEFORMER_MODEL_PATH, PANOPTIC, id="oneformer-panoptic"),
]


def build_model(model_path: str, segmentation_type: SegmentationType) -> HuggingfaceSegmentationModel:
    return HuggingfaceSegmentationModel(
        model_path=model_path,
        segmentation_type=segmentation_type,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        load_at_init=True,
    )


def assert_valid_segments(object_prediction_list: list[ObjectPrediction]) -> None:
    assert object_prediction_list
    for prediction in object_prediction_list:
        assert isinstance(prediction, ObjectPrediction)
        assert prediction.score.value >= CONFIDENCE_THRESHOLD
        assert prediction.mask is not None
        assert prediction.mask.segmentation


@pytest.mark.parametrize(("model_path", "segmentation_type"), CASES)
def test_load_model(model_path: str, segmentation_type: SegmentationType) -> None:
    assert build_model(model_path, segmentation_type).model is not None


@pytest.mark.parametrize(("model_path", "segmentation_type"), CASES)
def test_convert_original_predictions(model_path: str, segmentation_type: SegmentationType) -> None:
    model = build_model(model_path, segmentation_type)
    model.perform_inference(IMAGE)
    assert model.original_predictions is not None

    model.convert_original_predictions()
    assert_valid_segments(model.object_prediction_list)


@pytest.mark.parametrize(("model_path", "segmentation_type"), CASES)
def test_get_sliced_prediction(model_path: str, segmentation_type: SegmentationType) -> None:
    model = build_model(model_path, segmentation_type)
    result = get_sliced_prediction(
        image=IMAGE_PATH,
        detection_model=model,
        slice_height=int(IMAGE_H / 1.5),
        slice_width=int(IMAGE_W / 1.5),
        perform_standard_pred=False,
        batch_size=4,
    )
    assert_valid_segments(result.object_prediction_list)


def test_set_model() -> None:
    from transformers import AutoModelForUniversalSegmentation, AutoProcessor

    model = AutoModelForUniversalSegmentation.from_pretrained(C.MASK2FORMER_INSTANCE_MODEL_PATH)
    processor = AutoProcessor.from_pretrained(C.MASK2FORMER_INSTANCE_MODEL_PATH)
    seg_model = HuggingfaceSegmentationModel(
        model=model,
        processor=processor,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        segmentation_type=INSTANCE,
    )
    assert seg_model.model is not None


def test_get_prediction_detects_cars() -> None:
    """Accuracy check on a known instance checkpoint via AutoDetectionModel."""
    model = AutoDetectionModel.from_pretrained(
        model_type="huggingface_segmentation",
        model_path=C.MASK2FORMER_INSTANCE_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
        segmentation_type=INSTANCE,
    )
    assert isinstance(model, HuggingfaceSegmentationModel)

    result = get_prediction(image=IMAGE, detection_model=model, shift_amount=[0, 0], full_shape=None, postprocess=None)
    assert_valid_segments(result.object_prediction_list)
    car_segments = sum(p.category.name == "car" for p in result.object_prediction_list)
    assert car_segments >= 5
