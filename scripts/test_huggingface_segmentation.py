"""Standalone test for HuggingfaceSegmentationModel.

Exercises load_model, perform_inference, get_prediction, and
get_sliced_prediction on a small Mask2Former instance-segmentation
checkpoint against tests/data/small-vehicles1.jpeg.

Run from repo root:
    .venv/bin/python scripts/test_huggingface_segmentation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sahi.auto_model import AutoDetectionModel  # noqa: E402
from sahi.models.huggingface_segmentation import HuggingfaceSegmentationModel, SegmentationType  # noqa: E402
from sahi.predict import get_prediction, get_sliced_prediction  # noqa: E402
from sahi.prediction import ObjectPrediction  # noqa: E402
from sahi.utils.cv import read_image  # noqa: E402

MODEL_PATH = "facebook/mask2former-swin-tiny-coco-instance"
IMAGE_PATH = str(REPO_ROOT / "tests" / "data" / "small-vehicles1.jpeg")
DEVICE = "cpu"
CONFIDENCE = 0.5


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def assert_predictions(predictions: list[ObjectPrediction], label: str) -> None:
    print(f"  {label}: {len(predictions)} predictions")
    assert isinstance(predictions, list), f"{label}: expected list, got {type(predictions)}"
    for p in predictions:
        assert isinstance(p, ObjectPrediction), f"{label}: bad element type {type(p)}"
        assert p.mask is not None or p.bbox is not None, f"{label}: prediction has no mask or bbox"
        assert p.score.value >= CONFIDENCE, f"{label}: score {p.score.value} below threshold"
    if predictions:
        sample = predictions[0]
        print(f"    sample: cat={sample.category.name!r} score={sample.score.value:.3f} bbox={sample.bbox.to_xywh()}")


def main() -> int:
    section("1. Direct instantiation + load_model")
    model = HuggingfaceSegmentationModel(
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE,
        device=DEVICE,
        load_at_init=True,
        segmentation_type=SegmentationType.INSTANCE_SEGMENTATION,
    )
    assert model.model is not None and model.processor is not None
    print(f"  model class : {type(model.model).__name__}")
    print(f"  processor   : {type(model.processor).__name__}")
    print(f"  num classes : {model.num_categories}")

    section("2. perform_inference + convert_original_predictions")
    image = read_image(IMAGE_PATH)
    print(f"  image shape : {image.shape}")
    model.perform_inference(image)
    assert model.original_predictions is not None
    model.convert_original_predictions(shift_amount=[0, 0])
    assert_predictions(model.object_prediction_list, "direct inference")

    section("3. get_prediction (full image)")
    result = get_prediction(
        image=image,
        detection_model=model,
        shift_amount=[0, 0],
        full_shape=None,
        postprocess=None,
    )
    assert_predictions(result.object_prediction_list, "get_prediction")

    section("4. AutoDetectionModel registry")
    auto_model = AutoDetectionModel.from_pretrained(
        model_type="huggingface_segmentation",
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE,
        device=DEVICE,
        segmentation_type=SegmentationType.INSTANCE_SEGMENTATION,
    )
    assert isinstance(auto_model, HuggingfaceSegmentationModel)
    print(f"  AutoDetectionModel returned: {type(auto_model).__name__}")

    section("5. get_sliced_prediction")
    sliced = get_sliced_prediction(
        image=IMAGE_PATH,
        detection_model=auto_model,
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        perform_standard_pred=False,
    )
    assert_predictions(sliced.object_prediction_list, "sliced")

    section("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
