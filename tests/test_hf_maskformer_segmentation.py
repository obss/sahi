"""Tests for HuggingFace segmentation model integration."""

from __future__ import annotations

import sys

import pytest

from sahi.models.hugging_face_universal_segmentation import HuggingFaceUniversalSegmentationModel, SegmentationType
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image
from tests.utils.huggingface import HuggingfaceConstants

try:
    from transformers import AutoModelForUniversalSegmentation, MaskFormerImageProcessor
except ImportError:
    AutoModelForUniversalSegmentation = None
    MaskFormerImageProcessor = None

# Import transformers conditionally to avoid import errors in Python < 3.9
pytestmark = [
    pytest.mark.skipif(sys.version_info[:2] < (3, 9), reason="transformers>=4.49.0 requires Python 3.9 or higher"),
    pytest.mark.skipif(
        AutoModelForUniversalSegmentation is None or MaskFormerImageProcessor is None,
        reason="transformers not installed",
    ),
]

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

IMAGE_PATH = "tests/data/small-vehicles1.jpeg"

IMAGE_ARR = read_image(IMAGE_PATH)
ORIGINAL_IMAGE_H = IMAGE_ARR.shape[0]
ORIGINAL_IMAGE_W = IMAGE_ARR.shape[1]


# this is minimum number of cars in the image that we required the model to detect
# if the model fails below this, it might but not always indicate a problem.
MINIMUM_CAR_SEGMENTS = 9


def test_load_model() -> None:
    def test_by_segmentation_type(segmentation_type: SegmentationType) -> None:
        """Test loading a HuggingFace maskformer univeral segmentation model."""
        huggingface_segmentation_model = HuggingFaceUniversalSegmentationModel(
            model_path=HuggingfaceConstants.MASKFORMER_MODEL_PATH,
            device=MODEL_DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            segmentation_type=segmentation_type,
        )

        assert huggingface_segmentation_model.model is not None

    test_by_segmentation_type(SegmentationType.INSTANCE_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.SEMANTIC_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.PANOPTIC_SEGMENTATION)


def test_set_model() -> None:
    def test_by_segmentation_type(segmentation_type: SegmentationType) -> None:
        """Test setting a pre-loaded HuggingFace maskformer univeral segmentation model."""
        assert AutoModelForUniversalSegmentation
        assert MaskFormerImageProcessor

        huggingface_model = AutoModelForUniversalSegmentation.from_pretrained(
            HuggingfaceConstants.MASKFORMER_MODEL_PATH
        )
        huggingface_processor = MaskFormerImageProcessor.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)

        huggingface_segmentation_model = HuggingFaceUniversalSegmentationModel(
            model=huggingface_model,
            processor=huggingface_processor,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            device=MODEL_DEVICE,
            segmentation_type=segmentation_type,
        )

        assert huggingface_segmentation_model.model is not None

    test_by_segmentation_type(SegmentationType.INSTANCE_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.SEMANTIC_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.PANOPTIC_SEGMENTATION)


def test_pre_process_handler() -> None:
    def test_by_segmentation_type(segmentation_type: SegmentationType) -> None:
        """Test pre_process handler with HuggingFace maskformer univeral segmentation model."""
        assert AutoModelForUniversalSegmentation is not None and MaskFormerImageProcessor is not None

        model = AutoModelForUniversalSegmentation.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor = MaskFormerImageProcessor.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor.do_resize = False

        huggingface_segmentation_model = HuggingFaceUniversalSegmentationModel(
            model=model,
            processor=processor,
            mask_threshold=MASK_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            segmentation_type=segmentation_type,
        )

        pre_process_output = huggingface_segmentation_model.prepost_handler.handle_pre_process(IMAGE_ARR)

        assert pre_process_output.get("pixel_values", None) is not None
        assert pre_process_output.get("pixel_mask", None) is not None

        assert (pre_process_output["pixel_values"].shape[-2], pre_process_output["pixel_values"].shape[-1]) == (
            ORIGINAL_IMAGE_H,
            ORIGINAL_IMAGE_W,
        )

        assert pre_process_output["pixel_values"].device == huggingface_segmentation_model.device
        assert pre_process_output["pixel_mask"].device == huggingface_segmentation_model.device

    test_by_segmentation_type(SegmentationType.INSTANCE_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.SEMANTIC_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.PANOPTIC_SEGMENTATION)


def test_post_process_handler() -> None:
    def test_by_segmentation_type(segmentation_type: SegmentationType) -> None:
        """Test post_process_handler with HuggingFace maskformer univeral segmentation model."""
        assert AutoModelForUniversalSegmentation is not None and MaskFormerImageProcessor is not None

        model = AutoModelForUniversalSegmentation.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor = MaskFormerImageProcessor.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor.do_resize = False

        huggingface_segmentation_model = HuggingFaceUniversalSegmentationModel(
            model=model,
            processor=processor,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            device=MODEL_DEVICE,
            segmentation_type=segmentation_type,
        )

        # perform inference
        huggingface_segmentation_model.perform_inference(IMAGE_ARR)
        original_predictions = huggingface_segmentation_model.original_predictions
        assert original_predictions

        post_processed_outputs = huggingface_segmentation_model.prepost_handler.handle_post_process(
            original_predictions, [(ORIGINAL_IMAGE_H, ORIGINAL_IMAGE_W)]
        )

        assert len(post_processed_outputs) > 0
        assert len(post_processed_outputs[0]["segmentation"]) == len(post_processed_outputs[0]["segments_info"])

        assert (
            post_processed_outputs[0]["segmentation"][0].shape[-2],
            post_processed_outputs[0]["segmentation"][0].shape[-1],
        ) == (ORIGINAL_IMAGE_H, ORIGINAL_IMAGE_W)

        assert "label_id" in post_processed_outputs[0]["segments_info"][0].keys()
        assert "score" in post_processed_outputs[0]["segments_info"][0].keys()

    test_by_segmentation_type(SegmentationType.INSTANCE_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.SEMANTIC_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.PANOPTIC_SEGMENTATION)


def test_perform_inference() -> None:
    def test_by_segmentation_type(segmentation_type: SegmentationType) -> None:
        """Test perform_inference with HuggingFace maskformer univeral segmentation model."""
        assert AutoModelForUniversalSegmentation is not None and MaskFormerImageProcessor is not None

        model = AutoModelForUniversalSegmentation.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor = MaskFormerImageProcessor.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor.do_resize = False

        huggingface_segmentation_model = HuggingFaceUniversalSegmentationModel(
            model=model,
            processor=processor,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            device=MODEL_DEVICE,
            segmentation_type=segmentation_type,
        )

        # perform inference
        huggingface_segmentation_model.perform_inference(IMAGE_ARR)
        original_predictions = huggingface_segmentation_model.original_predictions
        assert original_predictions

        post_processed_outputs = huggingface_segmentation_model.prepost_handler.handle_post_process(
            original_predictions, [(ORIGINAL_IMAGE_H, ORIGINAL_IMAGE_W)]
        )

        scores, cat_ids, segments = huggingface_segmentation_model.get_valid_predictions(post_processed_outputs[0])

        # find all car segments
        car_segments = 0
        for cat_id in cat_ids:
            if huggingface_segmentation_model.category_mapping[cat_id] == "car":  # if category car
                car_segments += 1

        assert car_segments >= MINIMUM_CAR_SEGMENTS

        assert all(score >= CONFIDENCE_THRESHOLD for score in scores)

        assert all(len(seg[0]) >= 8 for seg in segments)

    test_by_segmentation_type(SegmentationType.INSTANCE_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.SEMANTIC_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.PANOPTIC_SEGMENTATION)


def test_convert_original_predictions() -> None:
    def test_by_segmentation_type(segmentation_type: SegmentationType) -> None:
        """Test converting HuggingFace maskformer univeral segmentation model predictions to ObjectPrediction."""
        assert AutoModelForUniversalSegmentation is not None and MaskFormerImageProcessor is not None

        model = AutoModelForUniversalSegmentation.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor = MaskFormerImageProcessor.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor.do_resize = False

        huggingface_segmentation_model = HuggingFaceUniversalSegmentationModel(
            model=model,
            processor=processor,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            mask_threshold=MASK_THRESHOLD,
            device=MODEL_DEVICE,
            segmentation_type=segmentation_type,
        )

        # perform inference
        huggingface_segmentation_model.perform_inference(IMAGE_ARR)

        # convert predictions to ObjectPrediction list
        huggingface_segmentation_model.convert_original_predictions()
        object_prediction_list = huggingface_segmentation_model.object_prediction_list
        assert object_prediction_list
        assert isinstance(object_prediction_list[0], ObjectPrediction)

        car_segments = 0
        for object_prediction in object_prediction_list:
            assert isinstance(object_prediction, ObjectPrediction)
            assert object_prediction.score.value >= CONFIDENCE_THRESHOLD

            assert object_prediction.mask
            coco_segmentation = object_prediction.mask.segmentation
            assert coco_segmentation is not None
            assert isinstance(coco_segmentation, list)
            assert len(coco_segmentation) > 0

            if object_prediction.category.name == "car":
                car_segments += 1

        assert car_segments >= MINIMUM_CAR_SEGMENTS

    test_by_segmentation_type(SegmentationType.INSTANCE_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.SEMANTIC_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.PANOPTIC_SEGMENTATION)


def test_get_prediction_huggingface() -> None:
    def test_by_segmentation_type(segmentation_type: SegmentationType) -> None:
        """Test full-image prediction with HuggingFace maskformer univeral segmentation model."""
        assert AutoModelForUniversalSegmentation is not None and MaskFormerImageProcessor is not None

        model = AutoModelForUniversalSegmentation.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor = MaskFormerImageProcessor.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor.do_resize = False

        huggingface_detection_model = HuggingFaceUniversalSegmentationModel(
            model=model,
            processor=processor,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            segmentation_type=segmentation_type,
        )

        # get full sized prediction
        prediction_result = get_prediction(
            image=IMAGE_ARR,
            detection_model=huggingface_detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        assert len(object_prediction_list) > 0
        car_segments = person_segments = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                person_segments += 1
            elif object_prediction.category.name == "car":
                car_segments += 1

        assert person_segments == 0
        assert car_segments >= MINIMUM_CAR_SEGMENTS

    test_by_segmentation_type(SegmentationType.INSTANCE_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.SEMANTIC_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.PANOPTIC_SEGMENTATION)


def test_get_sliced_prediction_huggingface() -> None:

    def test_by_segmentation_type(segmentation_type: SegmentationType) -> None:
        """Test sliced prediction with HuggingFace maskformer univeral segmentation model."""
        assert AutoModelForUniversalSegmentation is not None and MaskFormerImageProcessor is not None

        model = AutoModelForUniversalSegmentation.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor = MaskFormerImageProcessor.from_pretrained(HuggingfaceConstants.MASKFORMER_MODEL_PATH)
        processor.do_resize = False

        huggingface_segmentation_model = HuggingFaceUniversalSegmentationModel(
            model=model,
            processor=processor,
            mask_threshold=MASK_THRESHOLD,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            segmentation_type=segmentation_type,
        )

        slice_height = int(ORIGINAL_IMAGE_H / 1.5)
        slice_width = int(ORIGINAL_IMAGE_W / 1.5)

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=IMAGE_PATH,
            detection_model=huggingface_segmentation_model,
            slice_height=slice_height,
            slice_width=slice_width,
            perform_standard_pred=False,
            batch_size=9,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        assert len(object_prediction_list) > 0
        car_segments = person_segments = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                person_segments += 1
            elif object_prediction.category.name == "car":
                car_segments += 1

        assert person_segments == 0
        assert car_segments >= MINIMUM_CAR_SEGMENTS

    test_by_segmentation_type(SegmentationType.INSTANCE_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.SEMANTIC_SEGMENTATION)
    test_by_segmentation_type(SegmentationType.PANOPTIC_SEGMENTATION)
