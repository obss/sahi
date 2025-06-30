# OBSS SAHI Tool
# Code written by Fatih Cagatay Akyon, 2025.

import sys
import pytest

from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.ultralytics import (
    UltralyticsTestConstants,
    download_yolo11n_model,
    download_yolo11n_obb_model,
    download_yolo11n_onnx_model,
    download_yolo11n_seg_model,
)

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640

def test_load_yolo11_model():
    from sahi.models.ultralytics import UltralyticsDetectionModel

    download_yolo11n_model()

    detection_model = UltralyticsDetectionModel(
        model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=MODEL_DEVICE,
        category_remapping=None,
        load_at_init=True,
    )

    assert detection_model.model is not None
    assert hasattr(detection_model.model, "task")
    assert detection_model.model.task == "detect"

@pytest.mark.skipif(sys.version_info < (3, 9), reason="ONNX model tests require Python 3.9 or higher")
def test_load_yolo11_onnx_model():
    from sahi.models.ultralytics import UltralyticsDetectionModel

    download_yolo11n_onnx_model()

    detection_model = UltralyticsDetectionModel(
        model_path=UltralyticsTestConstants.YOLO11N_ONNX_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=MODEL_DEVICE,
        load_at_init=True,
    )

    assert detection_model.model is not None

def test_perform_inference_yolo11():
    from sahi.models.ultralytics import UltralyticsDetectionModel

    # init model
    detection_model = UltralyticsDetectionModel(
        model_path="yolo11n.pt",
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
    detection_model.perform_inference(image)
    original_predictions = detection_model.original_predictions
    assert original_predictions
    assert isinstance(original_predictions, list)

    boxes = original_predictions[0].data

    # verify predictions
    assert len(detection_model.category_names) == 80
    for box in boxes:  # type: ignore
        assert box[4].item() >= CONFIDENCE_THRESHOLD

def test_yolo11_segmentation():
    from sahi.models.ultralytics import UltralyticsDetectionModel

    # init model
    download_yolo11n_seg_model()

    detection_model = UltralyticsDetectionModel(
        model_path=UltralyticsTestConstants.YOLO11N_SEG_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=MODEL_DEVICE,
        category_remapping=None,
        load_at_init=True,
        image_size=IMAGE_SIZE,
    )

    # Verify model properties
    assert detection_model.has_mask
    assert detection_model.model.task == "segment"

    # prepare image and run inference
    image_path = "tests/data/small-vehicles1.jpeg"
    image = read_image(image_path)
    detection_model.perform_inference(image)

    # Verify segmentation output
    original_predictions = detection_model.original_predictions
    assert original_predictions
    assert isinstance(original_predictions, list)
    boxes = original_predictions[0][0]  # Boxes
    masks = original_predictions[0][1]  # Masks

    assert len(boxes) > 0
    assert masks.shape[0] == len(boxes)  # One mask per box
    assert len(masks.shape) == 3  # (num_predictions, height, width)

def test_yolo11_obb():
    from sahi.models.ultralytics import UltralyticsDetectionModel

    # init model
    download_yolo11n_obb_model()

    detection_model = UltralyticsDetectionModel(
        model_path=UltralyticsTestConstants.YOLO11N_OBB_MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=MODEL_DEVICE,
        category_remapping=None,
        load_at_init=True,
        image_size=640,
    )

    # Verify model task
    assert detection_model.is_obb
    assert detection_model.model.task == "obb"

    # prepare image and run inference
    image_url = "https://ultralytics.com/images/boats.jpg"
    image_path = "tests/data/boats.jpg"
    download_from_url(image_url, to_path=image_path)
    image = read_image(image_path)
    detection_model.perform_inference(image)

    # Verify OBB predictions
    original_predictions = detection_model.original_predictions
    assert original_predictions
    assert isinstance(original_predictions, list)
    boxes = original_predictions[0][0]  # Original box data
    obb_points = original_predictions[0][1]  # OBB points in xyxyxyxy format

    assert len(boxes) > 0
    # Check box format: x1,y1,x2,y2,conf,cls
    assert boxes.shape[1] == 6
    # Check OBB points format
    assert obb_points.shape[1:] == (4, 2)  # (N, 4, 2) format

    # Convert predictions and verify
    detection_model.convert_original_predictions()
    object_prediction_list = detection_model.object_prediction_list

    # Verify converted predictions
    assert len(object_prediction_list) == len(boxes)
    for object_prediction in object_prediction_list:
        # Verify confidence threshold
        assert isinstance(object_prediction, ObjectPrediction)
        assert object_prediction.score.value >= CONFIDENCE_THRESHOLD

        assert object_prediction.mask
        coco_segmentation = object_prediction.mask.segmentation
        # Verify segmentation exists (converted from OBB)
        assert coco_segmentation is not None
        # Verify segmentation is a list of points
        assert isinstance(coco_segmentation, list)
        assert len(coco_segmentation) > 0