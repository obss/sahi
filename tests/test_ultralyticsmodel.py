# OBSS SAHI Tool
# Code written by AnNT, 2024.

import unittest

from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.ultralytics import (
    UltralyticsTestConstants,
    download_yolo11n_model,
    download_yolo11n_obb_model,
    download_yolo11n_seg_model,
    download_yolov8n_model,
)

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640


class TestUltralyticsDetectionModel(unittest.TestCase):
    def test_load_yolov8_model(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel

        download_yolov8n_model()

        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsTestConstants.YOLOV8N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(detection_model.model, None)
        self.assertTrue(hasattr(detection_model.model, "task"))
        self.assertEqual(detection_model.model.task, "detect")

    def test_load_yolo11_model(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel

        download_yolo11n_model()

        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(detection_model.model, None)
        self.assertTrue(hasattr(detection_model.model, "task"))
        self.assertEqual(detection_model.model.task, "detect")

    def test_perform_inference_yolov8(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel

        # init model
        download_yolov8n_model()

        detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsTestConstants.YOLOV8N_MODEL_PATH,
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

        # find box of first car detection with conf greater than 0.5
        for box in boxes:  # type: ignore
            if box[5].item() == 2:  # if category car
                if box[4].item() > 0.5:
                    break

        # compare
        desired_bbox = [448, 309, 497, 342]
        predicted_bbox = list(map(int, box[:4].tolist()))
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(len(detection_model.category_names), 80)
        for box in boxes:  # type: ignore
            self.assertGreaterEqual(box[4].item(), CONFIDENCE_THRESHOLD)

    def test_perform_inference_yolo11(self):
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
        self.assertEqual(len(detection_model.category_names), 80)
        for box in boxes:  # type: ignore
            self.assertGreaterEqual(box[4].item(), CONFIDENCE_THRESHOLD)

    def test_yolo11_segmentation(self):
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
        self.assertTrue(detection_model.has_mask)
        self.assertEqual(detection_model.model.task, "segment")

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

        self.assertGreater(len(boxes), 0)
        self.assertEqual(masks.shape[0], len(boxes))  # One mask per box
        self.assertEqual(len(masks.shape), 3)  # (num_predictions, height, width)

    def test_yolo11_obb(self):
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
        self.assertTrue(detection_model.is_obb)
        self.assertEqual(detection_model.model.task, "obb")

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

        self.assertGreater(len(boxes), 0)
        # Check box format: x1,y1,x2,y2,conf,cls
        self.assertEqual(boxes.shape[1], 6)
        # Check OBB points format
        self.assertEqual(obb_points.shape[1:], (4, 2))  # (N, 4, 2) format

        # Convert predictions and verify
        detection_model.convert_original_predictions()
        object_prediction_list = detection_model.object_prediction_list

        # Verify converted predictions
        self.assertEqual(len(object_prediction_list), len(boxes))
        for object_prediction in object_prediction_list:
            # Verify confidence threshold
            assert isinstance(object_prediction, ObjectPrediction)
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

            assert object_prediction.mask
            coco_segmentation = object_prediction.mask.segmentation
            # Verify segmentation exists (converted from OBB)
            self.assertIsNotNone(coco_segmentation)
            # Verify segmentation is a list of points
            self.assertTrue(isinstance(coco_segmentation, list))
            self.assertGreater(len(coco_segmentation), 0)


if __name__ == "__main__":
    unittest.main()
