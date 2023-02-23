# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.yolov8 import Yolov8TestConstants, download_yolov8n_model, download_yolov8s_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 320


class TestYolov8DetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.yolov8 import Yolov8DetectionModel

        download_yolov8n_model()

        yolov8_detection_model = Yolov8DetectionModel(
            model_path=Yolov8TestConstants.YOLOV8N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(yolov8_detection_model.model, None)

    def test_set_model(self):

        from ultralytics import YOLO

        from sahi.models.yolov8 import Yolov8DetectionModel

        download_yolov8n_model()

        yolo_model = YOLO(Yolov8TestConstants.YOLOV8N_MODEL_PATH)

        yolov8_detection_model = Yolov8DetectionModel(
            model=yolo_model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(yolov8_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.models.yolov8 import Yolov8DetectionModel

        # init model
        download_yolov8n_model()

        yolov8_detection_model = Yolov8DetectionModel(
            model_path=Yolov8TestConstants.YOLOV8N_MODEL_PATH,
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
        yolov8_detection_model.perform_inference(image)
        original_predictions = yolov8_detection_model.original_predictions

        boxes = original_predictions

        # find box of first car detection with conf greater than 0.5
        for box in boxes[0]:
            if box[5].item() == 2:  # if category car
                if box[4].item() > 0.5:
                    break

        # compare
        desired_bbox = [448, 309, 497, 342]
        predicted_bbox = list(map(int, box[:4].tolist()))
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(len(yolov8_detection_model.category_names), 80)
        for box in boxes[0]:
            self.assertGreaterEqual(box[4].item(), CONFIDENCE_THRESHOLD)

    def test_convert_original_predictions(self):
        from sahi.models.yolov8 import Yolov8DetectionModel

        # init model
        download_yolov8n_model()

        yolov8_detection_model = Yolov8DetectionModel(
            model_path=Yolov8TestConstants.YOLOV8N_MODEL_PATH,
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
        yolov8_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        yolov8_detection_model.convert_original_predictions()
        object_prediction_list = yolov8_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 11)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        desired_bbox = [448, 309, 49, 33]
        predicted_bbox = object_prediction_list[0].bbox.to_xywh()
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(object_prediction_list[2].category.id, 2)
        self.assertEqual(object_prediction_list[2].category.name, "car")
        desired_bbox = [835, 307, 37, 37]
        predicted_bbox = object_prediction_list[2].bbox.to_xywh()
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin

        for object_prediction in object_prediction_list:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
