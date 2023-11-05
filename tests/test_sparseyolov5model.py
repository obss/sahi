# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import platform
import unittest
from decimal import Decimal

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.sparseyolov5 import Yolov5TestConstants

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 320

if platform.system() == "Linux":

    class TestSparseYolov5DetectionModel(unittest.TestCase):
        def test_load_model(self):
            from deepsparse import Pipeline

            from sahi.models.yolov5sparse import Yolov5SparseDetectionModel

            yolo_model = Pipeline.create(task="yolo", model_path=Yolov5TestConstants.YOLOV_MODEL_URL)

            yolov5_detection_model = Yolov5SparseDetectionModel(
                model=yolo_model,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                device=MODEL_DEVICE,
                category_remapping=None,
                load_at_init=True,
            )

            self.assertNotEqual(yolov5_detection_model.model, None)

        def test_set_model(self):
            from deepsparse import Pipeline

            from sahi.models.yolov5sparse import Yolov5SparseDetectionModel

            yolo_model = Pipeline.create(task="yolo", model_path=Yolov5TestConstants.YOLOV_MODEL_URL)

            yolov5_detection_model = Yolov5SparseDetectionModel(
                model=yolo_model,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                device=MODEL_DEVICE,
                category_remapping=None,
                load_at_init=True,
            )

            self.assertNotEqual(yolov5_detection_model.model, None)

        def test_perform_inference(self):
            from deepsparse import Pipeline

            from sahi.models.yolov5sparse import Yolov5SparseDetectionModel

            # init model
            yolo_model = Pipeline.create(task="yolo", model_path=Yolov5TestConstants.YOLOV_MODEL_URL)

            yolov5_detection_model = Yolov5SparseDetectionModel(
                model=yolo_model,
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
            yolov5_detection_model.perform_inference(image)
            original_predictions = yolov5_detection_model.original_predictions

            boxes = original_predictions.boxes

            # find box of first car detection with conf greater than 0.5
            for image_ind, (prediction_bboxes, prediction_scores, prediction_categories) in enumerate(
                original_predictions
            ):
                if int(Decimal(prediction_categories[0])) == 2:  # if category car
                    if prediction_scores[0] > 0.5:
                        break

            # compare
            desired_bbox = [321, 322, 384, 362]
            predicted_bbox = boxes[0][0]
            margin = 2
            for ind, point in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
            for box in boxes[0]:
                self.assertGreaterEqual(box[0], CONFIDENCE_THRESHOLD)

        def test_convert_original_predictions(self):
            from deepsparse import Pipeline

            from sahi.models.yolov5sparse import Yolov5SparseDetectionModel

            # init model
            yolo_model = Pipeline.create(task="yolo", model_path=Yolov5TestConstants.YOLOV_MODEL_URL)

            yolov5_detection_model = Yolov5SparseDetectionModel(
                model=yolo_model,
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
            yolov5_detection_model.perform_inference(image)

            # convert predictions to ObjectPrediction list
            yolov5_detection_model.convert_original_predictions()
            object_prediction_list = yolov5_detection_model.object_prediction_list

            # compare
            self.assertEqual(len(object_prediction_list), 16)
            self.assertEqual(object_prediction_list[0].category.id, 2)
            self.assertEqual(object_prediction_list[0].category.name, "car")
            desired_bbox = [321, 322, 63, 40]
            predicted_bbox = object_prediction_list[0].bbox.to_xywh()
            margin = 2
            for ind, point in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
            self.assertEqual(object_prediction_list[2].category.id, 2)
            self.assertEqual(object_prediction_list[2].category.name, "car")
            desired_bbox = [700, 234, 22, 17]
            predicted_bbox = object_prediction_list[2].bbox.to_xywh()
            for ind, point in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
            for object_prediction in object_prediction_list:
                self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
