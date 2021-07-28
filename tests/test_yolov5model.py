# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.yolov5 import Yolov5TestConstants, download_yolov5s6_model


class TestYolov5DetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.model import Yolov5DetectionModel

        download_yolov5s6_model()

        yolov5_detection_model = Yolov5DetectionModel(
            model_path=Yolov5TestConstants.YOLOV5S6_MODEL_PATH,
            confidence_threshold=0.3,
            device=None,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(yolov5_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.model import Yolov5DetectionModel

        # init model
        download_yolov5s6_model()

        yolov5_detection_model = Yolov5DetectionModel(
            model_path=Yolov5TestConstants.YOLOV5S6_MODEL_PATH,
            confidence_threshold=0.5,
            device=None,
            category_remapping=None,
            load_at_init=True,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        yolov5_detection_model.perform_inference(image)
        original_predictions = yolov5_detection_model.original_predictions

        boxes = original_predictions.xyxy

        # find box of first car detection with conf greater than 0.5
        for box in boxes[0]:
            if box[5].item() == 2:  # if category car
                if box[4].item() > 0.5:
                    break

        # compare
        desired_bbox = [321, 322, 383, 362]
        predicted_bbox = list(map(int, box[:4].tolist()))
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(len(original_predictions.names), 80)

    def test_convert_original_predictions(self):
        from sahi.model import Yolov5DetectionModel

        # init model
        download_yolov5s6_model()

        yolov5_detection_model = Yolov5DetectionModel(
            model_path=Yolov5TestConstants.YOLOV5S6_MODEL_PATH,
            confidence_threshold=0.5,
            device=None,
            category_remapping=None,
            load_at_init=True,
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
        self.assertEqual(len(object_prediction_list), 14)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        desired_bbox = [321, 322, 62, 40]
        predicted_bbox = object_prediction_list[0].bbox.to_coco_bbox()
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(object_prediction_list[5].category.id, 2)
        self.assertEqual(object_prediction_list[5].category.name, "car")
        self.assertEqual(
            object_prediction_list[5].bbox.to_coco_bbox(), [617, 195, 24, 23],
        )

    def test_create_original_predictions_from_object_prediction_list(self,):
        pass
        # TODO: implement object_prediction_list to yolov5 format conversion


if __name__ == "__main__":
    unittest.main()
