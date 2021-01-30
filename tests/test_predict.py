import unittest

import numpy as np
from sahi.utils.cv import read_image


class TestPredict(unittest.TestCase):
    def test_prediction_score(self):
        from sahi.prediction import PredictionScore

        prediction_score = PredictionScore(np.array(0.6))
        self.assertEqual(type(prediction_score.score), float)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.5), True)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.7), False)

    def test_object_prediction(self):
        from sahi.prediction import ObjectPrediction

    def test_prediction_input(self):
        from sahi.prediction import PredictionInput

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # init prediction input
        prediction_input = PredictionInput(
            image_list=[image],
        )

        # compare
        self.assertEqual(
            len(prediction_input.shift_amount_list), len(prediction_input.image_list)
        )

    def test_get_prediction(self):
        from sahi.model import MmdetDetectionModel
        from sahi.predict import get_prediction
        from sahi.prediction import PredictionInput

        from tests.test_utils import (
            download_mmdet_cascade_mask_rcnn_model,
            mmdet_cascade_mask_rcnn_config_path,
            mmdet_cascade_mask_rcnn_model_path,
        )

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=mmdet_cascade_mask_rcnn_model_path,
            config_path=mmdet_cascade_mask_rcnn_config_path,
            prediction_score_threshold=0.3,
            device=None,
            category_remapping=None,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image,
            detection_model=mmdet_detection_model,
            shift_amount=[0, 0],
            full_image_size=None,
            merger=None,
            matcher=None,
        )
        object_prediction_list = prediction_result["object_prediction_list"]

        # compare
        self.assertEqual(len(object_prediction_list), 23)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 3)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 20)

    def test_get_sliced_prediction(self):
        from sahi.model import MmdetDetectionModel
        from sahi.predict import get_sliced_prediction

        from tests.test_utils import (
            download_mmdet_cascade_mask_rcnn_model,
            mmdet_cascade_mask_rcnn_config_path,
            mmdet_cascade_mask_rcnn_model_path,
        )

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=mmdet_cascade_mask_rcnn_model_path,
            config_path=mmdet_cascade_mask_rcnn_config_path,
            prediction_score_threshold=0.3,
            device=None,
            category_remapping=None,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=mmdet_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        object_prediction_list = prediction_result["object_prediction_list"]

        # compare
        self.assertEqual(len(object_prediction_list), 24)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 2
        self.assertEqual(num_truck, 4)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 22)


if __name__ == "__main__":
    unittest.main()
