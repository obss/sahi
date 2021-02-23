# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import shutil
import unittest

import numpy as np
from sahi.utils.cv import read_image
from sahi.utils.file import list_files


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

        from tests.test_utils import (
            MmdetTestConstants,
            download_mmdet_cascade_mask_rcnn_model,
        )

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
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
            full_shape=None,
            merger=None,
            matcher=None,
        )
        object_prediction_list = prediction_result["object_prediction_list"]

        # compare
        self.assertEqual(len(object_prediction_list), 19)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 19)

    def test_get_sliced_prediction(self):
        from sahi.model import MmdetDetectionModel
        from sahi.predict import get_sliced_prediction

        from tests.test_utils import (
            MmdetTestConstants,
            download_mmdet_cascade_mask_rcnn_model,
        )

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            prediction_score_threshold=0.3,
            device=None,
            category_remapping=None,
            load_at_init=False,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        match_iou_threshold = 0.5

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=mmdet_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            match_iou_threshold=match_iou_threshold,
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

    def test_coco_json_prediction(self):
        from sahi.model import MmdetDetectionModel
        from sahi.predict import predict

        from tests.test_utils import (
            MmdetTestConstants,
            download_mmdet_cascade_mask_rcnn_model,
        )

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        model_parameters = {
            "model_path": MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            "config_path": MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            "prediction_score_threshold": 0.4,
            "device": None,  # cpu or cuda
            "category_mapping": None,
            "category_remapping": None,  # {"0": 1, "1": 2, "2": 3}
        }

        # prepare paths
        coco_file_path = "tests/data/coco_utils/terrain_all_coco.json"
        source = "tests/data/coco_utils/"
        project_dir = "tests/data/predict_result"

        # get full sized prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir)
        predict(
            model_name="MmdetDetectionModel",
            model_parameters=model_parameters,
            source=source,
            apply_sliced_prediction=True,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            match_iou_threshold=0.5,
            export_pickle=False,
            export_crop=False,
            coco_file_path=coco_file_path,
            project=project_dir,
            name="exp",
            verbose=1,
        )


if __name__ == "__main__":
    unittest.main()
