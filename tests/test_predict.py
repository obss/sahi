# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import shutil
import unittest

import numpy as np

from sahi.utils.cv import read_image

MODEL_DEVICE = "cpu"


class TestPredict(unittest.TestCase):
    def test_prediction_score(self):
        from sahi.prediction import PredictionScore

        prediction_score = PredictionScore(np.array(0.6))
        self.assertEqual(type(prediction_score.value), float)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.5), True)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.7), False)

    def test_object_prediction(self):
        from sahi.prediction import ObjectPrediction

    def test_get_prediction_mmdet(self):
        from sahi.model import MmdetDetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_cascade_mask_rcnn_model

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            confidence_threshold=0.3,
            device=MODEL_DEVICE,
            category_remapping=None,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=mmdet_detection_model, shift_amount=[0, 0], full_shape=None, image_size=320
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 3)
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
        self.assertEqual(num_car, 3)

    def test_get_prediction_yolov5(self):
        from sahi.model import Yolov5DetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.yolov5 import Yolov5TestConstants, download_yolov5n_model

        # init model
        download_yolov5n_model()

        yolov5_detection_model = Yolov5DetectionModel(
            model_path=Yolov5TestConstants.YOLOV5N_MODEL_PATH,
            confidence_threshold=0.3,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
        )
        yolov5_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=yolov5_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 15)
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
        self.assertEqual(num_car, 15)

    def test_get_sliced_prediction_mmdet(self):
        from sahi.model import MmdetDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_cascade_mask_rcnn_model

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            confidence_threshold=0.3,
            device=MODEL_DEVICE,
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
        postprocess_type = "UNIONMERGE"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True
        image_size = 320

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            image_size=image_size,
            detection_model=mmdet_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 15)
        num_person = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
        self.assertEqual(num_person, 0)
        num_truck = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "truck":
                num_truck += 1
        self.assertEqual(num_truck, 1)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 14)

    def test_get_sliced_prediction_yolov5(self):
        from sahi.model import Yolov5DetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.yolov5 import Yolov5TestConstants, download_yolov5n_model

        # init model
        download_yolov5n_model()

        yolov5_detection_model = Yolov5DetectionModel(
            model_path=Yolov5TestConstants.YOLOV5N_MODEL_PATH,
            confidence_threshold=0.3,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
        )
        yolov5_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "UNIONMERGE"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=yolov5_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list

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
                num_truck += 2
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 19)

    def test_coco_json_prediction(self):
        from sahi.predict import predict
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_cascade_mask_rcnn_model
        from sahi.utils.yolov5 import Yolov5TestConstants, download_yolov5n_model

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        postprocess_type = "UNIONMERGE"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # prepare paths
        dataset_json_path = "tests/data/coco_utils/terrain_all_coco.json"
        source = "tests/data/coco_utils/"
        project_dir = "tests/data/predict_result"

        # get full sized prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir)
        predict(
            model_type="mmdet",
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            model_config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            model_confidence_threshold=0.4,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=False,
            no_standard_prediction=True,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            export_visual=False,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=dataset_json_path,
            project=project_dir,
            name="exp",
            verbose=1,
        )

        # init model
        download_yolov5n_model()

        # prepare paths
        dataset_json_path = "tests/data/coco_utils/terrain_all_coco.json"
        source = "tests/data/coco_utils/"
        project_dir = "tests/data/predict_result"

        # get full sized prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir)
        predict(
            model_type="yolov5",
            model_path=Yolov5TestConstants.YOLOV5N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=0.4,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=False,
            no_standard_prediction=True,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            export_visual=False,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=dataset_json_path,
            project=project_dir,
            name="exp",
            verbose=1,
        )


if __name__ == "__main__":
    unittest.main()
