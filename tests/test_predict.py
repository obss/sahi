# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import shutil
import sys
import unittest

import numpy as np
import pytest

from sahi.utils.cv import read_image
from sahi.utils.ultralytics import UltralyticsTestConstants, download_yolo11n_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320


class TestPredict(unittest.TestCase):
    def test_prediction_score(self):
        from sahi.prediction import PredictionScore

        prediction_score = PredictionScore(np.array(0.6))  # type: ignore
        self.assertEqual(type(prediction_score.value), float)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.5), True)
        self.assertEqual(prediction_score.is_greater_than_threshold(0.7), False)

    @pytest.mark.skipif(sys.version_info[:2] > (3, 10), reason="Requires Python 3.10 or lower")
    def test_get_prediction_mmdet(self):
        from sahi.models.mmdet import MmdetDetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_yolox_tiny_model

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            image_size=IMAGE_SIZE,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=mmdet_detection_model, shift_amount=[0, 0], full_shape=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 2)
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
        self.assertEqual(num_car, 2)

    def test_get_prediction_automodel_yolo11(self):
        from sahi.auto_model import AutoDetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.ultralytics import UltralyticsTestConstants, download_yolo11n_model

        # init model
        download_yolo11n_model()

        yolo11_detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolo11_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=yolo11_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertGreater(len(object_prediction_list), 0)
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
        self.assertGreater(num_car, 0)

    @pytest.mark.skipif(sys.version_info[:2] > (3, 10), reason="Requires Python 3.10 or lower")
    def test_get_sliced_prediction_mmdet(self):
        from sahi.models.mmdet import MmdetDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_yolox_tiny_model

        # init model
        download_mmdet_yolox_tiny_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=mmdet_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_threshold=match_threshold,
            postprocess_match_metric=match_metric,
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
        self.assertEqual(num_truck, 0)
        num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_car, 15)

    def test_get_prediction_yolo11(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel
        from sahi.predict import get_prediction

        # init model
        download_yolo11n_model()

        yolo11_detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolo11_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image, detection_model=yolo11_detection_model, shift_amount=[0, 0], full_shape=None, postprocess=None
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertGreater(len(object_prediction_list), 0)
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
        self.assertGreater(num_car, 0)

    def test_get_sliced_prediction_yolo11(self):
        from sahi.models.ultralytics import UltralyticsDetectionModel
        from sahi.predict import get_sliced_prediction

        # init model
        download_yolo11n_model()

        yolo11_detection_model = UltralyticsDetectionModel(
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        yolo11_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=yolo11_detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,
            postprocess_type=postprocess_type,
            postprocess_match_threshold=match_threshold,
            postprocess_match_metric=match_metric,
            postprocess_class_agnostic=class_agnostic,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertGreater(len(object_prediction_list), 0)
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
        self.assertGreater(num_car, 0)

    @pytest.mark.skipif(sys.version_info[:2] > (3, 10), reason="Requires Python 3.10 or lower")
    def test_mmdet_yolox_tiny_prediction(self):
        from sahi.predict import predict
        from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_yolox_tiny_model

        # init model
        download_mmdet_yolox_tiny_model()

        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # prepare paths
        dataset_json_path = "tests/data/coco_utils/terrain_all_coco.json"
        source = "tests/data/coco_utils/"
        project_dir = "tests/data/predict_result"

        # get sliced prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="mmdet",
            model_path=MmdetTestConstants.MMDET_YOLOX_TINY_MODEL_PATH,
            model_config_path=MmdetTestConstants.MMDET_YOLOX_TINY_CONFIG_PATH,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
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
            novisual=True,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=dataset_json_path,
            project=project_dir,
            name="exp",
            verbose=1,
        )

    def test_ultralytics_yolo11n_prediction(self):
        from sahi.predict import predict
        from sahi.utils.ultralytics import UltralyticsTestConstants, download_yolo11n_model

        # init model
        download_yolo11n_model()

        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # prepare paths
        dataset_json_path = "tests/data/coco_utils/terrain_all_coco.json"
        source = "tests/data/coco_utils/"
        project_dir = "tests/data/predict_result"

        # get sliced prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
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
            novisual=True,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=dataset_json_path,
            project=project_dir,
            name="exp",
            verbose=1,
        )

    def test_video_prediction(self):
        from os import path

        from sahi.predict import predict
        from sahi.utils.file import download_from_url

        # download video file
        source_url = "https://github.com/obss/sahi/releases/download/0.9.2/test.mp4"
        destination_path = "tests/data/test.mp4"
        if not path.exists(destination_path):
            download_from_url(source_url, destination_path)

        # init model
        download_yolo11n_model()

        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        image_size = 320
        class_agnostic = True

        # prepare paths
        source = destination_path
        project_dir = "tests/data/predict_result"

        # get sliced inference from video input without exporting visual
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=False,
            no_standard_prediction=True,
            slice_height=512,
            slice_width=512,
            image_size=image_size,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            novisual=True,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=None,
            project=project_dir,
            name="exp",
            verbose=1,
        )

        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        image_size = 320
        class_agnostic = True

        # get standard inference from video input without exporting visual
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=True,
            no_standard_prediction=False,
            image_size=image_size,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            novisual=True,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=None,
            project=project_dir,
            name="exp",
            verbose=1,
        )

        # get standard inference from video input and export visual
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        image_size = 320
        class_agnostic = True

        # get full sized prediction
        if os.path.isdir(project_dir):
            shutil.rmtree(project_dir, ignore_errors=True)
        predict(
            model_type="ultralytics",
            model_path=UltralyticsTestConstants.YOLO11N_MODEL_PATH,
            model_config_path=None,
            model_confidence_threshold=CONFIDENCE_THRESHOLD,
            model_device=MODEL_DEVICE,
            model_category_mapping=None,
            model_category_remapping=None,
            source=source,
            no_sliced_prediction=True,
            no_standard_prediction=False,
            image_size=image_size,
            postprocess_type=postprocess_type,
            postprocess_match_metric=match_metric,
            postprocess_match_threshold=match_threshold,
            postprocess_class_agnostic=class_agnostic,
            export_pickle=False,
            export_crop=False,
            dataset_json_path=None,
            project=project_dir,
            name="exp",
            verbose=1,
        )


if __name__ == "__main__":
    unittest.main()
