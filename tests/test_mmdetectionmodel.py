import unittest

import numpy as np
from sahi.utils.cv import read_image

from tests.test_utils import (
    download_mmdet_cascade_mask_rcnn_model,
    mmdet_cascade_mask_rcnn_config_path,
    mmdet_cascade_mask_rcnn_model_path,
)


class TestMmdetDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.model import MmdetDetectionModel

        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=mmdet_cascade_mask_rcnn_model_path,
            config_path=mmdet_cascade_mask_rcnn_config_path,
            prediction_score_threshold=0.3,
            device=None,
            category_remapping=None,
        )
        mmdet_detection_model.load_model()
        self.assertNotEqual(mmdet_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.model import MmdetDetectionModel
        from sahi.prediction import PredictionInput

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=mmdet_cascade_mask_rcnn_model_path,
            config_path=mmdet_cascade_mask_rcnn_config_path,
            prediction_score_threshold=0.5,
            device=None,
            category_remapping=None,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions

        boxes = original_predictions[0]
        masks = original_predictions[1]

        # find box of first person detection with conf greater than 0.5
        for box in boxes[0]:
            print(len(box))
            if len(box) == 5:
                if box[4] > 0.5:
                    break

        # compare
        self.assertEqual(box[:4].astype("int").tolist(), [336, 123, 346, 139])
        self.assertEqual(len(boxes), 80)
        self.assertEqual(len(masks), 80)

    def test_convert_original_predictions(self):
        from sahi.model import MmdetDetectionModel
        from sahi.prediction import PredictionInput

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=mmdet_cascade_mask_rcnn_model_path,
            config_path=mmdet_cascade_mask_rcnn_config_path,
            prediction_score_threshold=0.5,
            device=None,
            category_remapping=None,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        mmdet_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        mmdet_detection_model.convert_original_predictions()
        object_prediction_list_w_category_mapping = (
            mmdet_detection_model.object_prediction_list
        )

        # compare
        self.assertEqual(len(object_prediction_list_w_category_mapping), 53)
        self.assertEqual(object_prediction_list_w_category_mapping[0].category.id, 0)
        self.assertEqual(
            object_prediction_list_w_category_mapping[0].category.name, "person"
        )
        self.assertEqual(
            object_prediction_list_w_category_mapping[0].bbox.to_coco_bbox(),
            [337, 124, 8, 14],
        )
        self.assertEqual(object_prediction_list_w_category_mapping[1].category.id, 2)
        self.assertEqual(
            object_prediction_list_w_category_mapping[1].category.name, "car"
        )
        self.assertEqual(
            object_prediction_list_w_category_mapping[1].bbox.to_coco_bbox(),
            [657, 204, 13, 10],
        )
        self.assertEqual(object_prediction_list_w_category_mapping[5].category.id, 2)
        self.assertEqual(
            object_prediction_list_w_category_mapping[5].category.name, "car"
        )
        self.assertEqual(
            object_prediction_list_w_category_mapping[2].bbox.to_coco_bbox(),
            [760, 232, 20, 15],
        )

    def test_create_original_predictions_from_object_prediction_list(self):
        from sahi.model import MmdetDetectionModel
        from sahi.prediction import PredictionInput

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=mmdet_cascade_mask_rcnn_model_path,
            config_path=mmdet_cascade_mask_rcnn_config_path,
            prediction_score_threshold=0.5,
            device=None,
            category_remapping=None,
        )
        mmdet_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions_1 = mmdet_detection_model.original_predictions

        # convert predictions to ObjectPrediction list
        mmdet_detection_model.convert_original_predictions()
        object_prediction_list = mmdet_detection_model.object_prediction_list

        original_predictions_2 = mmdet_detection_model._create_original_predictions_from_object_prediction_list(
            object_prediction_list
        )

        # compare
        self.assertEqual(len(original_predictions_1), len(original_predictions_2))  # 2
        self.assertEqual(
            len(original_predictions_1[0]), len(original_predictions_2[0])
        )  # 80
        self.assertEqual(
            len(original_predictions_1[0][2]), len(original_predictions_2[0][2])
        )  # 25
        self.assertEqual(
            type(original_predictions_1[0]), type(original_predictions_2[0])
        )  # list
        self.assertEqual(
            original_predictions_1[0][2].dtype, original_predictions_2[0][2].dtype
        )  # float32
        self.assertEqual(
            original_predictions_1[0][0][0].dtype, original_predictions_2[0][0][0].dtype
        )  # float32
        self.assertEqual(
            original_predictions_1[1][0][0].dtype, original_predictions_2[1][0][0].dtype
        )  # bool
        self.assertEqual(
            len(original_predictions_1[0][0][0]), len(original_predictions_2[0][0][0])
        )  # 5
        self.assertEqual(
            len(original_predictions_1[0][1]), len(original_predictions_1[0][1])
        )  # 0
        self.assertEqual(
            original_predictions_1[0][1].shape, original_predictions_1[0][1].shape
        )  # (0, 5)


if __name__ == "__main__":
    unittest.main()
