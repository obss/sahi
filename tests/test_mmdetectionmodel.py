# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.mmdet import MmdetTestConstants, download_mmdet_cascade_mask_rcnn_model, download_mmdet_retinanet_model

MODEL_DEVICE = "cpu"


class TestMmdetDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.model import MmdetDetectionModel

        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            confidence_threshold=0.3,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(mmdet_detection_model.model, None)

    def test_perform_inference_with_mask_output(self):
        from sahi.model import MmdetDetectionModel

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            confidence_threshold=0.5,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions

        boxes = original_predictions[0][0]
        masks = original_predictions[0][1]

        # find box of first person detection with conf greater than 0.5
        for box in boxes[0]:
            print(len(box))
            if len(box) == 5:
                if box[4] > 0.5:
                    break

        # compare
        self.assertEqual(box[:4].astype("int").tolist(), [1019, 417, 1027, 437])
        self.assertEqual(len(boxes), 80)
        self.assertEqual(len(masks), 80)

    def test_perform_inference_without_mask_output(self):
        from sahi.model import MmdetDetectionModel

        # init model
        download_mmdet_retinanet_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_RETINANET_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_RETINANET_CONFIG_PATH,
            confidence_threshold=0.5,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions

        boxes = original_predictions[0]

        # find box of first car detection with conf greater than 0.5
        for box in boxes[2]:
            print(len(box))
            if len(box) == 5:
                if box[4] > 0.5:
                    break

        # compare
        self.assertEqual(box[:4].astype("int").tolist(), [448, 309, 495, 341])
        self.assertEqual(len(boxes), 80)

    def test_convert_original_predictions_with_mask_output(self):
        from sahi.model import MmdetDetectionModel

        # init model
        download_mmdet_cascade_mask_rcnn_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_CASCADEMASKRCNN_CONFIG_PATH,
            confidence_threshold=0.5,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        mmdet_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        mmdet_detection_model.convert_original_predictions()
        object_prediction_list = mmdet_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 44)
        self.assertEqual(object_prediction_list[0].category.id, 0)
        self.assertEqual(object_prediction_list[0].category.name, "person")
        self.assertEqual(
            object_prediction_list[0].bbox.to_coco_bbox(),
            [1020, 419, 6, 17],
        )
        self.assertEqual(object_prediction_list[1].category.id, 2)
        self.assertEqual(object_prediction_list[1].category.name, "car")
        self.assertEqual(
            object_prediction_list[1].bbox.to_coco_bbox(),
            [449, 311, 45, 29],
        )
        self.assertEqual(object_prediction_list[5].category.id, 2)
        self.assertEqual(object_prediction_list[5].category.name, "car")
        self.assertEqual(
            object_prediction_list[2].bbox.to_coco_bbox(),
            [657, 204, 13, 10],
        )

    def test_convert_original_predictions_without_mask_output(self):
        from sahi.model import MmdetDetectionModel

        # init model
        download_mmdet_retinanet_model()

        mmdet_detection_model = MmdetDetectionModel(
            model_path=MmdetTestConstants.MMDET_RETINANET_MODEL_PATH,
            config_path=MmdetTestConstants.MMDET_RETINANET_CONFIG_PATH,
            confidence_threshold=0.5,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        mmdet_detection_model.perform_inference(image, image_size=256)

        # convert predictions to ObjectPrediction list
        mmdet_detection_model.convert_original_predictions()
        object_prediction_list = mmdet_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 100)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        self.assertEqual(
            object_prediction_list[0].bbox.to_coco_bbox(),
            [317, 313, 67, 50],
        )
        self.assertEqual(object_prediction_list[5].category.id, 2)
        self.assertEqual(object_prediction_list[5].category.name, "car")
        self.assertEqual(
            object_prediction_list[5].bbox.to_coco_bbox(),
            [22, 275, 93, 55],
        )


if __name__ == "__main__":
    unittest.main()
