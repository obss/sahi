# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.yolov8 import Yolov8TestConstants, download_yolov8n_model, download_yolov8n_seg_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640


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

        boxes = original_predictions[0].data

        # find box of first car detection with conf greater than 0.5
        for box in boxes:
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
        for box in boxes:
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

        # get raw predictions for reference
        original_results = yolov8_detection_model.model.predict(image_path, conf=CONFIDENCE_THRESHOLD)[0].boxes
        num_results = len(original_results)

        # perform inference
        yolov8_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        yolov8_detection_model.convert_original_predictions()
        object_prediction_list = yolov8_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), num_results)

        # loop through predictions and check that they are equal
        for i in range(num_results):
            desired_bbox = [
                original_results[i].xyxy[0][0],
                original_results[i].xyxy[0][1],
                original_results[i].xywh[0][2],
                original_results[i].xywh[0][3],
            ]
            desired_cat_id = int(original_results[i].cls[0])
            self.assertEqual(object_prediction_list[i].category.id, desired_cat_id)
            predicted_bbox = object_prediction_list[i].bbox.to_xywh()
            margin = 2
            for ind, point in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        for object_prediction in object_prediction_list:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

    def test_perform_inference_with_mask_output(self):
        from sahi.models.yolov8 import Yolov8DetectionModel

        # init model
        download_yolov8n_seg_model()

        yolov8_detection_model = Yolov8DetectionModel(
            model_path=Yolov8TestConstants.YOLOV8N_SEG_MODEL_PATH,
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
        boxes = original_predictions[0][0]
        masks = original_predictions[0][1]

        # find box of first car detection with conf greater than 0.5
        for box in boxes:
            if box[5].item() == 2:  # if category car
                if box[4].item() > 0.5:
                    break

        # compare
        desired_bbox = [320, 323, 380, 365]
        predicted_bbox = list(map(int, box[:4].tolist()))
        margin = 3
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(len(yolov8_detection_model.category_names), 80)
        for box in boxes:
            self.assertGreaterEqual(box[4].item(), CONFIDENCE_THRESHOLD)
        self.assertEqual(masks.shape, (12, 352, 640))
        self.assertEqual(masks.shape[0], len(boxes))

    def test_convert_original_predictions_with_mask_output(self):
        from sahi.models.yolov8 import Yolov8DetectionModel

        # init model
        download_yolov8n_seg_model()

        yolov8_detection_model = Yolov8DetectionModel(
            model_path=Yolov8TestConstants.YOLOV8N_SEG_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get raw predictions for reference
        original_results = yolov8_detection_model.model.predict(image_path, conf=CONFIDENCE_THRESHOLD)[0].boxes
        num_results = len(original_results)

        # perform inference
        yolov8_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        yolov8_detection_model.convert_original_predictions(full_shape=(image.shape[0], image.shape[1]))
        object_prediction_list = yolov8_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), num_results)

        # loop through predictions and check that they are equal
        for i in range(num_results):
            desired_bbox = [
                original_results[i].xyxy[0][0],
                original_results[i].xyxy[0][1],
                original_results[i].xywh[0][2],
                original_results[i].xywh[0][3],
            ]
            desired_cat_id = int(original_results[i].cls[0])
            self.assertEqual(object_prediction_list[i].category.id, desired_cat_id)
            predicted_bbox = object_prediction_list[i].bbox.to_xywh()
            margin = 20  # Margin high because for some reason some original predictions are really poor
            for ind, point in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        for object_prediction in object_prediction_list:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
