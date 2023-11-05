# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

from sahi.utils.cv import read_image
from sahi.utils.yolonas import YoloNasTestConstants, download_yolonas_s_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640
PRETRAINED_WEIGHTS = "coco"
YOLONAS_TEST_MODEL_NAME = "yolo_nas_s"
CLASS_NAMES_YAML_PATH = "tests/data/coco_utils/coco_class_names.yaml"
TEST_IMAGE_PATH = "tests/data/small-vehicles1.jpeg"


class TestYoloNasDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.yolonas import YoloNasDetectionModel

        download_yolonas_s_model()

        yolonas_detection_model = YoloNasDetectionModel(
            model_name=YOLONAS_TEST_MODEL_NAME,
            model_path=YoloNasTestConstants.YOLONAS_S_MODEL_PATH,
            class_names_yaml_path=CLASS_NAMES_YAML_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        self.assertNotEqual(yolonas_detection_model.model, None)

    def test_set_model(self):
        from super_gradients.training import models

        from sahi.models.yolonas import YoloNasDetectionModel

        download_yolonas_s_model()

        yolonas_model = models.get(model_name=YOLONAS_TEST_MODEL_NAME, pretrained_weights=PRETRAINED_WEIGHTS)

        yolonas_detection_model = YoloNasDetectionModel(
            model=yolonas_model,
            model_name=YOLONAS_TEST_MODEL_NAME,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(yolonas_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.models.yolonas import YoloNasDetectionModel

        # init model
        download_yolonas_s_model()

        yolonas_detection_model = YoloNasDetectionModel(
            model_name=YOLONAS_TEST_MODEL_NAME,
            model_path=YoloNasTestConstants.YOLONAS_S_MODEL_PATH,
            class_names_yaml_path=CLASS_NAMES_YAML_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image = read_image(TEST_IMAGE_PATH)

        # perform inference
        yolonas_detection_model.perform_inference(image)
        original_predictions = yolonas_detection_model.original_predictions

        pred = original_predictions[0].prediction

        # find box of first car detection with conf greater than 0.5
        for box, score, label in zip(pred.bboxes_xyxy, pred.confidence, pred.labels):
            if int(label) == 2:  # if category car
                if score > 0.5:
                    break

        # compare
        desired_bbox = [447, 309, 495, 341]
        predicted_bbox = list(map(int, box[:4].tolist()))
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(len(yolonas_detection_model.category_names), 80)
        self.assertGreaterEqual(score, CONFIDENCE_THRESHOLD)

    def test_convert_original_predictions(self):
        from sahi.models.yolonas import YoloNasDetectionModel

        # init model
        download_yolonas_s_model()

        yolonas_detection_model = YoloNasDetectionModel(
            model_name=YOLONAS_TEST_MODEL_NAME,
            model_path=YoloNasTestConstants.YOLONAS_S_MODEL_PATH,
            class_names_yaml_path=CLASS_NAMES_YAML_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image = read_image(TEST_IMAGE_PATH)

        # get raw predictions for reference
        original_results = list(yolonas_detection_model.model.predict(TEST_IMAGE_PATH, conf=CONFIDENCE_THRESHOLD))[
            0
        ].prediction
        num_results = len(original_results.bboxes_xyxy)

        # perform inference
        yolonas_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        yolonas_detection_model.convert_original_predictions()
        object_prediction_list = yolonas_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), num_results)

        # loop through predictions and check that they are equal
        for i in range(num_results):
            desired_bbox = [
                original_results.bboxes_xyxy[i][0],
                original_results.bboxes_xyxy[i][1],
                original_results.bboxes_xyxy[i][2],
                original_results.bboxes_xyxy[i][3],
            ]
            desired_cat_id = int(original_results.labels[i])
            self.assertEqual(object_prediction_list[i].category.id, desired_cat_id)
            predicted_bbox = object_prediction_list[i].bbox.to_xyxy()
            margin = 2
            for ind, point in enumerate(predicted_bbox):
                assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        for object_prediction in object_prediction_list:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
