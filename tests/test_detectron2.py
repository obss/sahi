# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

from sahi.model import Detectron2DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.detectron2 import Detectron2TestConstants

MODEL_DEVICE = "cpu"


class TestDetectron2DetectionModel(unittest.TestCase):
    def test_load_model(self):

        detector2_detection_model = Detectron2DetectionModel(
            model_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
            config_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
            confidence_threshold=0.5,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )
        self.assertNotEqual(detector2_detection_model.model, None)

    def test_perform_inference_without_mask_output(self):

        detectron2_detection_model = Detectron2DetectionModel(
            model_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
            config_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
            confidence_threshold=0.5,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=320,
        )
        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        detectron2_detection_model.perform_inference(image)
        original_predictions = detectron2_detection_model.original_predictions

        boxes = original_predictions["instances"].pred_boxes.tensor.cpu().numpy()
        scores = original_predictions["instances"].scores.cpu().numpy()
        category_ids = original_predictions["instances"].pred_classes.cpu().numpy()

        # find box of first car detection with conf greater than 0.5
        for ind, box in enumerate(boxes):
            if category_ids[ind] == 2 and scores[ind] > 0.5:
                break

        # compare
        self.assertEqual(boxes[ind].astype("int").tolist(), [320, 324, 381, 364])
        self.assertEqual(len(boxes), 10)

    def test_convert_original_predictions_without_mask_output(self):

        detectron2_detection_model = Detectron2DetectionModel(
            model_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
            config_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
            confidence_threshold=0.5,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=320,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        detectron2_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        detectron2_detection_model.convert_original_predictions()
        object_prediction_list = detectron2_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 10)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        predicted_bbox = object_prediction_list[0].bbox.to_coco_bbox()
        desired_bbox = [320, 324, 61, 40]
        margin = 5
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin, print(
                predicted_bbox, desired_bbox
            )

        self.assertEqual(object_prediction_list[5].category.id, 2)
        self.assertEqual(object_prediction_list[5].category.name, "car")
        predicted_bbox = object_prediction_list[2].bbox.to_coco_bbox()
        desired_bbox = [383, 275, 35, 33]
        margin = 5
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin, print(
                predicted_bbox, desired_bbox
            )

    def test_convert_original_predictions_with_mask_output(self):

        detectron2_detection_model = Detectron2DetectionModel(
            model_path=Detectron2TestConstants.MASKRCNN_MODEL_ZOO_NAME,
            config_path=Detectron2TestConstants.MASKRCNN_MODEL_ZOO_NAME,
            confidence_threshold=0.5,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=320,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        detectron2_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        detectron2_detection_model.convert_original_predictions()
        object_prediction_list = detectron2_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 10)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        predicted_bbox = object_prediction_list[0].bbox.to_coco_bbox()
        desired_bbox = [322, 325, 57, 38]
        margin = 5
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin, print(
                predicted_bbox, desired_bbox
            )

        self.assertEqual(object_prediction_list[5].category.id, 2)
        self.assertEqual(object_prediction_list[5].category.name, "car")
        predicted_bbox = object_prediction_list[5].bbox.to_coco_bbox()
        desired_bbox = [697, 226, 29, 29]
        margin = 5
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin, print(
                predicted_bbox, desired_bbox
            )


if __name__ == "__main__":
    unittest.main()
