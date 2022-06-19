# OBSS SAHI Tool
# Code written by Kadir Nar, 2022.

import unittest

from sahi.model import TensorflowHubDetectionModel
from sahi.utils.cv import read_image

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 320
EFFICIENTDET_URL = "https://tfhub.dev/tensorflow/efficientdet/d0/1"


class TestTensorflowHubDetectionModel(unittest.TestCase):
    def test_load_model(self):

        tensorflow_hub_model = TensorflowHubDetectionModel(
            model_path=EFFICIENTDET_URL,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )
        self.assertNotEqual(tensorflow_hub_model.model, None)

    def test_perform_inference(self):

        tensorflow_hub_model = TensorflowHubDetectionModel(
            model_path=EFFICIENTDET_URL,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)
        image_height, image_width = image.shape[0], image.shape[1]
        # perform inference

        tensorflow_hub_model.perform_inference(image)
        original_prediction = tensorflow_hub_model.original_predictions

        boxes = original_prediction["detection_boxes"][0]
        box = [float(box) for box in boxes[0].numpy()]
        x1, y1, x2, y2 = (
            int(box[1] * image_width),
            int(box[0] * image_height),
            int(box[3] * image_width),
            int(box[2] * image_height),
        )
        bbox = [x1, y1, x2, y2]
        # compare
        desidred_bbox = [317, 324, 381, 364]
        predicted_bbox = [x1, y1, x2, y2]
        self.assertEqual(desidred_bbox, predicted_bbox)

    def test_convert_original_predictions(self):

        tensorflow_hub_model = TensorflowHubDetectionModel(
            model_path=EFFICIENTDET_URL,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)
        image_height, image_width = image.shape[0], image.shape[1]

        # perform inference
        tensorflow_hub_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        tensorflow_hub_model.convert_original_predictions()
        object_prediction_list = tensorflow_hub_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 5)
        self.assertEqual(object_prediction_list[0].category.id, 3)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        desidred_bbox = [317, 324, 64, 40]
        predicted_bbox = object_prediction_list[0].bbox.to_coco_bbox()
        self.assertEqual(desidred_bbox, predicted_bbox)


if __name__ == "__main__":
    unittest.main()
