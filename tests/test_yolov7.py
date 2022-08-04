# OBSS SAHI Tool
# Code written by Kadir Nar, 2022.

import unittest

from sahi.utils.cv import read_image
from sahi.utils.yolov7 import Yolov7TestConstants, download_yolov7_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 320


class TestYolov7DetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.model import Yolov7DetectionModel

        download_yolov7_model()

        yolov7_detection_model = Yolov7DetectionModel(
            model_path=Yolov7TestConstants.YOLOV7_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
        )

        self.assertNotEqual(yolov7_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.model import Yolov7DetectionModel

        # init model
        download_yolov7_model()

        yolov7_detection_model = Yolov7DetectionModel(
            model_path=Yolov7TestConstants.YOLOV7_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        yolov7_detection_model.perform_inference(image)
        original_predictions = yolov7_detection_model.original_predictions
        for _, image_predictions_in_xyxy_format in enumerate(original_predictions.xyxy):
            for pred in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1, y1, x2, y2 = (
                    int(pred[0]),
                    int(pred[1]),
                    int(pred[2]),
                    int(pred[3]),
                )

        desidred_bbox = [521, 228, 543, 243]
        predicted_bbox = [x1, y1, x2, y2]
        self.assertEqual(desidred_bbox, predicted_bbox)

    def test_convert_original_predictions(self):
        from sahi.model import Yolov7DetectionModel

        # init model
        download_yolov7_model()

        yolov7_detection_model = Yolov7DetectionModel(
            model_path=Yolov7TestConstants.YOLOV7_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        yolov7_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        yolov7_detection_model.convert_original_predictions()
        object_prediction_list = yolov7_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 12)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        desidred_bbox = [322, 323, 61, 41]
        predicted_bbox = object_prediction_list[0].bbox.to_coco_bbox()
        self.assertEqual(desidred_bbox, predicted_bbox)


if __name__ == "__main__":
    unittest.main()
