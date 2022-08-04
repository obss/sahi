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

    def test_set_model(self):
        import torch

        from sahi.model import Yolov7DetectionModel

        download_yolov7_model()
        yolov7_model = torch.hub.load('WongKinYiu/yolov7', 'custom', Yolov7TestConstants.YOLOV7_MODEL_PATH) 

        yolov7_detection_model = Yolov7DetectionModel(
            model=yolov7_model,
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

        boxes = original_predictions.xyxy

        # find box of first car detection with conf greater than 0.5
        for box in boxes[0]:
            if box[5].item() == 2:  # if category car
                if box[4].item() > 0.5:
                    break

        # compare
        desired_bbox = [321, 329, 378, 368]
        predicted_bbox = list(map(int, box[:4].tolist()))
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(len(original_predictions.names), 80)
        for box in boxes[0]:
            self.assertGreaterEqual(box[4].item(), CONFIDENCE_THRESHOLD)

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
        self.assertEqual(len(object_prediction_list), 3)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        desired_bbox = [321, 329, 57, 39]
        predicted_bbox = object_prediction_list[0].bbox.to_coco_bbox()
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(object_prediction_list[2].category.id, 2)
        self.assertEqual(object_prediction_list[2].category.name, "car")
        desired_bbox = [381, 275, 42, 28]
        predicted_bbox = object_prediction_list[2].bbox.to_coco_bbox()
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin

        for object_prediction in object_prediction_list:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
