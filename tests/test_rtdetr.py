# OBSS SAHI Tool
# Code written by Fatih C Akyon (2020), Devrim Çavuşoğlu (2024).

import unittest

from sahi.utils.cv import read_image
from sahi.utils.rtdetr import RTDETRTestConstants, download_rtdetrl_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 640


class TestRTDetrDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.rtdetr import RTDetrDetectionModel

        download_rtdetrl_model()

        rtdetr_detection_model = RTDetrDetectionModel(
            model_path=RTDETRTestConstants.RTDETRL_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(rtdetr_detection_model.model, None)

    def test_set_model(self):
        from ultralytics import RTDETR

        from sahi.models.rtdetr import RTDetrDetectionModel

        download_rtdetrl_model()

        rtdetr_model = RTDETR(RTDETRTestConstants.RTDETRL_MODEL_PATH)

        rtdetr_detection_model = RTDetrDetectionModel(
            model=rtdetr_model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(rtdetr_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.models.rtdetr import RTDetrDetectionModel

        # init model
        download_rtdetrl_model()

        rtdetr_detection_model = RTDetrDetectionModel(
            model_path=RTDETRTestConstants.RTDETRL_MODEL_PATH,
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
        rtdetr_detection_model.perform_inference(image)
        original_predictions = rtdetr_detection_model.original_predictions

        boxes = original_predictions

        # find box of first car detection with conf greater than 0.5
        for box in boxes[0]:
            if box[5].item() == 2:  # if category car
                if box[4].item() > 0.5:
                    break

        # compare
        desired_bbox = [321, 322, 384, 362]
        predicted_bbox = list(map(round, box[:4].tolist()))
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(len(rtdetr_detection_model.category_names), 80)
        for box in boxes[0]:
            self.assertGreaterEqual(box[4].item(), CONFIDENCE_THRESHOLD)

    def test_convert_original_predictions(self):
        from sahi.models.rtdetr import RTDetrDetectionModel

        # init model
        download_rtdetrl_model()

        rtdetr_detection_model = RTDetrDetectionModel(
            model_path=RTDETRTestConstants.RTDETRL_MODEL_PATH,
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
        original_results = rtdetr_detection_model.model.predict(image_path, conf=CONFIDENCE_THRESHOLD)[0].boxes
        num_results = len(original_results)

        # perform inference
        rtdetr_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        rtdetr_detection_model.convert_original_predictions()
        object_prediction_list = rtdetr_detection_model.object_prediction_list

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


if __name__ == "__main__":
    unittest.main()
