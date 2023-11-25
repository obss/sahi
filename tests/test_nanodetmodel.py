# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

from sahi.models.nanodet import NanodetDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.nanodet import NanodetConstants

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320
CAR_INDEX = 2
nanodet_constants = NanodetConstants()


class TestNanodetDetectionModel(unittest.TestCase):
    def test_load_model(self):
        nanodet_detection_model = NanodetDetectionModel(
            model_path=nanodet_constants.NANODET_PLUS_MODEL,
            config_path=nanodet_constants.NANODET_PLUS_CONFIG,
            device=MODEL_DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            load_at_init=True,
        )
        self.assertNotEqual(nanodet_detection_model.model, None)

    def test_perform_inference(self):
        nanodet_detection_model = NanodetDetectionModel(
            model_path=nanodet_constants.NANODET_PLUS_MODEL,
            config_path=nanodet_constants.NANODET_PLUS_CONFIG,
            device=MODEL_DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            load_at_init=True,
        )
        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)
        # perform inference
        nanodet_detection_model.perform_inference(image)
        original_predictions = nanodet_detection_model.original_predictions[0]

        # find box of first car detection with conf greater than 0.5
        for detection in original_predictions[CAR_INDEX]:
            if detection[-1] > CONFIDENCE_THRESHOLD:
                box = detection[:4]
                break
        # compare

        self.assertEqual([i for i in map(int, box)], [445, 309, 493, 342])
        self.assertEqual(len(original_predictions), 80)

    def test_convert_original_predictions_without_mask_output(self):
        nanodet_detection_model = NanodetDetectionModel(
            model_path=nanodet_constants.NANODET_PLUS_MODEL,
            config_path=nanodet_constants.NANODET_PLUS_CONFIG,
            device=MODEL_DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            load_at_init=True,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # perform inference
        nanodet_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        nanodet_detection_model.convert_original_predictions()
        object_prediction_list = nanodet_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 3)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        predicted_bbox = object_prediction_list[0].bbox.to_xywh()
        desired_bbox = [445, 309, 47, 33]
        margin = 3
        for ind, point in enumerate(predicted_bbox):
            if not (point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin):
                raise AssertionError(f"desired_bbox: {desired_bbox}, predicted_bbox: {predicted_bbox}")

        self.assertEqual(object_prediction_list[2].category.id, 2)
        self.assertEqual(object_prediction_list[2].category.name, "car")
        predicted_bbox = object_prediction_list[2].bbox.to_xywh()
        desired_bbox = [377, 281, 41, 24]
        margin = 3
        for ind, point in enumerate(predicted_bbox):
            if not (point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin):
                raise AssertionError(f"desired_bbox: {desired_bbox}, predicted_bbox: {predicted_bbox}")

    def test_get_prediction_detectron2(self):
        from sahi.predict import get_prediction

        # init model
        nanodet_detection_model = NanodetDetectionModel(
            model_path=nanodet_constants.NANODET_PLUS_MODEL,
            config_path=nanodet_constants.NANODET_PLUS_CONFIG,
            device=MODEL_DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            load_at_init=True,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image,
            detection_model=nanodet_detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )

        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 3)
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
        self.assertEqual(num_car, 3)

    def test_get_sliced_prediction_detectron2(self):
        from sahi.models.nanodet import NanodetDetectionModel
        from sahi.predict import get_sliced_prediction

        # init model
        nanodet_detection_model = NanodetDetectionModel(
            model_path=nanodet_constants.NANODET_PLUS_MODEL,
            config_path=nanodet_constants.NANODET_PLUS_CONFIG,
            device=MODEL_DEVICE,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            load_at_init=True,
        )

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 416
        slice_width = 416
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=nanodet_detection_model,
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

        self.assertEqual(len(object_prediction_list), 11)
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

        self.assertEqual(num_car, 11)


if __name__ == "__main__":
    unittest.main()
