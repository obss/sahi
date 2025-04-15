# OBSS SAHI Tool
# Code written by Devrim Cavusoglu, 2022.

import logging
import sys
import unittest

import pybboxes.functional as pbf
import pytest

from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image
from sahi.utils.huggingface import HuggingfaceTestConstants

pytestmark = pytest.mark.skipif(
    sys.version_info[:2] < (3, 9), reason="transformers>=4.49.0 requires Python 3.9 or higher"
)
MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320

logger = logging.getLogger(__name__)


class TestHuggingfaceDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.huggingface import HuggingfaceDetectionModel

        huggingface_detection_model = HuggingfaceDetectionModel(
            model_path=HuggingfaceTestConstants.RTDETRV2_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(huggingface_detection_model.model, None)

    def test_set_model(self):
        from transformers import AutoModelForObjectDetection, AutoProcessor

        from sahi.models.huggingface import HuggingfaceDetectionModel

        huggingface_model = AutoModelForObjectDetection.from_pretrained(HuggingfaceTestConstants.RTDETRV2_MODEL_PATH)
        huggingface_processor = AutoProcessor.from_pretrained(HuggingfaceTestConstants.RTDETRV2_MODEL_PATH)

        huggingface_detection_model = HuggingfaceDetectionModel(
            model=huggingface_model,
            processor=huggingface_processor,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(huggingface_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.models.huggingface import HuggingfaceDetectionModel

        huggingface_detection_model = HuggingfaceDetectionModel(
            model_path=HuggingfaceTestConstants.RTDETRV2_MODEL_PATH,
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
        huggingface_detection_model.perform_inference(image)
        original_predictions = huggingface_detection_model.original_predictions
        assert original_predictions

        scores, cat_ids, boxes = huggingface_detection_model.get_valid_predictions(
            logits=original_predictions.logits[0], pred_boxes=original_predictions.pred_boxes[0]
        )

        # find box of first car detection with conf greater than 0.5
        for i, box in enumerate(boxes):
            if huggingface_detection_model.category_mapping[cat_ids[i].item()] == "car":  # if category car
                break

        image_height, image_width, _ = huggingface_detection_model.image_shapes[0]
        box = list(
            pbf.convert_bbox(  #  type: ignore
                box.tolist(),
                from_type="yolo",
                to_type="voc",
                image_size=(image_width, image_height),
                return_values=True,
            )
        )
        desired_bbox = [451, 312, 490, 341]
        predicted_bbox = list(map(int, box[:4]))
        logger.debug(predicted_bbox)
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        for score in scores:
            self.assertGreaterEqual(score.item(), CONFIDENCE_THRESHOLD)

    def test_convert_original_predictions(self):
        from sahi.models.huggingface import HuggingfaceDetectionModel

        huggingface_detection_model = HuggingfaceDetectionModel(
            model_path=HuggingfaceTestConstants.RTDETRV2_MODEL_PATH,
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
        huggingface_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        huggingface_detection_model.convert_original_predictions()
        object_prediction_list = huggingface_detection_model.object_prediction_list
        assert object_prediction_list
        assert isinstance(object_prediction_list[0], ObjectPrediction)
        assert isinstance(object_prediction_list[2], ObjectPrediction)

        # compare
        self.assertEqual(len(object_prediction_list), 10)
        self.assertEqual(object_prediction_list[0].category.id, 2)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        desired_bbox = [451, 312, 39, 29]
        predicted_bbox = object_prediction_list[0].bbox.to_xywh()
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(object_prediction_list[2].category.id, 2)
        self.assertEqual(object_prediction_list[2].category.name, "car")
        desired_bbox = [609, 240, 21, 18]
        predicted_bbox = object_prediction_list[2].bbox.to_xywh()
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin

        for object_prediction in object_prediction_list:
            assert isinstance(object_prediction, ObjectPrediction)
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

    def test_get_prediction_huggingface(self):
        from sahi.models.huggingface import HuggingfaceDetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.huggingface import HuggingfaceTestConstants

        huggingface_detection_model = HuggingfaceDetectionModel(
            model_path=HuggingfaceTestConstants.RTDETRV2_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        huggingface_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image,
            detection_model=huggingface_detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 10)
        num_person = num_truck = num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
            elif object_prediction.category.name == "truck":
                num_truck += 1
            elif object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_person, 0)
        self.assertEqual(num_truck, 0)
        self.assertEqual(num_car, 10)

    def test_get_prediction_automodel_huggingface(self):
        from sahi.auto_model import AutoDetectionModel
        from sahi.predict import get_prediction
        from sahi.utils.huggingface import HuggingfaceTestConstants

        huggingface_detection_model = AutoDetectionModel.from_pretrained(
            model_type="huggingface",
            model_path=HuggingfaceTestConstants.RTDETRV2_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        huggingface_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image,
            detection_model=huggingface_detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 10)
        num_person = num_truck = num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
            elif object_prediction.category.name == "truck":
                num_truck += 1
            elif object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_person, 0)
        self.assertEqual(num_truck, 0)
        self.assertEqual(num_car, 10)

    def test_get_sliced_prediction_huggingface(self):
        from sahi.models.huggingface import HuggingfaceDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.utils.huggingface import HuggingfaceTestConstants

        huggingface_detection_model = HuggingfaceDetectionModel(
            model_path=HuggingfaceTestConstants.RTDETRV2_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
            image_size=IMAGE_SIZE,
        )
        huggingface_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"

        slice_height = 512
        slice_width = 512
        overlap_height_ratio = 0.1
        overlap_width_ratio = 0.2
        postprocess_type = "GREEDYNMM"
        match_metric = "IOS"
        match_threshold = 0.5
        class_agnostic = True

        # get sliced prediction
        prediction_result = get_sliced_prediction(
            image=image_path,
            detection_model=huggingface_detection_model,
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
        self.assertEqual(len(object_prediction_list), 17)
        num_person = num_truck = num_car = 0
        for object_prediction in object_prediction_list:
            if object_prediction.category.name == "person":
                num_person += 1
            elif object_prediction.category.name == "truck":
                num_truck += 1
            elif object_prediction.category.name == "car":
                num_car += 1
        self.assertEqual(num_person, 0)
        self.assertEqual(num_truck, 0)
        self.assertEqual(num_car, 17)


if __name__ == "__main__":
    unittest.main()
