# OBSS SAHI Tool
# Code written by Devrim Cavusoglu, 2022.

import unittest

import pybboxes.functional as pbf

from sahi.utils.cv import read_image

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IMAGE_SIZE = 320
TEST_MODEL_PATH = "hustvl/yolos-tiny"


class TestHuggingfaceDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.model import HuggingfaceDetectionModel

        huggingface_detection_model = HuggingfaceDetectionModel(
            model_path=TEST_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(huggingface_detection_model.model, None)

    def test_set_model(self):
        from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

        from sahi.model import HuggingfaceDetectionModel

        huggingface_model = AutoModelForObjectDetection.from_pretrained(TEST_MODEL_PATH)
        huggingface_feature_extractor = AutoFeatureExtractor.from_pretrained(TEST_MODEL_PATH)

        huggingface_detection_model = HuggingfaceDetectionModel(
            model=huggingface_model,
            feature_extractor=huggingface_feature_extractor,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(huggingface_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.model import HuggingfaceDetectionModel

        huggingface_detection_model = HuggingfaceDetectionModel(
            model_path=TEST_MODEL_PATH,
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

        scores, cat_ids, boxes = huggingface_detection_model.get_valid_predictions(
            logits=original_predictions.logits[0], pred_boxes=original_predictions.pred_boxes[0]
        )

        # find box of first car detection with conf greater than 0.5
        for i, box in enumerate(boxes):
            if huggingface_detection_model.category_mapping[cat_ids[i].item()] == "car":  # if category car
                break

        image_width, image_height, _ = huggingface_detection_model.image_shapes[0]
        box = list(
            pbf.convert_bbox(
                box.tolist(),
                from_type="yolo",
                to_type="voc",
                image_size=(image_width, image_height),
                return_values=True,
            )
        )

        # compare
        desired_bbox = [347, 365, 360, 401]
        predicted_bbox = list(map(int, box[:4]))
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        for score in scores:
            self.assertGreaterEqual(score.item(), CONFIDENCE_THRESHOLD)

    def test_convert_original_predictions(self):
        from sahi.model import HuggingfaceDetectionModel

        huggingface_detection_model = HuggingfaceDetectionModel(
            model_path=TEST_MODEL_PATH,
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

        # compare
        self.assertEqual(len(object_prediction_list), 46)
        self.assertEqual(object_prediction_list[0].category.id, 3)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        desired_bbox = [347, 365, 13, 36]
        predicted_bbox = object_prediction_list[0].bbox.to_coco_bbox()
        margin = 2
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin
        self.assertEqual(object_prediction_list[2].category.id, 3)
        self.assertEqual(object_prediction_list[2].category.name, "car")
        desired_bbox = [360, 345, 9, 28]
        predicted_bbox = object_prediction_list[2].bbox.to_coco_bbox()
        for ind, point in enumerate(predicted_bbox):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin

        for object_prediction in object_prediction_list:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
