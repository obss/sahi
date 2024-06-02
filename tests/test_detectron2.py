# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest

from sahi.models.detectron2 import Detectron2DetectionModel
from sahi.utils.cv import read_image
from sahi.utils.detectron2 import Detectron2TestConstants
from sahi.utils.import_utils import get_package_info

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320

# note that detectron2 binaries are available only for linux

torch_version = get_package_info("torch", verbose=False)[1]
if "1.10." in torch_version:

    class TestDetectron2DetectionModel(unittest.TestCase):
        def test_load_model(self):
            detector2_detection_model = Detectron2DetectionModel(
                model_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
                config_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                device=MODEL_DEVICE,
                category_remapping=None,
                load_at_init=True,
                image_size=IMAGE_SIZE,
            )
            self.assertNotEqual(detector2_detection_model.model, None)

        def test_perform_inference_without_mask_output(self):
            detectron2_detection_model = Detectron2DetectionModel(
                model_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
                config_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
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
            self.assertEqual(boxes[ind].astype("int").tolist(), [831, 303, 873, 346])
            self.assertEqual(len(boxes), 35)

        def test_convert_original_predictions_without_mask_output(self):
            detectron2_detection_model = Detectron2DetectionModel(
                model_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
                config_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
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
            detectron2_detection_model.perform_inference(image)

            # convert predictions to ObjectPrediction list
            detectron2_detection_model.convert_original_predictions()
            object_prediction_list = detectron2_detection_model.object_prediction_list

            # compare
            self.assertEqual(len(object_prediction_list), 16)
            self.assertEqual(object_prediction_list[0].category.id, 2)
            self.assertEqual(object_prediction_list[0].category.name, "car")
            predicted_bbox = object_prediction_list[0].bbox.to_xywh()
            desired_bbox = [831, 303, 42, 43]
            margin = 3
            for ind, point in enumerate(predicted_bbox):
                if not (point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin):
                    raise AssertionError(f"desired_bbox: {desired_bbox}, predicted_bbox: {predicted_bbox}")

            self.assertEqual(object_prediction_list[5].category.id, 2)
            self.assertEqual(object_prediction_list[5].category.name, "car")
            predicted_bbox = object_prediction_list[2].bbox.to_xywh()
            desired_bbox = [383, 277, 36, 29]
            margin = 3
            for ind, point in enumerate(predicted_bbox):
                if not (point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin):
                    raise AssertionError(f"desired_bbox: {desired_bbox}, predicted_bbox: {predicted_bbox}")

        def test_convert_original_predictions_with_mask_output(self):
            detectron2_detection_model = Detectron2DetectionModel(
                model_path=Detectron2TestConstants.MASKRCNN_MODEL_ZOO_NAME,
                config_path=Detectron2TestConstants.MASKRCNN_MODEL_ZOO_NAME,
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
            detectron2_detection_model.perform_inference(image)
            # convert predictions to ObjectPrediction list
            detectron2_detection_model.convert_original_predictions(full_shape=(image.shape[0], image.shape[1]))
            object_prediction_list = detectron2_detection_model.object_prediction_list

            # compare
            self.assertEqual(len(object_prediction_list), 13)
            self.assertEqual(object_prediction_list[0].category.id, 2)
            self.assertEqual(object_prediction_list[0].category.name, "car")
            predicted_bbox = object_prediction_list[0].bbox.to_xywh()
            desired_bbox = [321, 324, 59, 38]
            margin = 3
            for ind, point in enumerate(predicted_bbox):
                if not (point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin):
                    raise AssertionError(f"desired_bbox: {desired_bbox}, predicted_bbox: {predicted_bbox}")

            self.assertEqual(object_prediction_list[5].category.id, 2)
            self.assertEqual(object_prediction_list[5].category.name, "car")
            predicted_bbox = object_prediction_list[5].bbox.to_xywh()
            desired_bbox = [719, 243, 27, 30]
            margin = 3
            for ind, point in enumerate(predicted_bbox):
                if not (point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin):
                    raise AssertionError(f"desired_bbox: {desired_bbox}, predicted_bbox: {predicted_bbox}")

        def test_get_prediction_detectron2(self):
            from sahi.models.detectron2 import Detectron2DetectionModel
            from sahi.predict import get_prediction
            from sahi.utils.detectron2 import Detectron2TestConstants

            # init model
            detector2_detection_model = Detectron2DetectionModel(
                model_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
                config_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                device=MODEL_DEVICE,
                category_remapping=None,
                load_at_init=False,
                image_size=IMAGE_SIZE,
            )
            detector2_detection_model.load_model()

            # prepare image
            image_path = "tests/data/small-vehicles1.jpeg"
            image = read_image(image_path)

            # get full sized prediction
            prediction_result = get_prediction(
                image=image,
                detection_model=detector2_detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                postprocess=None,
            )
            object_prediction_list = prediction_result.object_prediction_list

            # compare
            self.assertEqual(len(object_prediction_list), 16)
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
            self.assertEqual(num_car, 16)

        def test_get_sliced_prediction_detectron2(self):
            from sahi.models.detectron2 import Detectron2DetectionModel
            from sahi.predict import get_sliced_prediction
            from sahi.utils.detectron2 import Detectron2TestConstants

            # init model
            detector2_detection_model = Detectron2DetectionModel(
                model_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
                config_path=Detectron2TestConstants.FASTERCNN_MODEL_ZOO_NAME,
                confidence_threshold=CONFIDENCE_THRESHOLD,
                device=MODEL_DEVICE,
                category_remapping=None,
                load_at_init=False,
                image_size=IMAGE_SIZE,
            )
            detector2_detection_model.load_model()

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
                detection_model=detector2_detection_model,
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
            self.assertEqual(len(object_prediction_list), 19)
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
            self.assertEqual(num_car, 19)


if __name__ == "__main__":
    unittest.main()
