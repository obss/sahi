# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest
import unittest.mock

import numpy as np

from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.utils.mmdet import MmdetTestConstants

try:
    import mmdet

    mmdet_major_version = int(mmdet.__version__.split(".")[0])
except:
    mmdet_major_version = -1  # not installed


MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320

WITH_MASK_CONFIG_PATH = "tests/data/mmdet3/configs/cascade_rcnn/cascade-mask-rcnn_r50_fpn_1x_coco.py"
WITH_MASK_MODEL_URL = "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"

WITHOUT_MASK_CONFIG_PATH = MmdetTestConstants.MMDET3_YOLOX_TINY_CONFIG_PATH
WITHOUT_MASK_MODEL_URL = "https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth"

IMAGE_PATH = "tests/data/small-vehicles1.jpeg"


def get_dummy_predictions(image_shape, mask_type=None):
    import pycocotools

    h, w = image_shape[:2]
    bbox1 = [10, 20, 30, 40]  # xywh: 10, 20, 10, 20
    bbox2 = [100, 100, 200, 150]  # xywh: 100, 100, 100, 50
    mask1 = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = bbox1
    mask1[y0 : y1 + 1, x0 : x1 + 1] = True
    mask2 = np.zeros((h, w), dtype=bool)
    x0, y0, x1, y1 = bbox2
    mask2[y0 : y1 + 1, x0 : x1 + 1] = True

    bin_masks = [mask1, mask2]
    rle_masks = [pycocotools.mask.encode(np.asfortranarray(m)) for m in [mask1, mask2]]
    preds = dict(
        predictions=[
            dict(
                bboxes=[bbox1, bbox2],
                labels=[2, 2],
                scores=[0.3, 0.8],
            ),
        ]
    )
    if mask_type is not None:
        for pred in preds["predictions"]:
            pred["masks"] = bin_masks if mask_type == "bin" else rle_masks
    return preds


@unittest.skipIf(mmdet_major_version < 3, "mmdet v3 is not supported")
class TestMmdet3DetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.mmdet3 import Mmdet3DetectionModel

        mmdet_detection_model = Mmdet3DetectionModel(
            model_path=WITH_MASK_MODEL_URL,
            config_path=WITH_MASK_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        self.assertNotEqual(mmdet_detection_model.model, None)

    def test_perform_inference_with_mask_output(self):
        from sahi.models.mmdet3 import Mmdet3DetectionModel

        # init model
        mmdet_detection_model = Mmdet3DetectionModel(
            model_path=WITH_MASK_MODEL_URL,
            config_path=WITH_MASK_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        # check has mask and coco classes
        self.assertTrue(mmdet_detection_model.has_mask)
        self.assertTrue(mmdet_detection_model.num_categories == 80)

        # prepare image
        image = read_image(IMAGE_PATH)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions

        # check actual prediction structures
        pred = original_predictions[0]
        self.assertTrue("bboxes" in pred)
        self.assertTrue("masks" in pred)
        self.assertTrue("scores" in pred)
        self.assertTrue("labels" in pred)

        # all annotations have the same length
        n_preds = len(pred["bboxes"])
        self.assertTrue(len(pred["bboxes"]) == n_preds)
        self.assertTrue(len(pred["masks"]) == n_preds)
        self.assertTrue(len(pred["labels"]) == n_preds)
        self.assertTrue(len(pred["scores"]) == n_preds)

        # The following checks are data dependent.
        # The tests can fail if model weight or input are changed.

        # check all scores in (0.0, 1.0)
        self.assertTrue(all([0 <= score <= 1.0 for score in pred["scores"]]))

        # bbox has 4 coordinates
        self.assertTrue(len(pred["bboxes"][0]) == 4)

    def test_convert_original_predictions_with_mask_output(self):
        from sahi.models.mmdet3 import Mmdet3DetectionModel

        # init model
        mmdet_detection_model = Mmdet3DetectionModel(
            model_path=WITH_MASK_MODEL_URL,
            config_path=WITH_MASK_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        # prepare image
        image = read_image(IMAGE_PATH)

        # prepare mocked model
        inferencer = unittest.mock.MagicMock()
        inferencer.model.with_mask = True
        inferencer.model.dataset_meta = {"classes": mmdet_detection_model.category_names}
        inferencer.return_value = get_dummy_predictions(image.shape, mask_type="rle")
        mmdet_detection_model.model = inferencer

        # perform inference
        mmdet_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        mmdet_detection_model.convert_original_predictions()
        object_predictions = mmdet_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_predictions), 1)
        self.assertEqual(object_predictions[0].category.id, 2)
        self.assertEqual(object_predictions[0].category.name, "car")
        self.assertEqual(
            object_predictions[0].bbox.to_xywh(),
            [100, 100, 100, 50],
        )

        for object_prediction in object_predictions:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

    def test_convert_original_predictions_without_mask_output(self):
        from sahi.models.mmdet3 import Mmdet3DetectionModel

        # init model
        mmdet_detection_model = Mmdet3DetectionModel(
            model_path=WITHOUT_MASK_MODEL_URL,
            config_path=WITHOUT_MASK_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        # no mask
        self.assertFalse(mmdet_detection_model.has_mask)

        # prepare image
        image = read_image(IMAGE_PATH)

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions

        # check actual prediction structures
        pred = original_predictions[0]
        self.assertTrue("bboxes" in pred)
        self.assertTrue("masks" not in pred)
        self.assertTrue("scores" in pred)
        self.assertTrue("labels" in pred)

        # all annotations have the same length
        n_preds = len(pred["bboxes"])
        self.assertTrue(len(pred["bboxes"]) == n_preds)
        self.assertTrue(len(pred["labels"]) == n_preds)
        self.assertTrue(len(pred["scores"]) == n_preds)

        # prepare mocked model
        inferencer = unittest.mock.MagicMock()
        inferencer.model.with_mask = False
        inferencer.model.dataset_meta = {"classes": mmdet_detection_model.category_names}
        inferencer.return_value = get_dummy_predictions(image.shape, mask_type=None)
        mmdet_detection_model.model = inferencer

        # perform inference
        mmdet_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        mmdet_detection_model.convert_original_predictions()
        object_predictions = mmdet_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_predictions), 1)
        self.assertEqual(object_predictions[0].category.id, 2)
        self.assertEqual(object_predictions[0].category.name, "car")
        np.testing.assert_almost_equal(object_predictions[0].bbox.to_xywh(), [100, 100, 100, 50], decimal=1)
        for object_prediction in object_predictions:
            self.assertGreaterEqual(object_prediction.score.value, CONFIDENCE_THRESHOLD)

    def test_perform_inference_without_mask_output_with_automodel(self):
        from sahi import AutoDetectionModel

        # init model
        mmdet_detection_model = AutoDetectionModel.from_pretrained(
            model_type="mmdet3",
            model_path=None,
            config_path=WITHOUT_MASK_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            category_remapping=None,
            load_at_init=True,
        )

        # prepare image
        image = read_image(IMAGE_PATH)

        # prepare mocked model
        inferencer = unittest.mock.MagicMock()
        inferencer.model.with_mask = False
        inferencer.model.dataset_meta = {"classes": mmdet_detection_model.category_names}
        inferencer.return_value = get_dummy_predictions(image.shape, mask_type=None)
        mmdet_detection_model.model = inferencer

        # perform inference
        mmdet_detection_model.perform_inference(image)
        original_predictions = mmdet_detection_model.original_predictions
        boxes = original_predictions[0]["bboxes"]
        scores = original_predictions[0]["scores"]

        n_boxes = len(boxes)
        for i in range(n_boxes):
            if scores[i] > 0.5:
                break
        box = boxes[i]

        # compare
        self.assertEqual(box, [100, 100, 200, 150])


if __name__ == "__main__":
    unittest.main()
