# OBSS SAHI Tool
# Code written by Kadir Nar, 2022.


import unittest

from sahi.utils.torchvision import TorchVisionTestConstants, download_torchvision_model, read_image

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320


class TestTorchVisionDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.model import TorchVisionDetectionModel

        download_torchvision_model()

        torchvision_detection_model = TorchVisionDetectionModel(
            model_path=TorchVisionTestConstants.FASTERCNN_MODEL_PATH,
            config_path=TorchVisionTestConstants.FASTERCNN_CONFIG_ZOO_NAME,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )
        self.assertNotEqual(torchvision_detection_model.model, None)

    def test_get_prediction_torchvision(self):
        from sahi.model import TorchVisionDetectionModel
        from sahi.predict import get_prediction

        download_torchvision_model()

        # init model
        torchvision_detection_model = TorchVisionDetectionModel(
            model_path=TorchVisionTestConstants.FASTERCNN_MODEL_PATH,
            config_path=TorchVisionTestConstants.FASTERCNN_CONFIG_ZOO_NAME,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        torchvision_detection_model.load_model()

        # prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        # get full sized prediction
        prediction_result = get_prediction(
            image=image,
            detection_model=torchvision_detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            postprocess=None,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 33)
        self.assertEqual(object_prediction_list[0].category.id, 3)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        self.assertEqual(object_prediction_list[0].bbox.to_coco_bbox(), [321, 320, 61, 43])

    def test_get_sliced_prediction_torchvision(self):
        from sahi.model import TorchVisionDetectionModel
        from sahi.predict import get_sliced_prediction

        download_torchvision_model()

        # init model
        torchvision_detection_model = TorchVisionDetectionModel(
            model_path=TorchVisionTestConstants.FASTERCNN_MODEL_PATH,
            config_path=TorchVisionTestConstants.FASTERCNN_CONFIG_ZOO_NAME,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )
        torchvision_detection_model.load_model()

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
            detection_model=torchvision_detection_model,
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
        self.assertEqual(len(object_prediction_list), 118)
        self.assertEqual(object_prediction_list[0].category.id, 3)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        self.assertEqual(object_prediction_list[0].bbox.to_coco_bbox(), [843, 286, 30, 20])


if __name__ == "__main__":
    unittest.main()
