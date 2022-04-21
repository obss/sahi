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

    def test_perform_inference_without_mask_output(self):
        from sahi.model import TorchVisionDetectionModel

        # init model
        download_torchvision_model()

        torchvision_detection_model = TorchVisionDetectionModel(
            model_path=TorchVisionTestConstants.FASTERCNN_MODEL_PATH,
            config_path=TorchVisionTestConstants.FASTERCNN_CONFIG_ZOO_NAME,
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
        torchvision_detection_model.perform_inference(image)
        original_predictions = torchvision_detection_model.original_predictions

        from sahi.utils.torchvision import COCO_CLASSES

        boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(original_predictions[0]["boxes"].detach().numpy())]
        scores = list(original_predictions[0]["scores"].detach().numpy())
        thresh = [scores.index(x) for x in scores if x > CONFIDENCE_THRESHOLD][-1]
        prediction_class = [COCO_CLASSES[i] for i in list(original_predictions[0]["labels"].numpy())]
        category_name = prediction_class[: thresh + 1]
        category_map = {}
        for i in range(len(COCO_CLASSES)):
            category_map[COCO_CLASSES[i]] = i
            category_map[i] = COCO_CLASSES[i]
        category_id = [category_map[i] for i in category_name]
        for ind in range(len(boxes)):
            bbox = []
            for i in range(len(boxes[ind])):
                bbox.append(boxes[ind][i][0])
                bbox.append(boxes[ind][i][1])

        self.assertEqual(len(bbox), 4)
        self.assertEqual(len(category_id), 25)
        for i in range(len(bbox)):
            self.assertEqual(bbox[i].astype(int), [372, 85, 376, 89][i])
            self.assertEqual(category_id[i], 3)

    def test_convert_original_predictions_without_mask_output(self):
        from sahi.model import TorchVisionDetectionModel

        download_torchvision_model()

        torchvision_detection_model = TorchVisionDetectionModel(
            model_path=TorchVisionTestConstants.FASTERCNN_MODEL_PATH,
            config_path=TorchVisionTestConstants.FASTERCNN_CONFIG_ZOO_NAME,
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
        torchvision_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        torchvision_detection_model.convert_original_predictions()
        object_prediction_list = torchvision_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 25)
        self.assertEqual(object_prediction_list[0].category.id, 3)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        self.assertEqual(object_prediction_list[0].bbox.to_coco_bbox(), [177, 176, 33, 24])

    def test_convert_original_predictions_with_mask_output(self):
        from sahi.model import TorchVisionDetectionModel

        download_torchvision_model()

        torchvision_detection_model = TorchVisionDetectionModel(
            model_path=TorchVisionTestConstants.FASTERCNN_MODEL_PATH,
            config_path=TorchVisionTestConstants.FASTERCNN_CONFIG_ZOO_NAME,
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
        torchvision_detection_model.perform_inference(image)

        # convert predictions to ObjectPrediction list
        torchvision_detection_model.convert_original_predictions()
        object_prediction_list = torchvision_detection_model.object_prediction_list

        # compare
        self.assertEqual(len(object_prediction_list), 25)
        self.assertEqual(object_prediction_list[0].category.id, 3)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        self.assertEqual(object_prediction_list[0].bbox.to_coco_bbox(), [177, 176, 33, 24])

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
        self.assertEqual(len(object_prediction_list), 25)
        self.assertEqual(object_prediction_list[0].category.id, 3)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        self.assertEqual(object_prediction_list[0].bbox.to_coco_bbox(), [177, 176, 33, 24])

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
        self.assertEqual(len(object_prediction_list), 121)
        self.assertEqual(object_prediction_list[0].category.id, 3)
        self.assertEqual(object_prediction_list[0].category.name, "car")
        self.assertEqual(object_prediction_list[0].bbox.to_coco_bbox(), [843, 286, 30, 20])


if __name__ == "__main__":
    unittest.main()
