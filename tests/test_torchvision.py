import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import RoIHeads
from torchvision.models.detection.ssd import SSDHead

from sahi.constants import COCO_CLASSES
from sahi.models.torchvision import TorchVisionDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import read_image

from .utils.torchvision import TorchVisionConstants

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 320


class TestTorchVisionDetectionModel:
    def test_load_model(self):
        torchvision_detection_model = TorchVisionDetectionModel(
            config_path=TorchVisionConstants.FASTERRCNN_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )
        assert isinstance(torchvision_detection_model.model.roi_heads, RoIHeads)

    def test_load_model_without_config_path(self):
        torchvision_detection_model = TorchVisionDetectionModel(
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )
        assert isinstance(torchvision_detection_model.model.roi_heads, RoIHeads)

    def test_set_model(self):
        NUM_CLASSES = 15
        WEIGHTS = None  # Using weights=None instead of deprecated pretrained=False

        model = torchvision.models.detection.ssd300_vgg16(num_classes=NUM_CLASSES, weights=WEIGHTS)
        torchvision_detection_model = TorchVisionDetectionModel(
            model=model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=True,
        )

        assert isinstance(torchvision_detection_model.model.head, SSDHead)

    def test_perform_inference_without_mask_output(self):
        from sahi.models.torchvision import TorchVisionDetectionModel

        # init model
        torchvision_detection_model = TorchVisionDetectionModel(
            config_path=TorchVisionConstants.SSD300_CONFIG_PATH,
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
        assert original_predictions is not None
        assert isinstance(original_predictions, list)
        assert len(original_predictions) > 0

        boxes = list(original_predictions[0]["boxes"].cpu().detach().numpy())
        scores = list(original_predictions[0]["scores"].cpu().detach().numpy())
        category_ids = list(original_predictions[0]["labels"].cpu().detach().numpy())

        # get image height and width
        image_height, image_width = image.shape[:2]

        # ensure all box coords are valid
        for box_ind in range(len(boxes)):
            assert len(boxes[box_ind]) == 4
            assert boxes[box_ind][0] <= image_width
            assert boxes[box_ind][1] <= image_height
            assert boxes[box_ind][2] <= image_width
            assert boxes[box_ind][3] <= image_height
            for coord_ind in range(len(boxes[box_ind])):
                assert boxes[box_ind][coord_ind] >= 0

        # ensure all category ids are valid
        for category_id_ind in range(len(category_ids)):
            assert category_ids[category_id_ind] < len(COCO_CLASSES)
            assert category_ids[category_id_ind] >= 0

        # ensure all scores are valid
        for score_ind in range(len(scores)):
            assert scores[score_ind] <= 1
            assert scores[score_ind] >= 0

    def test_convert_original_predictions_without_mask_output(self):
        torchvision_detection_model = TorchVisionDetectionModel(
            config_path=TorchVisionConstants.FASTERRCNN_CONFIG_PATH,
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
        assert isinstance(object_prediction_list, list)
        assert isinstance(object_prediction_list[0], ObjectPrediction)

        # confirm that masks do not exist
        assert object_prediction_list[0].mask is None

        # compare
        assert len(object_prediction_list) == 7
        assert object_prediction_list[0].category.id == 3
        assert object_prediction_list[0].category.name == "car"
        np.testing.assert_almost_equal(
            object_prediction_list[0].bbox.to_xywh(), [315.79, 309.33, 64.28, 56.94], decimal=1
        )

    def test_convert_original_predictions_with_mask_output(self):
        torchvision_detection_model = TorchVisionDetectionModel(
            config_path=TorchVisionConstants.MASKRCNN_CONFIG_PATH,
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
        torchvision_detection_model.convert_original_predictions(full_shape=[image.shape[0], image.shape[1]])
        object_prediction_list = torchvision_detection_model.object_prediction_list
        assert isinstance(object_prediction_list, list)
        assert isinstance(object_prediction_list[0], ObjectPrediction)

        # confirm that masks exist
        assert object_prediction_list[0].mask is not None

        # compare
        assert len(object_prediction_list) == 8
        assert object_prediction_list[0].category.id == 3
        assert object_prediction_list[0].category.name == "car"
        np.testing.assert_allclose(object_prediction_list[0].bbox.to_xywh(), [317, 312, 60, 50], atol=1)

    def test_get_prediction_torchvision(self):
        # init model
        torchvision_detection_model = TorchVisionDetectionModel(
            config_path=TorchVisionConstants.FASTERRCNN_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
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
        assert len(object_prediction_list) == 7
        assert object_prediction_list[0].category.id == 3
        assert object_prediction_list[0].category.name == "car"
        np.testing.assert_almost_equal(
            object_prediction_list[0].bbox.to_xywh(), [315.79, 309.33, 64.28, 56.94], decimal=1
        )

    def test_get_sliced_prediction_torchvision(self):
        # init model
        torchvision_detection_model = TorchVisionDetectionModel(
            config_path=TorchVisionConstants.FASTERRCNN_CONFIG_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            device=MODEL_DEVICE,
            category_remapping=None,
            load_at_init=False,
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

        def progress_callback(progress, total):
            print(f"Progress: {progress}/{total} slices processed.")

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
            progress_bar=True,
            progress_callback=progress_callback,
        )
        object_prediction_list = prediction_result.object_prediction_list

        # compare
        assert len(object_prediction_list) == 20
        assert object_prediction_list[0].category.id == 3
        assert object_prediction_list[0].category.name == "car"
        np.testing.assert_almost_equal(
            object_prediction_list[0].bbox.to_xywh(), [765.81, 259.37, 28.62, 24.63], decimal=1
        )
