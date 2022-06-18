# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybboxes.functional as pbf

from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.import_utils import check_requirements, is_available
from sahi.utils.torch import is_torch_cuda_available

logger = logging.getLogger(__name__)


class DetectionModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
    ):
        """
        Init object detection/instance segmentation model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: str
                Torch device, "cpu" or "cuda"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.device = device
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None

        # automatically set device if its None
        if not (self.device):
            self.device = "cuda:0" if is_torch_cuda_available() else "cpu"

        # automatically load model if load_at_init is True
        if load_at_init:
            if model:
                self.set_model(model)
            else:
                self.load_model()

    def load_model(self):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.model_path, self.config_path, and self.device should be utilized)
        """
        raise NotImplementedError()

    def set_model(self, model: Any, **kwargs):
        """
        This function should be implemented to instantiate a DetectionModel out of an already loaded model
        Args:
            model: Any
                Loaded model
        """
        raise NotImplementedError()

    def unload_model(self):
        """
        Unloads the model from CPU/GPU.
        """
        self.model = None
        if is_available("torch"):
            from sahi.utils.torch import empty_cuda_cache

            empty_cuda_cache()

    def perform_inference(self, image: np.ndarray):
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction result should be set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
        """
        raise NotImplementedError()

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        This function should be implemented in a way that self._original_predictions should
        be converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list. self.mask_threshold can also be utilized.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        raise NotImplementedError()

    def _apply_category_remapping(self):
        """
        Applies category remapping based on mapping given in self.category_remapping
        """
        # confirm self.category_remapping is not None
        assert self.category_remapping is not None, "self.category_remapping cannot be None"
        # remap categories
        for object_prediction_list in self._object_prediction_list_per_image:
            for object_prediction in object_prediction_list:
                old_category_id_str = str(object_prediction.category.id)
                new_category_id_int = self.category_remapping[old_category_id_str]
                object_prediction.category.id = new_category_id_int

    def convert_original_predictions(
        self,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Converts original predictions of the detection model to a list of
        prediction.ObjectPrediction object. Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        self._create_object_prediction_list_from_original_predictions(
            shift_amount_list=shift_amount,
            full_shape_list=full_shape,
        )
        if self.category_remapping:
            self._apply_category_remapping()

    @property
    def object_prediction_list(self):
        return self._object_prediction_list_per_image[0]

    @property
    def object_prediction_list_per_image(self):
        return self._object_prediction_list_per_image

    @property
    def original_predictions(self):
        return self._original_predictions


@check_requirements(["torch", "mmdet", "mmcv"])
class MmdetDetectionModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            import mmdet
        except ImportError:
            raise ImportError(
                'Please run "pip install -U mmcv mmdet" ' "to install MMDetection first for MMDetection inference."
            )

        from mmdet.apis import init_detector

        # create model
        model = init_detector(
            config=self.config_path,
            checkpoint=self.model_path,
            device=self.device,
        )

        # update model image size
        if self.image_size is not None:
            model.cfg.data.test.pipeline[1]["img_scale"] = (self.image_size, self.image_size)

        # set self.model
        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        try:
            import mmdet
        except ImportError:
            raise ImportError(
                'Please run "pip install -U mmcv mmdet" ' "to install MMDetection first for MMDetection inference."
            )

        # Confirm model is loaded
        assert self.model is not None, "Model is not loaded, load it by calling .load_model()"

        # Supports only batch of 1
        from mmdet.apis import inference_detector

        # perform inference
        if isinstance(image, np.ndarray):
            # https://github.com/obss/sahi/issues/265
            image = image[:, :, ::-1]
        # compatibility with sahi v0.8.15
        if not isinstance(image, list):
            image = [image]
        prediction_result = inference_detector(self.model, image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        if isinstance(self.model.CLASSES, str):
            num_categories = 1
        else:
            num_categories = len(self.model.CLASSES)
        return num_categories

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        has_mask = self.model.with_mask
        return has_mask

    @property
    def category_names(self):
        if type(self.model.CLASSES) == str:
            # https://github.com/open-mmlab/mmdetection/pull/4973
            return (self.model.CLASSES,)
        else:
            return self.model.CLASSES

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        category_mapping = self.category_mapping

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # parse boxes and masks from predictions
        num_categories = self.num_categories
        object_prediction_list_per_image = []
        for image_ind, original_prediction in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]

            if self.has_mask:
                boxes = original_prediction[0]
                masks = original_prediction[1]
            else:
                boxes = original_prediction

            object_prediction_list = []

            # process predictions
            for category_id in range(num_categories):
                category_boxes = boxes[category_id]
                if self.has_mask:
                    category_masks = masks[category_id]
                num_category_predictions = len(category_boxes)

                for category_predictions_ind in range(num_category_predictions):
                    bbox = category_boxes[category_predictions_ind][:4]
                    score = category_boxes[category_predictions_ind][4]
                    category_name = category_mapping[str(category_id)]

                    # ignore low scored predictions
                    if score < self.confidence_threshold:
                        continue

                    # parse prediction mask
                    if self.has_mask:
                        bool_mask = category_masks[category_predictions_ind]
                        # check if mask is valid
                        # https://github.com/obss/sahi/issues/389
                        if get_bbox_from_bool_mask(bool_mask) is None:
                            continue
                    else:
                        bool_mask = None

                    # fix negative box coords
                    bbox[0] = max(0, bbox[0])
                    bbox[1] = max(0, bbox[1])
                    bbox[2] = max(0, bbox[2])
                    bbox[3] = max(0, bbox[3])

                    # fix out of image box coords
                    if full_shape is not None:
                        bbox[0] = min(full_shape[1], bbox[0])
                        bbox[1] = min(full_shape[0], bbox[1])
                        bbox[2] = min(full_shape[1], bbox[2])
                        bbox[3] = min(full_shape[0], bbox[3])

                    # ignore invalid predictions
                    if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                        logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                        continue

                    object_prediction = ObjectPrediction(
                        bbox=bbox,
                        category_id=category_id,
                        score=score,
                        bool_mask=bool_mask,
                        category_name=category_name,
                        shift_amount=shift_amount,
                        full_shape=full_shape,
                    )
                    object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)
        self._object_prediction_list_per_image = object_prediction_list_per_image


@check_requirements(["torch", "yolov5"])
class Yolov5DetectionModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        import yolov5

        try:
            model = yolov5.load(self.model_path, device=self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov5 model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv5 model.
        Args:
            model: Any
                A YOLOv5 model
        """

        if model.__class__.__module__ not in ["yolov5.models.common", "models.common"]:
            raise Exception(f"Not a yolov5 model: {type(model)}")

        model.conf = self.confidence_threshold
        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        assert self.model is not None, "Model is not loaded, load it by calling .load_model()"
        if self.image_size is not None:
            prediction_result = self.model(image, size=self.image_size)
        else:
            prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        has_mask = self.model.with_mask
        return has_mask

    @property
    def category_names(self):
        return self.model.names

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions.xyxy):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1 = int(prediction[0])
                y1 = int(prediction[1])
                x2 = int(prediction[2])
                y2 = int(prediction[3])
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


@check_requirements(["torch", "detectron2"])
class Detectron2DetectionModel(DetectionModel):
    def load_model(self):
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog
        from detectron2.engine import DefaultPredictor
        from detectron2.model_zoo import model_zoo

        cfg = get_cfg()

        try:  # try to load from model zoo
            config_file = model_zoo.get_config_file(self.config_path)
            cfg.merge_from_file(config_file)
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_path)
        except Exception as e:  # try to load from local
            print(e)
            if self.config_path is not None:
                cfg.merge_from_file(self.config_path)
            cfg.MODEL.WEIGHTS = self.model_path

        # set model device
        cfg.MODEL.DEVICE = self.device
        # set input image size
        if self.image_size is not None:
            cfg.INPUT.MIN_SIZE_TEST = self.image_size
            cfg.INPUT.MAX_SIZE_TEST = self.image_size
        # init predictor
        model = DefaultPredictor(cfg)

        self.model = model

        # detectron2 category mapping
        if self.category_mapping is None:
            try:  # try to parse category names from metadata
                metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
                category_names = metadata.thing_classes
                self.category_names = category_names
                self.category_mapping = {
                    str(ind): category_name for ind, category_name in enumerate(self.category_names)
                }
            except Exception as e:
                logger.warning(e)
                # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
                if cfg.MODEL.META_ARCHITECTURE == "RetinaNet":
                    num_categories = cfg.MODEL.RETINANET.NUM_CLASSES
                else:  # fasterrcnn/maskrcnn etc
                    num_categories = cfg.MODEL.ROI_HEADS.NUM_CLASSES
                self.category_names = [str(category_id) for category_id in range(num_categories)]
                self.category_mapping = {
                    str(ind): category_name for ind, category_name in enumerate(self.category_names)
                }
        else:
            self.category_names = list(self.category_mapping.values())

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        if isinstance(image, np.ndarray) and self.model.input_format == "BGR":
            # convert RGB image to BGR format
            image = image[:, :, ::-1]

        prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        num_categories = len(self.category_mapping)
        return num_categories

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        # parse boxes, masks, scores, category_ids from predictions
        boxes = original_predictions["instances"].pred_boxes.tensor.tolist()
        scores = original_predictions["instances"].scores.tolist()
        category_ids = original_predictions["instances"].pred_classes.tolist()

        # check if predictions contain mask
        try:
            masks = original_predictions["instances"].pred_masks.tolist()
        except AttributeError:
            masks = None

        # create object_prediction_list
        object_prediction_list_per_image = []
        object_prediction_list = []

        # detectron2 DefaultPredictor supports single image
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        for ind in range(len(boxes)):
            score = scores[ind]
            if score < self.confidence_threshold:
                continue

            category_id = category_ids[ind]

            if masks is None:
                bbox = boxes[ind]
                mask = None
            else:
                mask = np.array(masks[ind])

                # check if mask is valid
                # https://github.com/obss/sahi/issues/389
                if get_bbox_from_bool_mask(mask) is None:
                    continue
                else:
                    bbox = None

            object_prediction = ObjectPrediction(
                bbox=bbox,
                bool_mask=mask,
                category_id=category_id,
                category_name=self.category_mapping[str(category_id)],
                shift_amount=shift_amount,
                score=score,
                full_shape=full_shape,
            )
            object_prediction_list.append(object_prediction)

        # detectron2 DefaultPredictor supports single image
        object_prediction_list_per_image = [object_prediction_list]

        self._object_prediction_list_per_image = object_prediction_list_per_image


@check_requirements(["torch", "transformers"])
class HuggingfaceDetectionModel(DetectionModel):
    import torch

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        feature_extractor: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
    ):
        self._feature_extractor = feature_extractor
        self._image_shapes = []
        super().__init__(
            model_path,
            model,
            config_path,
            device,
            mask_threshold,
            confidence_threshold,
            category_mapping,
            category_remapping,
            load_at_init,
            image_size,
        )

    @property
    def feature_extractor(self):
        return self._feature_extractor

    @property
    def image_shapes(self):
        return self._image_shapes

    @property
    def num_categories(self) -> int:
        """
        Returns number of categories
        """
        return self.model.config.num_labels

    def load_model(self):
        from transformers import AutoFeatureExtractor, AutoModelForObjectDetection

        model = AutoModelForObjectDetection.from_pretrained(self.model_path)
        if self.image_size is not None:
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.model_path, size=self.image_size, do_resize=True
            )
        else:
            feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
        self.set_model(model, feature_extractor)

    def set_model(self, model: Any, feature_extractor: Any = None):
        feature_extractor = feature_extractor or self.feature_extractor
        if feature_extractor is None:
            raise ValueError(f"'feature_extractor' is required to be set, got {feature_extractor}.")
        elif (
            "ObjectDetection" not in model.__class__.__name__
            or "FeatureExtractor" not in feature_extractor.__class__.__name__
        ):
            raise ValueError(
                f"Given 'model' is not an ObjectDetectionModel or 'feature_extractor' is not a valid FeatureExtractor."
            )
        self.model = model
        self.model.to(self.device)
        self._feature_extractor = feature_extractor
        self.category_mapping = self.model.config.id2label

    def perform_inference(self, image: Union[List, np.ndarray]):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        import torch

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        with torch.no_grad():
            inputs = self.feature_extractor(images=image, return_tensors="pt")
            inputs["pixel_values"] = inputs.pixel_values.to(self.device)
            if hasattr(inputs, "pixel_mask"):
                inputs["pixel_mask"] = inputs.pixel_mask.to(self.device)
            outputs = self.model(**inputs)

        if isinstance(image, list):
            self._image_shapes = [img.shape for img in image]
        else:
            self._image_shapes = [image.shape]
        self._original_predictions = outputs

    def get_valid_predictions(
        self, logits: torch.Tensor, pred_boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        import torch

        probs = logits.softmax(-1)
        scores = probs.max(-1).values
        cat_ids = probs.argmax(-1)
        valid_detections = torch.where(cat_ids < self.num_categories, 1, 0)
        valid_confidences = torch.where(scores >= self.confidence_threshold, 1, 0)
        valid_mask = valid_detections.logical_and(valid_confidences)
        scores = scores[valid_mask]
        cat_ids = cat_ids[valid_mask]
        boxes = pred_boxes[valid_mask]
        return scores, cat_ids, boxes

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        n_image = original_predictions.logits.shape[0]
        object_prediction_list_per_image = []
        for image_ind in range(n_image):
            image_height, image_width, _ = self.image_shapes[image_ind]
            scores, cat_ids, boxes = self.get_valid_predictions(
                logits=original_predictions.logits[image_ind], pred_boxes=original_predictions.pred_boxes[image_ind]
            )

            # create object_prediction_list
            object_prediction_list = []

            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]

            for ind in range(len(boxes)):
                category_id = cat_ids[ind].item()
                yolo_bbox = boxes[ind].tolist()
                bbox = list(
                    pbf.convert_bbox(
                        yolo_bbox,
                        from_type="yolo",
                        to_type="voc",
                        image_size=(image_width, image_height),
                        return_values=True,
                    )
                )

                # fix negative box coords
                bbox[0] = max(0, int(bbox[0]))
                bbox[1] = max(0, int(bbox[1]))
                bbox[2] = min(bbox[2], image_width)
                bbox[3] = min(bbox[3], image_height)

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    bool_mask=None,
                    category_id=category_id,
                    category_name=self.category_mapping[category_id],
                    shift_amount=shift_amount,
                    score=scores[ind].item(),
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image


@check_requirements(["torch", "torchvision"])
class TorchVisionDetectionModel(DetectionModel):
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
    ):

        super().__init__(
            model_path=model_path,
            model=model,
            config_path=config_path,
            device=device,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=load_at_init,
            image_size=image_size,
        )

    def load_model(self):
        import torch

        from sahi.utils.torchvision import MODEL_NAME_TO_CONSTRUCTOR

        # read config params
        model_name = None
        num_classes = None
        if self.config_path is not None:
            import yaml

            with open(self.config_path, "r") as stream:
                try:
                    config = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    raise RuntimeError(exc)

            model_name = config.get("model_name", None)
            num_classes = config.get("num_classes", None)

        # complete params if not provided in config
        if not model_name:
            model_name = "fasterrcnn_resnet50_fpn"
            logger.warning(f"model_name not provided in config, using default model_type: {model_name}'")
        if num_classes is None:
            logger.warning("num_classes not provided in config, using default num_classes: 91")
            num_classes = 91
        if self.model_path is None:
            logger.warning("model_path not provided in config, using pretrained weights and default num_classes: 91.")
            pretrained = True
            num_classes = 91
        else:
            pretrained = False

        # load model
        model = MODEL_NAME_TO_CONSTRUCTOR[model_name](num_classes=num_classes, pretrained=pretrained)
        try:
            model.load_state_dict(torch.load(self.model_path))
        except Exception as e:
            TypeError("model_path is not a valid torchvision model path: ", e)

        self.set_model(model)

    def set_model(self, model: Any):
        """
        Sets the underlying TorchVision model.
        Args:
            model: Any
                A TorchVision model
        """

        model.eval()
        self.model = model.to(self.device)

        # set category_mapping
        from sahi.utils.torchvision import COCO_CLASSES

        if self.category_mapping is None:
            category_names = {str(i): COCO_CLASSES[i] for i in range(len(COCO_CLASSES))}
            self.category_mapping = category_names

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
            image_size: int
                Inference input size.
        """
        from sahi.utils.torch import to_float_tensor

        # arrange model input size
        if self.image_size is not None:
            # get min and max of image height and width
            min_shape, max_shape = min(image.shape[:2]), max(image.shape[:2])
            # torchvision resize transform scales the shorter dimension to the target size
            # we want to scale the longer dimension to the target size
            image_size = self.image_size * min_shape / max_shape
            self.model.transform.min_size = (image_size,)  # default is (800,)
            self.model.transform.max_size = image_size  # default is 1333

        image = to_float_tensor(image)
        image = image.to(self.device)
        prediction_result = self.model([image])

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return self.model.with_mask

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.20
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        for image_predictions in original_predictions:
            object_prediction_list_per_image = []

            # get indices of boxes with score > confidence_threshold
            scores = image_predictions["scores"].cpu().detach().numpy()
            selected_indices = np.where(scores > self.confidence_threshold)[0]

            # parse boxes, masks, scores, category_ids from predictions
            category_ids = list(image_predictions["labels"][selected_indices].cpu().detach().numpy())
            boxes = list(image_predictions["boxes"][selected_indices].cpu().detach().numpy())
            scores = scores[selected_indices]

            # check if predictions contain mask
            masks = image_predictions.get("masks", None)
            if masks is not None:
                masks = list(image_predictions["masks"][selected_indices].cpu().detach().numpy())
            else:
                masks = None

            # create object_prediction_list
            object_prediction_list = []

            shift_amount = shift_amount_list[0]
            full_shape = None if full_shape_list is None else full_shape_list[0]

            for ind in range(len(boxes)):

                if masks is not None:
                    mask = np.array(masks[ind])
                else:
                    mask = None

                object_prediction = ObjectPrediction(
                    bbox=boxes[ind],
                    bool_mask=mask,
                    category_id=int(category_ids[ind]),
                    category_name=self.category_mapping[str(int(category_ids[ind]))],
                    shift_amount=shift_amount,
                    score=scores[ind],
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

@check_requirements(["tensorflow","tensorflow_hub"])
class TensorflowHubDetectionModel(DetectionModel):
    def load_model(self):
        try:
            import tensorflow_hub as hub
            
        except ImportError:
            raise ImportError("Please install tensorflow and tensorflow_hub via `pip install tensorflow' and 'pip install tensorflow_hub`")

        self.model = hub.load(self.model_path)
        
    def perform_inference(self, image: np.ndarray):
        if self.image_size is not None:
            from sahi.utils.tfhub import get_image,resize
            
            img = get_image(image)
            img = resize(img, self.image_size)
            
            prediction_result = self.model(img)
        
        else:
            from sahi.utils.tfhub import get_image
            img = get_image(image)
            
            prediction_result = self.model(img)
        
        self._original_predictions = prediction_result
        self.img = get_image(image)
        
        if self.category_mapping is None:
            from sahi.utils.tfhub import COCO_CLASSES
            category_mapping = {str(i): COCO_CLASSES[i] for i in range(len(COCO_CLASSES))}
            self.category_mapping = category_mapping 
        
    @property
    def num_categories(self):
        num_categories = len(self.category_mapping)
        return num_categories
    
    @property
    def has_mask(self):
        return self.model.with_mask
    
    @property
    def category_names(self):     
        return list(self.category_mapping.values()) 
    
    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ): 
        original_predictions = self._original_predictions
        
        # compatilibty for sahi v0.8.20
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]
            
        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]
            
        boxes = original_predictions["detection_boxes"][0]
        scores = original_predictions["detection_scores"][0]
        category_ids = original_predictions["detection_classes"][0]
        
        # create object_prediction_list
        object_prediction_list = []
        object_prediction_list_per_image = []

        for i in range(min(boxes.shape[0], 100)):
            if scores[i] >= self.confidence_threshold:
                score = float(scores[i].numpy())
                category_id = int(category_ids[i].numpy())
                category_names = self.category_mapping[str(category_id-1)]
                box = [float(box) for box in boxes[i].numpy()]
                x1, y1, x2, y2 = int(box[1] * self.img.shape[2]), int(box[0] * self.img.shape[1]), int(box[3] * self.img.shape[2]), int(box[2] * self.img.shape[1])
                bbox = [x1, y1, x2, y2]

                object_prediction = ObjectPrediction(
                    bbox = bbox,
                    bool_mask=None,
                    category_id= category_id,
                    category_name= category_names,
                    shift_amount=shift_amount,
                    score= score,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
        object_prediction_list_per_image.append(object_prediction_list)
        self._object_prediction_list_per_image = object_prediction_list_per_image                   