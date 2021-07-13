# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import numpy as np

from sahi.prediction import ObjectPrediction
from sahi.utils.torch import cuda_is_available, empty_cuda_cache
from typing import List, Dict, Optional, Union


class DetectionModel:
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
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
        """
        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.device = device
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self._original_predictions = None
        self._object_prediction_list = None

        # automatically set device if its None
        if not (self.device):
            self.device = "cuda:0" if cuda_is_available() else "cpu"

        # automatically load model if load_at_init is True
        if load_at_init:
            self.load_model()

    def load_model(self):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.model_path, self.config_path, and self.device should be utilized)
        """
        NotImplementedError()

    def unload_model(self):
        """
        Unloads the model from CPU/GPU.
        """
        self.model = None
        empty_cuda_cache()

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction result should be set to self._original_predictions.

        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
            image_size: int
                Inference input size.
        """
        NotImplementedError()

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        This function should be implemented in a way that self._original_predictions should
        be converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list. self.mask_threshold can also be utilized.

        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        NotImplementedError()

    def _apply_category_remapping(self):
        """
        Applies category remapping based on mapping given in self.category_remapping
        """
        # confirm self.category_remapping is not None
        assert self.category_remapping is not None, "self.category_remapping cannot be None"
        # remap categories
        for object_prediction in self._object_prediction_list:
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
            shift_amount=shift_amount,
            full_shape=full_shape,
        )
        if self.category_remapping:
            self._apply_category_remapping()

    @property
    def object_prediction_list(self):
        return self._object_prediction_list

    @property
    def original_predictions(self):
        return self._original_predictions

    def _create_predictions_from_object_prediction_list(
        object_prediction_list: List[ObjectPrediction],
    ):
        """
        This function should be implemented in a way that it converts a list of
        prediction.ObjectPrediction instance to detection model's original prediction format.
        Then returns the converted predictions.
        Can be considered as inverse of _create_object_prediction_list_from_predictions().
        Args:
            object_prediction_list: a list of prediction.ObjectPrediction
        Returns:
            original_predictions: a list of converted predictions in models original output format
        """
        NotImplementedError()


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

        # set model
        model = init_detector(
            config=self.config_path,
            checkpoint=self.model_path,
            device=self.device,
        )
        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.

        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
            image_size: int
                Inference input size.
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

        # update model image size
        if image_size is not None:
            self.model.cfg.data.test.pipeline[1]["img_scale"] = (image_size,)
        # perform inference
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
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list.

        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        original_predictions = self._original_predictions
        category_mapping = self.category_mapping

        # parse boxes and masks from predictions
        num_categories = self.num_categories
        if self.has_mask:
            boxes = original_predictions[0]
            masks = original_predictions[1]
        else:
            boxes = original_predictions

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
                if self.has_mask:
                    bool_mask = category_masks[category_predictions_ind]
                else:
                    bool_mask = None
                category_name = category_mapping[str(category_id)]

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

        self._object_prediction_list = object_prediction_list

    def _create_original_predictions_from_object_prediction_list(
        self,
        object_prediction_list: List[ObjectPrediction],
    ):
        """
        Converts a list of prediction.ObjectPrediction instance to detection model's original prediction format.
        Then returns the converted predictions.
        Can be considered as inverse of _create_object_prediction_list_from_predictions().

        Args:
            object_prediction_list: a list of prediction.ObjectPrediction
        Returns:
            original_predictions: a list of converted predictions in models original output format
        """
        # init variables
        boxes = []
        masks = []
        num_categories = self.num_categories
        category_id_list = np.arange(num_categories)
        category_id_to_bbox = {category_id: [] for category_id in category_id_list}
        category_id_to_mask = {category_id: [] for category_id in category_id_list}
        # form category_to_bbox and category_to_mask dicts from object_prediction_list
        for object_prediction in object_prediction_list:
            category_id = object_prediction.category.id
            # form bbox as 1x5 list [xmin, ymin, xmax, ymax, score]
            bbox = object_prediction.bbox.to_voc_bbox()
            bbox.extend([object_prediction.score.value])
            category_id_to_bbox[category_id].append(np.array(bbox, dtype=np.float32))
            # form 2d bool mask
            if self.has_mask:
                mask = object_prediction.mask.bool_mask
                category_id_to_mask[category_id].append(mask)

        for category_id in category_id_to_bbox.keys():
            if not category_id_to_bbox[category_id]:
                # add 0x5 array to boxes for empty categories
                boxes.append(np.zeros((0, 5), dtype=np.float32))
                if self.has_mask:
                    masks.append([])
            else:
                # form boxes and masks
                boxes.append(np.array(category_id_to_bbox[category_id]))
                if self.has_mask:
                    masks.append(np.array(category_id_to_mask[category_id]))
        # form final output
        if self.has_mask:
            original_predictions = (boxes, masks)
        else:
            original_predictions = boxes

        return original_predictions


class Yolov5DetectionModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
        try:
            import yolov5
        except ImportError:
            raise ImportError('Please run "pip install -U yolov5" ' "to install YOLOv5 first for YOLOv5 inference.")

        # set model
        try:
            model = yolov5.load(self.model_path, device=self.device)
            self.model = model
        except Exception as e:
            TypeError("model_path is not a valid yolov5 model path: ", e)

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.

        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
            image_size: int
                Inference input size.
        """
        try:
            import yolov5
        except ImportError:
            raise ImportError('Please run "pip install -U yolov5" ' "to install YOLOv5 first for YOLOv5 inference.")

        # Confirm model is loaded
        assert self.model is not None, "Model is not loaded, load it by calling .load_model()"

        if image_size:
            prediction_result = self.model(image, size=image_size)
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
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list.

        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        original_predictions = self._original_predictions

        # handle only first image (batch=1)
        predictions_in_xyxy_format = original_predictions.xyxy[0]

        object_prediction_list = []

        # process predictions
        for prediction in predictions_in_xyxy_format:
            x1 = int(prediction[0].item())
            y1 = int(prediction[1].item())
            x2 = int(prediction[2].item())
            y2 = int(prediction[3].item())
            bbox = [x1, y1, x2, y2]
            score = prediction[4].item()
            category_id = int(prediction[5].item())
            category_name = original_predictions.names[category_id]

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

        self._object_prediction_list = object_prediction_list

    def _create_original_predictions_from_object_prediction_list(
        self,
        object_prediction_list: List[ObjectPrediction],
    ):
        """
        Converts a list of prediction.ObjectPrediction instance to detection model's original
        prediction format. Then returns the converted predictions.
        Can be considered as inverse of _create_object_prediction_list_from_predictions().

        Args:
            object_prediction_list: a list of prediction.ObjectPrediction
        Returns:
            original_predictions: a list of converted predictions in models original output format
        """
        assert self.original_predictions is not None, (
            "self.original_predictions" " cannot be empty, call .perform_inference() first"
        )
        # TODO: implement object_prediction_list to yolov5 format conversion
        NotImplementedError()
