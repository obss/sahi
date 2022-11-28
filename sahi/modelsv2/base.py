# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2022.

from typing import Any, Dict, List, Optional

import numpy as np

from sahi.utils.import_utils import is_available
from sahi.utils.torch import select_device as select_torch_device


class DetectionModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[Any] = None,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        label_id_to_name: Optional[Dict[int, str]] = None,
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
            label_id_to_name: dict: int to str
                Mapping from category id (int) to category name (str) e.g. {1: "pedestrian"}
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
        self.label_id_to_name = label_id_to_name
        self.image_size = image_size
        self._original_predictions = None
        self._object_predictions_per_image = None

        self.set_device()

        # automatically load model if load_at_init is True
        if load_at_init:
            if model:
                self.set_model(model)
            else:
                self.load_model()

    def check_dependencies(self) -> None:
        """
        This function can be implemented to ensure model dependencies are installed.
        """
        pass

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

    def set_device(self):
        """
        Sets the device for the model.
        """
        if is_available("torch"):
            self.device = select_torch_device(self.device)
        else:
            raise NotImplementedError()

    def unload_model(self):
        """
        Unloads the model from CPU/GPU.
        """
        self.model = None
        if is_available("torch"):
            from sahi.utils.torch import empty_cuda_cache

            empty_cuda_cache()

    def predict(self, image: np.ndarray) -> List["PredictionResult"]:
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction results should be returned
        as a PredictionResult object.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.

        Returns:
            List[PredictionResult]
        """
        raise NotImplementedError()

    @property
    def original_predictions(self):
        return self._original_predictions
