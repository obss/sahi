from __future__ import annotations

from typing import Any

import numpy as np

from sahi.annotation import Category
from sahi.logger import logger
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import check_requirements
from sahi.utils.torch_utils import empty_cuda_cache, select_device


class DetectionModel:
    """Base class for all detection models in SAHI.

    Subclasses must implement ``load_model``, ``perform_inference``, and
    ``_create_object_prediction_list_from_original_predictions`` to integrate
    a new detection framework. The base class handles device management,
    dependency checking, category remapping, and the public prediction API.
    """

    required_packages: list[str] | None = None

    def __init__(
        self,
        model_path: str | None = None,
        model: Any | None = None,
        config_path: str | None = None,
        device: str | None = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: dict | None = None,
        category_remapping: dict | None = None,
        load_at_init: bool = True,
        image_size: int | None = None,
    ):
        """Init object detection/instance segmentation model.

        Args:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: Torch device, "cpu", "mps", "cuda", "cuda:0", "cuda:1", etc.
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initialization
            image_size: int
                Inference input size.
        """

        self.model_path = model_path
        self.config_path = config_path
        self.model = None
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None
        self.set_device(device)

        # automatically ensure dependencies
        self.check_dependencies()

        # automatically load model if load_at_init is True
        if load_at_init:
            if model:
                self.set_model(model)
            else:
                self.load_model()

    def check_dependencies(self, packages: list[str] | None = None) -> None:
        """Ensures required dependencies are installed.

        If 'packages' is None, uses self.required_packages. Subclasses may still call with a custom list for dynamic
        needs.
        """
        pkgs = packages if packages is not None else getattr(self, "required_packages", [])
        if pkgs:
            check_requirements(pkgs)

    def load_model(self):
        """Load the detection model from disk and assign it to ``self.model``.

        Subclasses must override this method. The implementation should use
        ``self.model_path``, ``self.config_path``, and ``self.device`` to
        construct the underlying model object and store it in ``self.model``.
        """
        raise NotImplementedError()

    def set_model(self, model: Any, **kwargs):
        """Set an already-instantiated model as the underlying detection model.

        Subclasses must override this method to assign ``model`` to
        ``self.model`` and perform any additional setup (e.g. category mapping).

        Args:
            model: Any
                A pre-loaded detection model instance.
        """
        raise NotImplementedError()

    def set_device(self, device: str | None = None):
        """Sets the device pytorch should use for the model.

        Args:
            device: Torch device, "cpu", "mps", "cuda", "cuda:0", "cuda:1", etc.
        """

        self.device = select_device(device)

    def unload_model(self):
        """Unloads the model from CPU/GPU."""
        self.model = None
        empty_cuda_cache()

    def perform_inference(self, image: np.ndarray):
        """Run inference on a single image and store raw predictions.

        Subclasses must override this method. The implementation should run
        the model on ``image`` and assign the raw results to
        ``self._original_predictions``.

        Args:
            image: np.ndarray
                A numpy array (H, W, C) containing the image to run inference on.
        """
        raise NotImplementedError()

    def perform_batch_inference(self, images: list[np.ndarray]):
        """Performs inference on a batch of images.

        Subclasses can override this for native batch support. The default
        implementation falls back to sequential single-image inference.

        After this call, ``self._original_predictions`` contains one entry per
        image and ``self._original_shapes`` stores the shape of each input image.

        Args:
            images: list[np.ndarray]
                List of numpy arrays (H, W, C) to run inference on.
        """
        all_predictions: list = []
        original_shapes: list = []
        for image in images:
            self.perform_inference(image)
            all_predictions.extend(self._original_predictions)
            original_shapes.append(image.shape)
        self._original_predictions = all_predictions
        self._original_shapes = original_shapes

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        """Convert raw predictions to a list of ObjectPrediction instances.

        Subclasses must override this method. The implementation should read
        ``self._original_predictions``, convert each raw prediction into an
        ``ObjectPrediction``, and store the result in
        ``self._object_prediction_list_per_image``. ``self.mask_threshold``
        may be used to threshold segmentation masks.

        Args:
            shift_amount_list: list of list
                Per-image pixel shifts for mapping sliced predictions back to
                the full image, in the form ``[[shift_x, shift_y], ...]``.
            full_shape_list: list of list
                Per-image full image dimensions after shifting, in the form
                ``[[height, width], ...]``.
        """
        raise NotImplementedError()

    def _apply_category_remapping(self):
        """Applies category remapping based on mapping given in self.category_remapping."""
        # confirm self.category_remapping is not None
        if self.category_remapping is None:
            raise ValueError("self.category_remapping cannot be None")
        # remap categories
        if not isinstance(self._object_prediction_list_per_image, list):
            logger.error(
                f"Unknown type for self._object_prediction_list_per_image: "
                f"{type(self._object_prediction_list_per_image)}"
            )
            return
        for object_prediction_list in self._object_prediction_list_per_image:  # type: ignore
            for object_prediction in object_prediction_list:
                old_category_id_str = str(object_prediction.category.id)
                new_category_id_int = self.category_remapping[old_category_id_str]
                object_prediction.category = Category(id=new_category_id_int, name=object_prediction.category.name)

    def convert_original_predictions(
        self,
        shift_amount: list[list[int]] | None = [[0, 0]],
        full_shape: list[list[int]] | None = None,
    ):
        """Converts original predictions of the detection model to a list of prediction.ObjectPrediction object.

        Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image,
                    should be in the form of [shift_x, shift_y]
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
    def object_prediction_list(self) -> list[list[ObjectPrediction]]:
        """Returns the object predictions for the first image.

        This is a convenience accessor for single-image inference. For batch
        inference results, use ``object_prediction_list_per_image`` instead.
        """
        if self._object_prediction_list_per_image is None:
            return []
        if len(self._object_prediction_list_per_image) == 0:
            return []
        return self._object_prediction_list_per_image[0]

    @property
    def object_prediction_list_per_image(self) -> list[list[ObjectPrediction]]:
        """Returns object predictions grouped by image.

        Each element is a list of ``ObjectPrediction`` instances for the
        corresponding image in the batch.
        """
        return self._object_prediction_list_per_image or []

    @property
    def original_predictions(self):
        """Returns the raw predictions from the underlying model.

        The format is model-specific and is set by ``perform_inference`` or
        ``perform_batch_inference``.
        """
        return self._original_predictions
