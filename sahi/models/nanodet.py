# OBSS SAHI Tool
# Code written by AnNT, 2023.

import contextlib
import logging
import os
from typing import Any, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

from nanodet.data.batch_process import stack_batch_img
from nanodet.data.collate import naive_collate
from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import cfg, load_config, load_model_weight

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements


class NanodetDetectionModel(DetectionModel):
    """A class for performing object detection using the Nanodet model."""

    def check_dependencies(self) -> None:
        """Checks the system for the following dependencies: ["nanodet", "torch", "torchvision"].

        Raises:
            AssertionError: If any of the required dependencies is not installed.
        """
        check_requirements(["nanodet", "torch", "torchvision"])

    def load_model(self):
        """Loads the detection model from configuration and weights.

        Raises:
            IOError: If the model weights file is not found or unreadable.
        """
        load_config(cfg, self.config_path)
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

        self.cfg = cfg
        # create model
        model = build_model(self.cfg.model)
        ckpt = torch.load(self.model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.eval()
        self.model.to(self.device)
        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def set_model(self, model: Any, **kwargs):
        """Sets the Nanodet model to self.model and prepares it for inference.

        Args:
            model (Any): A Nanodet model

        Raises:
            TypeError: If the model provided is not a Nanodet model.
        """
        self.model = model
        self.model.eval()
        self.model.to(self.device)
        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """Performs prediction using self.model and sets the result to self._original_predictions.

        Args:
            image (np.ndarray): A numpy array that contains the image to be predicted.
                                3 channel image should be in RGB order.
            image_size (int, optional): Inference input size.

        Raises:
            AssertionError: If the model is not loaded.
        """
        assert self.model is not None, "Model is not loaded, load it by calling .load_model()"

        img_info = {"id": 0}
        img_info["file_name"] = None
        height, width = image.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        meta = dict(img_info=img_info, raw_img=image, img=image)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = torch.from_numpy(meta["img"].transpose(2, 0, 1)).to(self.device)
        meta = naive_collate([meta])
        meta["img"] = stack_batch_img(meta["img"], divisible=32)

        # Muting nanodet logs to avoid clutter
        with torch.no_grad():
            with open(os.devnull, "w") as dev_null, contextlib.redirect_stdout(dev_null):
                results = self.model.inference(meta)
        # compatibility with sahi v0.8.15
        if not isinstance(image, list):
            image = [image]
        self._original_predictions = results

    @property
    def category_names(self):
        """Returns category names in the configuration."""
        if isinstance(self.cfg.class_names, str):
            return (self.cfg.class_names,)
        return self.cfg.class_names

    @property
    def num_categories(self):
        """Returns the number of categories in the configuration."""
        if isinstance(self.cfg.class_names, str):
            num_categories = 1
        else:
            num_categories = len(self.cfg.class_names)
        return num_categories

    @property
    def has_mask(self):
        """Returns False as Nanodet does not support segmentation models as of now."""
        return False  # fix when Nanodet supports segmentation models

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = None,
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]] * len(self._original_predictions)
        original_predictions = self._original_predictions
        category_mapping = self.category_mapping

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # parse boxes from predictions
        num_categories = self.num_categories
        object_prediction_list_per_image = []

        for image_ind, original_prediction in original_predictions.items():
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]

            object_prediction_list = []

            # process predictions
            for category_id in range(num_categories):
                category_boxes = original_prediction[category_id]

                for *bbox, score in category_boxes:
                    # ignore low scored predictions
                    if score < self.confidence_threshold:
                        continue

                    category_name = category_mapping[str(category_id)]

                    bool_mask = None

                    # fix negative box coords
                    bbox = [max(0, coord) for coord in bbox]

                    # fix out of image box coords
                    if full_shape is not None:
                        bbox = [min(full_shape[i % 2], bbox[i]) for i in range(4)]

                    # ignore invalid predictions
                    if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                        logger.warning(f"Ignoring invalid prediction with bbox: {bbox}")
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


"""
                    if full_shape is not None:
                        bbox[0] = min(full_shape[1], bbox[0])
                        bbox[1] = min(full_shape[0], bbox[1])
                        bbox[2] = min(full_shape[1], bbox[2])
                        bbox[3] = min(full_shape[0], bbox[3])"""
