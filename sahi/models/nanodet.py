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
        self.model.cuda()
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
        self.model.cuda()
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

    def process_prediction(self, category_box, category_id, shift_amount, full_shape):
        """Processes a single category prediction.

        Args:
            category_box: The bounding box of the category prediction.
            category_id: The category ID.
            shift_amount: The shift amount.
            full_shape: The full shape of the prediction.

        Returns:
            ObjectPrediction or None: The processed object prediction if valid, otherwise None.
        """
        bbox, score = category_box[:4], category_box[4]
        category_name = self.category_mapping[str(category_id)]

        if score < self.confidence_threshold:
            return None

        bool_mask = None
        bbox = [max(0, x) for x in bbox]

        if full_shape is not None:
            bbox = [min(dim, x) for dim, x in zip(full_shape[::-1], bbox)]

        if (not (bbox[0] < bbox[2])) or (not (bbox[1] < bbox[3])):
            logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
            return None

        return ObjectPrediction(
            bbox=bbox,
            category_id=category_id,
            score=score,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = None,
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """Creates a list of ObjectPrediction from the original predictions.

        Args:
            shift_amount_list (List[List[int]], optional): The shift amount list.
            full_shape_list (List[List[int]], optional): The full shape list.

        Returns:
            List[List[ObjectPrediction]]: The list of ObjectPrediction per image.
        """
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]]
        original_predictions = self._original_predictions
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)
        num_categories = self.num_categories

        object_prediction_list_per_image = [
            [
                pred
                for category_id in range(num_categories)
                for pred in (
                    self.process_prediction(
                        category_box,
                        category_id,
                        shift_amount_list[image_ind],
                        full_shape_list[image_ind] if full_shape_list else None,
                    )
                    for category_box in original_predictions[image_ind][category_id]
                )
                if pred is not None
            ]
            for image_ind in original_predictions.keys()
        ]

        self._object_prediction_list_per_image = object_prediction_list_per_image
