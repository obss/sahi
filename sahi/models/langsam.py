# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import cv2
from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_coco_segmentation_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)


class LangSamDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["lang_sam"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        from lang_sam import LangSAM

        try:
            self.model = LangSAM(sam_type="sam2.1_hiera_large")

        except Exception as e:
            raise TypeError("Could not load model")

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        image_pil = Image.fromarray(image)
        prediction_result = self.model.predict([image_pil], ["fish."])

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return 1

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        return True

    @property
    def category_names(self):
        return ["fish"]

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
        original_predictions = self._original_predictions[0]

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []

        scores = original_predictions["scores"].tolist()  # numpy array
        labels = original_predictions["labels"]  # list
        boxes = original_predictions["boxes"].tolist()  # numpy array nx4
        masks = original_predictions["masks"]  # numpy array nximagesize

        shift_amount = shift_amount_list[0]
        full_shape = full_shape_list[0]

        object_prediction_list = []

        for score, label, bbox, mask in zip(scores, labels, boxes, masks):

            # fix negative box coords
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(bbox[2], full_shape[0])
            bbox[3] = min(bbox[3], full_shape[1])

            object_prediction = ObjectPrediction(
                bbox=bbox,
                category_id=1,
                score=score,
                segmentation=get_coco_segmentation_from_bool_mask(mask),
                category_name=label,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )

            object_prediction_list.append(object_prediction)

        object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
