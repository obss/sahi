# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)


class MmdetDetectionModel(DetectionModel):
    def check_dependencies(self):
        check_requirements(["torch", "mmdet", "mmcv"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """
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

        self.set_model(model)

    def set_model(self, model: Any):
        """
        Sets the underlying MMDetection model.
        Args:
            model: Any
                A MMDetection model
        """

        # set self.model
        self.model = model

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping

    def perform_inference(self, images: List):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            images: List[np.ndarray, str, PIL.Image.Image]
                A numpy array that contains a list of images to be predicted. 3 channel image should be in RGB order.
        """

        if not isinstance(images, list):
            images = [images]

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        # Supports only batch of 1
        from mmdet.apis import inference_detector

        # perform inference
        for ind, image in enumerate(images):
            if isinstance(image, Image.Image):
                image = np.array(image)
            if isinstance(image, np.ndarray):
                # mmdet accepts in BGR order
                images[ind] = image[..., ::-1]

        prediction_result = inference_detector(self.model, images)

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

    def _create_object_predictions_from_original_predictions(
        self,
        shift_amounts: Optional[List[List[int]]] = [[0, 0]],
        full_shapes: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_predictions_per_image.
        Args:
            shift_amounts: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shapes: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        category_mapping = self.category_mapping

        # parse boxes and masks from predictions
        num_categories = self.num_categories
        object_predictions_per_image = []
        for image_ind, original_prediction in enumerate(original_predictions):
            shift_amount = shift_amounts[image_ind]
            full_shape = None if full_shapes is None else full_shapes[image_ind]

            if self.has_mask:
                boxes = original_prediction[0]
                masks = original_prediction[1]
            else:
                boxes = original_prediction

            object_predictions = []

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
                        # https://github.com/obss/sahi/discussions/696
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
                    object_predictions.append(object_prediction)
            object_predictions_per_image.append(object_predictions)
        self._object_predictions_per_image = object_predictions_per_image
