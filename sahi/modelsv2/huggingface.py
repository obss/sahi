# OBSS SAHI Tool
# Code written by Fatih C Akyon and Devrim Cavusoglu, 2022.

import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from sahi.modelsv2 import DetectionModel
from sahi.pipelines.core import BoundingBoxes, PredictionResult
from sahi.utils.import_utils import check_requirements, ensure_package_minimum_version
from sahi.utils.torch import tensor_to_numpy

logger = logging.getLogger(__name__)


def center_to_corners_format(x):
    import torch

    """
    Converts a PyTorch tensor of bounding boxes of center format (center_x, center_y, width, height) to corners format
    (x_0, y_0, x_1, y_1).
    """
    center_x, center_y, width, height = x.unbind(-1)
    b = [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)]
    return torch.stack(b, dim=-1)


def nxyhw_to_xyxy(boxes, image_size: Sequence[int]):
    """
    Converts a PyTorch tensor of bounding boxes of normalzied (center_x, center_y, width, height) to real (x_0, y_0, x_1, y_1).

    Args:
        boxes: torch.Tensor
            A PyTorch tensor of bounding boxes of normalized format (center_x, center_y, width, height).
        image_size: Tuple[int, int]
            The image size of the bounding boxes (height, width).
    """
    import torch

    boxes = center_to_corners_format(boxes)

    img_h, img_w = image_size

    scale_fct = torch.Tensor([img_w, img_h, img_w, img_h]).to(boxes.device)
    boxes = boxes * scale_fct
    return boxes


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
        label_id_to_name: Optional[Dict[int, str]] = None,
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
            label_id_to_name,
            load_at_init,
            image_size,
        )

    def check_dependencies(self):
        check_requirements(["torch", "transformers"])
        ensure_package_minimum_version("transformers", "4.24.0")

    @property
    def feature_extractor(self):
        return self._feature_extractor

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
                "Given 'model' is not an ObjectDetectionModel or 'feature_extractor' is not a valid FeatureExtractor."
            )
        self.model = model
        self.model.to(self.device)
        self._feature_extractor = feature_extractor
        self.category_mapping = self.model.config.id2label

    def _get_valid_predictions(
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

    def predict(self, images: List):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.

        Args:
            images: List[np.ndarray, PIL.Image.Image]
                A numpy array that contains a list of images to be predicted. 3 channel image should be in RGB order.

        Returns:
            List[PredictionResult]: PredictionResult per image.
        """
        import torch

        if not isinstance(images, list):
            images = [images]

        # confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        # save image shapes
        image_shapes = []
        for image in images:
            if isinstance(image, np.ndarray):
                image_shapes.append(image.shape[:2])
            elif isinstance(image, Image.Image):
                image_shapes.append(image.size[::-1])
            else:
                raise ValueError(f"Unsupported image type {type(image)}")

        # peform forward pass
        with torch.no_grad():
            inputs = self.feature_extractor(images=copy.deepcopy(images), return_tensors="pt")
            inputs["pixel_values"] = inputs.pixel_values.to(self.device)
            if hasattr(inputs, "pixel_mask"):
                inputs["pixel_mask"] = inputs.pixel_mask.to(self.device)
            original_predictions = self.model(**inputs)

        self._original_predictions = original_predictions

        # convert to PredictionResult
        number_of_images = original_predictions.logits.shape[0]
        prediction_results = []
        for image_ind in range(number_of_images):
            scores, cat_ids, boxes = self._get_valid_predictions(
                logits=original_predictions.logits[image_ind], pred_boxes=original_predictions.pred_boxes[image_ind]
            )

            # convert yolo formatted numpy 'bboxes' to voc formatted numpy 'bboxes'
            boxes = nxyhw_to_xyxy(boxes, image_size=image_shapes[image_ind])

            bboxes = BoundingBoxes(
                bboxes=tensor_to_numpy(boxes), scores=tensor_to_numpy(scores), labels=tensor_to_numpy(cat_ids)
            )
            prediction_result = PredictionResult(
                bboxes=bboxes,
                label_id_to_name=self.label_id_to_name,
            )
            prediction_results.append(prediction_result)

        return prediction_results
