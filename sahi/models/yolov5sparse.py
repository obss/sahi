# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_package_minimum_version, check_requirements

logger = logging.getLogger(__name__)


class Yolov5SparseDetectionModel(DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["deepsparse", "sparseml"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        from deepsparse import Pipeline

        try:
            model_stub = "zoo:cv/detection/yolov5-l/pytorch/ultralytics/coco/pruned-aggressive_98"
            model = Pipeline.create(task="yolo", model_path=model_stub,)
            self.set_model(model)
        except Exception as e:
            raise TypeError("Could not load the model: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv5 model.
        Args:
            model: Any
                A YOLOv5 model
        """
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
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")
        if self.image_size is not None:
            prediction_result = self.model(images=[image], iou_thres=0.6, conf_thres=0.001)

        else:
            prediction_result = self.model(image)

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return 80

    @property
    def category_names(self):
        return ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign', 'parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe' 'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']

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
        object_prediction_list_per_image = []
        object_prediction_list = []
        # process predictions
        for prediction in original_predictions.boxes:
            x1 = prediction[0]
            y1 = prediction[1]
            x2 = prediction[2]
            y2 = prediction[3]
            bbox = [x1, y1, x2, y2]
            score = prediction[4]
            category_id = int(prediction[5])
            category_name = self.category_mapping[str(category_id)]

            # fix negative box coords
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = max(0, bbox[2])
            bbox[3] = max(0, bbox[3])


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
            )
            object_prediction_list.append(object_prediction)
        object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
