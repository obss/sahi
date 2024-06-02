# OBSS SAHI Tool
# Code written by Karl-Joan Alesma and Michael García, 2023.

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements
from sahi.utils.yolov8onnx import non_max_supression, xywh2xyxy


class Yolov8OnnxDetectionModel(DetectionModel):
    def __init__(self, *args, iou_threshold: float = 0.7, **kwargs):
        """
        Args:
            iou_threshold: float
                IOU threshold for non-max supression, defaults to 0.7.
        """
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold

    def check_dependencies(self) -> None:
        check_requirements(["onnxruntime"])

    def load_model(self, ort_session_kwargs: Optional[dict] = {}) -> None:
        """Detection model is initialized and set to self.model.

        Options for onnxruntime sessions can be passed as keyword arguments.
        """

        import onnxruntime

        try:
            if self.device == torch.device("cpu"):
                EP_list = ["CPUExecutionProvider"]
            else:
                EP_list = ["CUDAExecutionProvider"]

            options = onnxruntime.SessionOptions()

            for key, value in ort_session_kwargs.items():
                setattr(options, key, value)

            ort_session = onnxruntime.InferenceSession(self.model_path, sess_options=options, providers=EP_list)

            self.set_model(ort_session)

        except Exception as e:
            raise TypeError("model_path is not a valid onnx model path: ", e)

    def set_model(self, model: Any) -> None:
        """
        Sets the underlying ONNX model.

        Args:
            model: Any
                A ONNX model
        """

        self.model = model

        # set category_mapping
        if not self.category_mapping:
            raise TypeError("Category mapping values are required")

    def _preprocess_image(self, image: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
        """Prepapre image for inference by resizing, normalizing and changing dimensions.

        Args:
            image: np.ndarray
                Input image with color channel order RGB.
        """
        input_image = cv2.resize(image, input_shape)

        input_image = input_image / 255.0
        input_image = input_image.transpose(2, 0, 1)
        image_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

        return image_tensor

    def _post_process(
        self, outputs: np.ndarray, input_shape: Tuple[int, int], image_shape: Tuple[int, int]
    ) -> List[torch.Tensor]:
        image_h, image_w = image_shape
        input_w, input_h = input_shape

        predictions = np.squeeze(outputs[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.confidence_threshold, :]
        scores = scores[scores > self.confidence_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]

        # Scale boxes to original dimensions
        input_shape = np.array([input_w, input_h, input_w, input_h])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_w, image_h, image_w, image_h])
        boxes = boxes.astype(np.int32)

        # Convert from xywh two xyxy
        boxes = xywh2xyxy(boxes).round().astype(np.int32)

        # Perform non-max supressions
        indices = non_max_supression(boxes, scores, self.iou_threshold)

        # Format the results
        prediction_result = []
        for bbox, score, label in zip(boxes[indices], scores[indices], class_ids[indices]):
            bbox = bbox.tolist()
            cls_id = int(label)
            prediction_result.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])

        prediction_result = [torch.tensor(prediction_result)]
        # prediction_result = [prediction_result]

        return prediction_result

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

        # Get input/output names shapes
        model_inputs = self.model.get_inputs()
        model_output = self.model.get_outputs()

        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        output_names = [model_output[i].name for i in range(len(model_output))]

        input_shape = model_inputs[0].shape[2:]  # w, h
        image_shape = image.shape[:2]  # h, w

        # Prepare image
        image_tensor = self._preprocess_image(image, input_shape)

        # Inference
        outputs = self.model.run(output_names, {input_names[0]: image_tensor})

        # Post-process
        prediction_results = self._post_process(outputs, input_shape, image_shape)
        self._original_predictions = prediction_results

    @property
    def category_names(self):
        return list(self.category_mapping.values())

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

        Not yet supported
        """
        return False

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
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
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
                    segmentation=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
