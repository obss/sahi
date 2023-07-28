# OBSS SAHI Tool
# Code written by Michael García, 2023.

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


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


class ONNXDetectionModel(DetectionModel):
    def __init__(self, *args, iou_threshold: float = 0.7, **kwargs):
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold

    def check_dependencies(self) -> None:
        check_requirements(["onnxruntime"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        import onnxruntime

        try:
            EP_list = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            opt_session = onnxruntime.SessionOptions()
            opt_session.enable_mem_pattern = False
            opt_session.enable_cpu_mem_arena = True
            opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
            ort_session = onnxruntime.InferenceSession(self.model_path, providers=EP_list)

            self.set_model(ort_session)

        except Exception as e:
            raise TypeError("model_path is not a valid onnx model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying ONNX model.

        Args:
            model: Any
                A ONNX model
        """

        self.model = model

        # set category_mapping
        if not self.category_mapping:
            raise TypeError("Class mapping values are required")

    def _pad_and_resize_image(
        self, image: np.ndarray, new_shape: Tuple[int, int] = (640, 640), color: Tuple[int, int, int] = (125, 125, 125)
    ) -> np.ndarray:
        """Resize and pad image with color if necessary, maintaining aspect ratio

        Args:
            image: numpy.ndarray
                Image to pad and resize.
            new_shape: tuple(int, int)
                Width, height
            color: tuple(int, int, int)
                B, G, R
        """
        img_h, img_w = image.shape[:2]
        new_w, new_h = new_shape
        # rescale down
        scale = min(new_w / img_w, new_h / img_h)

        # get new sacled widths and heights
        scale_new_w, scale_new_h = int(round(img_w * scale)), int(round(img_h * scale))

        resized_img = cv2.resize(image, (scale_new_w, scale_new_h))

        # calculate deltas for padding
        dw = max(new_w - scale_new_w, 0)
        dh = max(new_h - scale_new_h, 0)

        # center image with padding on top/bottom or left/right
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)
        pad_resized_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return pad_resized_img

    def _preprocess_image(self, image: np.ndarray, input_shape: Tuple[int, int]) -> np.ndarray:
        """Prepapre image for ineference.

        Args:
            image: np.ndarray
                Input image
        """
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = cv2.resize(image, input_shape)

        input_image = input_image / 255.0
        input_image = input_image.transpose(2, 0, 1)
        image_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)

        return image_tensor

    def _post_process(self, outputs: np.ndarray, input_shape, image_shape):
        image_h, image_w = image_shape
        input_w, input_h = input_shape

        predictions = np.squeeze(outputs).T

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

        indices = nms(boxes, scores, self.iou_threshold)

        # Format the results
        prediction_result = []
        for (bbox, score, label) in zip(boxes[indices], scores[indices], class_ids[indices]):
            bbox = bbox.tolist()
            cls_id = int(label)
            prediction_result.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])

        # prediction_result = [torch.tensor(prediction_result)]
        prediction_result = [prediction_result]

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

        # input_h, input_w = input_shape[2:]
        # image_h, image_w = image.shape[:2]

        # Prepare image
        # image = self._pad_and_resize_image(image, input_shape)
        image_tensor = self._preprocess_image(image)

        # Inference
        outputs = self.model.run(output_names, {input_names[0]: image_tensor})[0]

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
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
