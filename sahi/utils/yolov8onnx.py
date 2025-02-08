from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from sahi.utils.ultralytics import download_yolov8n_model


# TODO: This class has no purpose, replace by just the constant
class Yolov8ONNXTestConstants:
    YOLOV8N_ONNX_MODEL_PATH = "tests/data/models/yolov8/yolov8n.onnx"


def download_yolov8n_onnx_model(
    destination_path: Union[str, Path] = Yolov8ONNXTestConstants.YOLOV8N_ONNX_MODEL_PATH,
    image_size: Optional[int] = 640,
):
    destination_path = Path(destination_path)
    model_path = destination_path.parent / (destination_path.stem + ".pt")
    download_yolov8n_model(str(model_path))

    from ultralytics import YOLO

    model = YOLO(model_path)
    model.export(format="onnx")  # , imgsz=image_size)


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Perform non-max suppression.

    Args:
        boxes: np.ndarray
            Predicted bounding boxes, shape (num_of_boxes, 4)
        scores: np.ndarray
            Confidence for predicted bounding boxes, shape (num_of_boxes).
        iou_threshold: float
            Maximum allowed overlap between bounding boxes.

    Returns:
        list of box_ids of the kept bounding boxes
    """
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


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> float:
    """Compute the IOU between a selected box and other boxes.

    Args:
        box: np.ndarray
            Selected box, shape (4)
        boxes: np.ndarray
            Other boxes used for computing IOU, shape (num_of_boxes, 4).

    Returns:
        float: intersection over union
    """
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


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)

    Args:
        x: np.ndarray
            Input bboxes, shape (num_of_boxes, 4).

    Returns:
        np.ndarray: (num_of_boxes, 4)
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
