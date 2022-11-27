import copy
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image

from sahi.annotation import Mask
from sahi.models.base import DetectionModel
from sahi.utils.coco import CocoAnnotation, CocoPrediction
from sahi.utils.cv import Colors, apply_color_mask, read_image_as_pil, visualize_object_predictions


class BoundingBoxes:
    def __init__(self, bboxes: np.ndarray, scores: np.ndarray = None, labels: np.ndarray = None):
        """
        Arguments:
            bboxes: numpy array of shape (N, 4) where N is number of bounding boxes.
                Each bounding box is in the form of [minx, miny, maxx, maxy]
            scores: numpy array of shape (N, 1) where N is number of bounding boxes.
                Each score is in the form of [score]
            labels: numpy array of shape (N, 1) where N is number of bounding boxes.
                Each label is in the form of [label]
        """
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)
        elif type(bboxes).__module__ == "torch":
            bboxes = bboxes.numpy(force=True)

        if not isinstance(bboxes, np.ndarray):
            raise TypeError(f"bboxes should be a numpy array or list, got {type(bboxes)}")

        if bboxes.ndim != 2:
            raise ValueError("bboxes should be 2D array")
        if bboxes.shape[1] != 4:
            raise ValueError("bboxes should be of shape (N, 4)")

        if scores is not None:
            if isinstance(scores, list):
                scores = np.array(scores)
            elif type(scores).__module__ == "torch":
                scores = scores.numpy(force=True)

            if not isinstance(scores, np.ndarray):
                raise TypeError(f"scores should be a numpy array or list, got {type(scores)}")

            if scores.ndim != 2:
                raise ValueError("scores should be 2D array")
            if scores.shape[1] != 1:
                raise ValueError("scores should be of shape (N, 1)")

            if scores.shape[0] != bboxes.shape[0]:
                raise ValueError("scores and bboxes should have same number of rows")

        if labels is not None:
            if isinstance(labels, list):
                labels = np.array(labels)
            elif type(labels).__module__ == "torch":
                labels = labels.numpy(force=True)

            if not isinstance(labels, np.ndarray):
                raise TypeError(f"labels should be a numpy array or list, got {type(labels)}")

            if labels.ndim != 2:
                raise ValueError("labels should be 2D array")
            if labels.shape[1] != 1:
                raise ValueError("labels should be of shape (N, 1)")

            if labels.shape[0] != bboxes.shape[0]:
                raise ValueError("labels and bboxes should have same number of rows")

        self.bboxes = bboxes
        self.scores = scores
        self.labels = labels

    def __len__(self):
        return len(self.bboxes)

    def __repr__(self):
        return f"BoundingBoxes: <number of boxes: {len(self)}>"

    def to_dict(self):
        return {
            "bboxes": self.bboxes.tolist(),
            "scores": self.scores.tolist() if self.scores is not None else None,
            "labels": self.labels.tolist() if self.labels is not None else None,
        }

    def to_xyxys(self):
        """
        Returns bboxes in [xmin, ymin, xmax, ymax, score] format

        Returns:
            numpy array of shape (N, 5) where N is number of bounding boxes.
        """

        if self.scores is None:
            raise ValueError("scores is None")

        return np.hstack((self.bboxes, self.scores))

    def to_xyxysl(self):
        """
        Returns bboxes in [xmin, ymin, xmax, ymax, score, label] format

        Returns:
            numpy array of shape (N, 6) where N is number of bounding boxes.
        """

        if self.scores is None:
            raise ValueError("scores is None")

        if self.labels is None:
            raise ValueError("labels is None")

        return np.hstack((self.bboxes, self.scores, self.labels))

    def to_xywh(self):
        """
        Returns bboxes in [xmin, ymin, width, height] format

        Returns:
            numpy array of shape (N, 4) where N is number of bounding boxes.
        """
        return np.hstack(
            (
                self.bboxes[:, :2],
                self.bboxes[:, 2:] - self.bboxes[:, :2],
            )
        )

    def to_xywhs(self):
        """
        Returns bboxes in [xmin, ymin, width, height, score] format

        Returns:
            numpy array of shape (N, 5) where N is number of bounding boxes.
        """

        if self.scores is None:
            raise ValueError("scores is None")

        return np.hstack((self.to_xywh(), self.scores))

    def to_xywhsl(self):
        """
        Returns bboxes in [xmin, ymin, width, height, score, label] format

        Returns:
            numpy array of shape (N, 6) where N is number of bounding boxes.
        """

        if self.scores is None:
            raise ValueError("scores is None")

        if self.labels is None:
            raise ValueError("labels is None")

        return np.hstack((self.to_xywh(), self.scores, self.labels))

    def remap(self, offset_amount: List[int], full_shape: List[int], inplace: bool = False):
        """
        Remaps the bounding boxes from sliced image to full sized image.

        Arguments:
            offset_amount: list
                To remap the box and mask predictions from sliced image
                to full sized image, should be in the form of [offset_x, offset_y]
            full_shape: list
                Size of the full image after remapping, should be in
                the form of [height, width]
        """
        offset_amount = np.array(offset_amount)
        full_shape = np.array(full_shape)

        if not inplace:
            bboxes = deepcopy(self.bboxes)

            bboxes[:, :2] += offset_amount
            bboxes[:, 2:] += offset_amount

            bboxes[:, 0] = np.clip(bboxes[:, 0], 0, full_shape[1])
            bboxes[:, 1] = np.clip(bboxes[:, 1], 0, full_shape[0])
            bboxes[:, 2] = np.clip(bboxes[:, 2], 0, full_shape[1])
            bboxes[:, 3] = np.clip(bboxes[:, 3], 0, full_shape[0])

            return BoundingBoxes(bboxes, self.scores)

        else:
            self.bboxes[:, :2] += offset_amount
            self.bboxes[:, 2:] += offset_amount

            self.bboxes[:, 0] = np.clip(self.bboxes[:, 0], 0, full_shape[1])
            self.bboxes[:, 1] = np.clip(self.bboxes[:, 1], 0, full_shape[0])
            self.bboxes[:, 2] = np.clip(self.bboxes[:, 2], 0, full_shape[1])
            self.bboxes[:, 3] = np.clip(self.bboxes[:, 3], 0, full_shape[0])

            return self

    def __array__(self):
        if self.scores is None and self.labels is None:
            return self.bboxes
        elif self.scores is not None and self.labels is None:
            return self.to_xyxys()
        elif self.scores is not None and self.labels is not None:
            return self.to_xyxysl()
        else:
            raise ValueError("Invalid combination of scores and labels")


class Masks:
    def __init__(self, masks: np.ndarray):
        """
        Arguments:
            masks: numpy array of shape (N, H, W) where N is number of masks.
        """
        if isinstance(masks, list):
            masks = np.array(masks)
        if not isinstance(masks, np.ndarray):
            raise TypeError(f"masks should be a numpy array or list, got {type(masks)}")

        if masks.ndim != 3:
            raise ValueError("masks should be 3D array")
        if masks.shape[0] == 0:
            raise ValueError("masks should have atleast one mask")

    @property
    def masks(self) -> List[Mask]:
        return self._masks

    @masks.setter
    def masks(self, masks):
        masks = []
        for mask in masks:
            masks.append(Mask(mask))

        self._masks = masks

    def _extend_masks(self, masks: np.ndarray):
        for mask in masks:
            self._masks.append(Mask(mask))

    def __len__(self):
        return len(self.masks)

    def __repr__(self):
        return f"Masks: <number of masks: {len(self)}>"

    def __getitem__(self, idx) -> Mask:

        if isinstance(idx, int):
            return self.masks[idx]

    def __add__(self, other) -> "Masks":
        if not isinstance(other, Masks):
            raise TypeError(f"other should be of type Masks, got {type(other)}")
        new_instance = deepcopy(self)
        new_instance._extend_masks(other.masks)
        return new_instance

    def remap(self, offset_amount: List[int], full_shape: List[int], inplace: bool = False):
        """
        Remaps the masks from sliced image to full sized image.

        Arguments:
            offset_amount: list
                To remap the mask predictions from sliced image
                to full sized image, should be in the form of [offset_x, offset_y]
            full_shape: list
                Size of the full image after remapping, should be in
                the form of [height, width]
            inplace: bool
                If True, remaps the masks in place, else returns a new Masks object.

        Returns:
            Masks object
        """

        # empty masks
        if self.masks is None:
            return self.masks

        remapped_masks = []
        for mask in self.masks:
            mask.offset_amount = offset_amount
            mask.full_shape = full_shape
            remapped_masks.append(mask.remap(inplace=inplace))

        if not inplace:
            return Masks(remapped_masks)
        else:
            self.masks = remapped_masks
            return self


class PredictionResult:
    def __init__(
        self,
        image: Union[Image.Image, str, np.ndarray],
        bboxes: BoundingBoxes,
        masks: Masks = None,
        durations_in_seconds: Optional[Dict] = None,
        label_id_to_name: Optional[Dict] = None,
    ):
        self.image: Image.Image = read_image_as_pil(image)
        self.image_width, self.image_height = self.image.size
        self.durations_in_seconds = durations_in_seconds
        self.label_id_to_name = label_id_to_name

        self.bboxes = bboxes
        self.masks = masks

    def export_visuals(
        self,
        export_dir: str,
        text_size: float = None,
        rect_th: int = None,
        file_name: str = "prediction_visual",
        export_format: str = "jpg",
    ):
        """
        Exports the prediction visuals to the specified directory.

        Args:
            export_dir: directory for resulting visualization to be exported
            text_size: size of the category name over box
            rect_th: rectangle thickness
            file_name: saving name
            export_format: format of the exported image
        """
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        visualize_predictions(
            image=np.ascontiguousarray(self.image),
            bboxes=self.bboxes,
            masks=self.masks,
            rect_th=rect_th,
            text_size=text_size,
            text_th=None,
            color=None,
            label_id_to_name=self.label_id_to_name,
            output_dir=export_dir,
            file_name=file_name,
            export_format=export_format,
        )

    def to_coco_annotations(self):
        """
        Returns the predictions bboxes and masks as a COCO dict.

        Returns:
            dict
        """

        coco_annotations = []

        xywhsl = self.bboxes.to_xywhsl()

        for ind in range(len(self.bboxes)):
            label_id = xywhsl[ind][5]
            if self.masks is not None:
                coco_annotation = CocoAnnotation.from_coco_segmentation(
                    segmentation=self.masks[ind].to_coco_segmentation(),
                    category_id=label_id,
                    category_name=self.label_id_to_name[label_id] if self.label_id_to_name else str(label_id),
                )
            else:
                coco_annotation = CocoAnnotation.from_coco_bbox(
                    bbox=xywhsl[ind][:4],
                    category_id=label_id,
                    category_name=self.label_id_to_name[label_id] if self.label_id_to_name else str(label_id),
                )
            coco_annotations.append(coco_annotation)

        return coco_annotations

    def to_coco_predictions(self, image_id: Optional[int] = None):
        coco_predictions = []

        xywhsl = self.bboxes.to_xywhsl()
        for ind in range(len(self.bboxes)):
            label_id = xywhsl[ind][5]
            label_name = self.label_id_to_name[label_id] if self.label_id_to_name else str(label_id)
            xywh = xywhsl[ind][:4]
            if self.masks is not None:
                coco_prediction = CocoPrediction.from_coco_segmentation(
                    segmentation=self.masks[ind].to_coco_segmentation(),
                    category_id=label_id,
                    category_name=label_name,
                    image_id=image_id,
                )
            else:
                coco_prediction = CocoPrediction.from_coco_bbox(
                    bbox=xywh,
                    category_id=label_id,
                    category_name=label_name,
                    image_id=image_id,
                )
            coco_predictions.append(coco_prediction)
        return coco_predictions


def batch_predict(
    images,
    detection_model: DetectionModel,
    batch_size: int = 1,
    verbose: int = 0,
):
    """
    Function for performing batch prediction for given image using given detection_model.

    Arguments:
        images: List[str, np.ndarray]
            List of image file paths or numpy images to predict
        detection_model: model.DetectionModel
            Detection model to use for prediction
        batch_size: int
            Batch size for prediction
        verbose: int
            0: no print (default)
            1: print prediction duration

    Returns:
        A dict with fields:
            object_predictions: a list of ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    pass


def visualize_predictions(
    image: np.ndarray,
    bboxes: BoundingBoxes,
    masks: Masks = None,
    label_id_to_name: Dict[int, str] = None,
    output_dir: str = None,
    file_name: str = "prediction_visual",
    export_format: str = "png",
    color: tuple = None,
    rect_th: int = None,
    text_size: float = None,
    text_th: float = None,
):
    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.001), 1)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3
    # add bbox and mask to image if present
    for ind in range(len(bboxes)):
        bbox = bboxes.to_xyxysl()
        score = bbox[4]
        label_id = bbox[5]
        label_name = label_id_to_name[label_id] if label_id_to_name else str(label_id)

        # set color
        if colors is not None:
            color = colors(label_id)
        # visualize masks if present
        if masks is not None:
            # deepcopy mask so that original is not altered
            mask = copy.deepcopy(np.array(masks[ind]))
            # draw mask
            rgb_mask = apply_color_mask(mask, color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.4, 0)
        # set bbox points
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        # visualize boxes
        cv2.rectangle(
            image,
            p1,
            p2,
            color=color,
            thickness=rect_th,
        )
        # arange bounding box text location
        label = f"{label_name} {score:.2f}"
        w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # add bounding box text
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            text_size,
            (255, 255, 255),
            thickness=text_th,
        )

    # export if output_dir is present
    if output_dir is not None:
        # export image with predictions
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(Path(output_dir) / (file_name + "." + export_format))
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}
