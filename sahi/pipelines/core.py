# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2022.

import copy
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image

from sahi.annotation import Mask
from sahi.pipelines.utils import visualize_predictions
from sahi.utils.coco import CocoAnnotation, CocoPrediction
from sahi.utils.cv import read_image_as_pil
from sahi.utils.torch import tensor_to_numpy


class BoundingBoxes:
    def __init__(self, bboxes: np.ndarray, scores: np.ndarray = None, labels: np.ndarray = None):
        """
        Arguments:
            bboxes: numpy array of shape (N, 4) where N is number of bounding boxes.
                Each bounding box is in the form of [minx, miny, maxx, maxy]
            scores: numpy array of shape (N) where N is number of bounding boxes.
                Each score is in the form of [score]
            labels: numpy array of shape (N) where N is number of bounding boxes.
                Each label is in the form of [label]
        """
        bboxes, scores, labels = self._sanity_check(bboxes=bboxes, scores=scores, labels=labels)

        self.bboxes = bboxes
        self.scores = scores
        self.labels = labels

    def _sanity_check(self, bboxes: np.ndarray, scores: np.ndarray, labels: np.ndarray):
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)
        elif type(bboxes).__module__ == "torch":
            bboxes = tensor_to_numpy(bboxes)

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
                scores = tensor_to_numpy(scores)

            if not isinstance(scores, np.ndarray):
                raise TypeError(f"scores should be a numpy array or list, got {type(scores)}")

            if scores.ndim != 1:
                raise ValueError("scores should be 1D array")

            if scores.shape[0] != bboxes.shape[0]:
                raise ValueError("scores and bboxes should have same number of rows")

        if labels is not None:
            if isinstance(labels, list):
                labels = np.array(labels)
            elif type(labels).__module__ == "torch":
                labels = tensor_to_numpy(labels)

            if not isinstance(labels, np.ndarray):
                raise TypeError(f"labels should be a numpy array or list, got {type(labels)}")

            if labels.ndim != 1:
                raise ValueError("labels should be 1D array")

            if labels.shape[0] != bboxes.shape[0]:
                raise ValueError("labels and bboxes should have same number of rows")

        return bboxes, scores, labels

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

    def __add__(self, other):
        if not isinstance(other, BoundingBoxes):
            raise TypeError(f"other should be of type BoundingBoxes, got {type(other)}")

        bboxes = np.vstack((self.bboxes, other.bboxes))
        scores = np.hstack((self.scores, other.scores)) if self.scores is not None else None
        labels = np.hstack((self.labels, other.labels)) if self.labels is not None else None

        return BoundingBoxes(bboxes=bboxes, scores=scores, labels=labels)

    def to_xyxys(self):
        """
        Returns bboxes in [xmin, ymin, xmax, ymax, score] format

        Returns:
            numpy array of shape (N, 5) where N is number of bounding boxes.
        """

        if self.scores is None:
            raise ValueError("scores is None")

        return np.hstack((self.bboxes, np.expand_dims(self.scores, axis=1)))

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

        return np.hstack((self.bboxes, np.expand_dims(self.scores, axis=1), np.expand_dims(self.labels, axis=1)))

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

        return np.hstack((self.to_xywh(), np.expand_dims(self.scores, axis=1)))

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

        return np.hstack((self.to_xywh(), np.expand_dims(self.scores, axis=1), np.expand_dims(self.labels, axis=1)))

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
            bboxes = copy.deepcopy(self.bboxes)

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
        new_instance = copy.deepcopy(self)
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
        bboxes: BoundingBoxes,
        masks: Masks = None,
        image: Union[Image.Image, str, np.ndarray, None] = None,
        label_id_to_name: Optional[Dict] = None,
    ):
        self.image = read_image_as_pil(image) if image is not None else None
        self.image_width, self.image_height = self.image.size if self.image is not None else (None, None)
        self.label_id_to_name = label_id_to_name

        self.bboxes = bboxes
        self.masks = masks

        self.image_id = None
        self.offset_amount = None
        self.full_shape = None
        self.filepath = None

    def remap(self):
        """
        Remaps the prediction result from sliced image to full sized image.

        Returns:
            PredictionResult object
        """
        self.bboxes.remap(self.offset_amount, self.full_shape, inplace=True)
        if self.masks is not None:
            self.masks.remap(self.offset_amount, self.full_shape, inplace=True)

        return self

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
        image = self.image if self.image is not None else read_image_as_pil(self.filepath)
        visualize_predictions(
            image=np.ascontiguousarray(image),
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

    def postprocess(self, type: str = "batched_nms", iou_threshold: float = 0.5, inplace: bool = False):
        """
        Postprocesses the prediction results.

        Args:
            type: type of postprocessing to be applied
            iou_threshold: threshold for match IoU
        """
        return postprocess_prediction_result(
            prediction_result=self, postprocess_type=type, postprocess_match_threshold=iou_threshold, inplace=inplace
        )

    def __add__(self, other):
        if not isinstance(other, PredictionResult):
            raise TypeError(f"other should be of type PredictionResult, got {type(other)}")
        new_instance = copy.deepcopy(self)
        new_instance.bboxes = new_instance.bboxes + other.bboxes
        if (
            new_instance.masks is not None
            and other.masks is not None
            and len(new_instance.masks) > 0
            and len(other.masks) > 0
        ):
            new_instance.masks = new_instance.masks + other.masks
        elif new_instance.masks is None and other.masks is not None and len(other.masks) > 0:
            new_instance.masks = other.masks
        else:
            new_instance.masks = new_instance.masks
        return new_instance


def postprocess_prediction_result(
    prediction_result: PredictionResult,
    postprocess_type: str = "batched_nms",
    postprocess_match_threshold: float = 0.5,
    inplace: bool = False,
) -> PredictionResult:
    """
    Postprocesses the prediction results.

    Args:
        prediction_result: prediction result to be postprocessed
        postprocess_type: type of postprocessing to be applied
        postprocess_match_threshold: threshold for match IoU
        inplace: whether to perform the postprocessing in-place

    Returns:
        PredictionResult: postprocessed prediction result
    """

    import torch
    from torchvision.ops import batched_nms, nms

    if prediction_result.bboxes is None or len(prediction_result.bboxes) == 0:
        return prediction_result

    xyxysl = torch.from_numpy(prediction_result.bboxes.to_xyxysl())

    bboxes = xyxysl[:, :4]
    scores = xyxysl[:, 4]
    labels = xyxysl[:, 5]

    if postprocess_type == "batched_nms":
        keep = batched_nms(bboxes, scores, labels, postprocess_match_threshold)
    elif postprocess_type == "nms":
        keep = nms(bboxes, scores, postprocess_match_threshold)
    else:
        raise ValueError(f"postprocess_type should be either batched_nms or nms, got {postprocess_type}")

    if inplace:
        prediction_result.bboxes = BoundingBoxes(bboxes=bboxes[keep], labels=labels[keep], scores=scores[keep])
        if prediction_result.masks is not None:
            prediction_result.masks = prediction_result.masks[keep.tolist()]
    else:
        prediction_result = copy.deepcopy(prediction_result)
        prediction_result.bboxes = BoundingBoxes(bboxes=bboxes[keep], labels=labels[keep], scores=scores[keep])
        if prediction_result.masks is not None:
            prediction_result.masks = prediction_result.masks[keep.tolist()]

    return prediction_result
