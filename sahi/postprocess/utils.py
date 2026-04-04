from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.collection import GeometryCollection

from sahi.annotation import BoundingBox, Category, Mask
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import is_available
from sahi.utils.shapely import ShapelyAnnotation, get_shapely_multipolygon


def _is_tensor_like(obj: Any) -> bool:
    """Check if an object is a torch Tensor or numpy array (without importing torch)."""
    return isinstance(obj, np.ndarray) or (hasattr(obj, "tolist") and not isinstance(obj, (int, float, list, tuple)))


class ObjectPredictionList(Sequence):
    """Sequence wrapper around a list of ObjectPrediction instances.

    Provides indexing by int, list, or tensor-like objects, and conversion
    to numpy arrays or torch tensors for batch postprocessing operations.

    Args:
        prediction_list: List of ObjectPrediction instances to wrap.
    """

    def __init__(self, prediction_list: list[ObjectPrediction]) -> None:
        self.list: list[ObjectPrediction] = prediction_list
        super().__init__()

    def __getitem__(self, i) -> ObjectPredictionList:
        """Retrieve predictions by index, list of indices, or tensor-like.

        Args:
            i: An integer index, list/tuple of indices, or tensor-like
                object convertible via ``.tolist()``.

        Returns:
            A new ObjectPredictionList containing the selected predictions.
        """
        if _is_tensor_like(i):
            i = i.tolist()
        if isinstance(i, int):
            return ObjectPredictionList([self.list[i]])
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(self.list.__getitem__, i)
            return ObjectPredictionList(list(accessed_mapping))
        else:
            raise NotImplementedError(f"{type(i)}")

    def __setitem__(self, i, elem) -> None:
        """Set predictions at the given index or indices.

        Args:
            i: An integer index, list/tuple of indices, or tensor-like.
            elem: An ObjectPrediction, ObjectPredictionList, or list of
                ObjectPrediction instances to assign.
        """
        if _is_tensor_like(i):
            i = i.tolist()
        if isinstance(i, int):
            self.list[i] = elem
        elif isinstance(i, (tuple, list)):
            if len(i) != len(elem):
                raise ValueError()
            if isinstance(elem, ObjectPredictionList):
                for ind, el in enumerate(elem.list):
                    self.list[i[ind]] = el
            else:
                for ind, el in enumerate(elem):
                    self.list[i[ind]] = el
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self) -> int:
        return len(self.list)

    def __str__(self) -> str:
        return str(self.list)

    def extend(self, object_prediction_list: ObjectPredictionList) -> None:
        """Extend this list with predictions from another ObjectPredictionList.

        Args:
            object_prediction_list: The list whose predictions to append.
        """
        self.list.extend(object_prediction_list.list)

    def totensor(self):
        """Convert to torch.Tensor. Requires torch to be installed."""
        return object_prediction_list_to_torch(self)

    def tonumpy(self) -> np.ndarray:
        """Convert to a numpy array of shape (N, 6).

        Returns:
            np.ndarray with columns [x1, y1, x2, y2, score, category_id].
        """
        return object_prediction_list_to_numpy(self)

    def tolist(self) -> ObjectPrediction | list[ObjectPrediction]:
        """Unwrap to a single ObjectPrediction or a list.

        Returns:
            A single ObjectPrediction if the list has one element,
            otherwise the full list of ObjectPrediction instances.
        """
        if len(self.list) == 1:
            return self.list[0]
        else:
            return self.list


def repair_polygon(shapely_polygon: Polygon) -> Polygon:
    """Attempt to fix an invalid Shapely polygon using a zero-width buffer.

    If the repaired result is a MultiPolygon or GeometryCollection, the
    polygon with the largest area is returned.

    Args:
        shapely_polygon: A Shapely Polygon that may be invalid.

    Returns:
        A valid Polygon, or the original if it was already valid or
        could not be repaired.
    """
    if not shapely_polygon.is_valid:
        fixed_polygon = shapely_polygon.buffer(0)
        if fixed_polygon.is_valid:
            if isinstance(fixed_polygon, Polygon):
                return fixed_polygon
            elif isinstance(fixed_polygon, MultiPolygon):
                return max(fixed_polygon.geoms, key=lambda p: p.area)
            elif isinstance(fixed_polygon, GeometryCollection):
                polygons = [geom for geom in fixed_polygon.geoms if isinstance(geom, Polygon)]
                return max(polygons, key=lambda p: p.area) if polygons else shapely_polygon

    return shapely_polygon


def repair_multipolygon(shapely_multipolygon: MultiPolygon) -> MultiPolygon:
    """Attempt to fix an invalid Shapely MultiPolygon using a zero-width buffer.

    If the repaired result is a single Polygon, it is wrapped in a
    MultiPolygon. GeometryCollection results are filtered to polygons only.

    Args:
        shapely_multipolygon: A Shapely MultiPolygon that may be invalid.

    Returns:
        A valid MultiPolygon, or the original if it was already valid or
        could not be repaired.
    """
    if not shapely_multipolygon.is_valid:
        fixed_geometry = shapely_multipolygon.buffer(0)

        if fixed_geometry.is_valid:
            if isinstance(fixed_geometry, MultiPolygon):
                return fixed_geometry
            elif isinstance(fixed_geometry, Polygon):
                return MultiPolygon([fixed_geometry])
            elif isinstance(fixed_geometry, GeometryCollection):
                polygons = [geom for geom in fixed_geometry.geoms if isinstance(geom, Polygon)]
                return MultiPolygon(polygons) if polygons else shapely_multipolygon

    return shapely_multipolygon


def coco_segmentation_to_shapely(segmentation: list | list[list]) -> MultiPolygon:
    """Fix segment data in COCO format :param segmentation: segment data in COCO format :return:"""
    if isinstance(segmentation, list) and all([not isinstance(seg, list) for seg in segmentation]):
        segmentation = [segmentation]
    elif isinstance(segmentation, list) and all([isinstance(seg, list) for seg in segmentation]):
        pass
    else:
        raise ValueError("segmentation must be List or List[List]")

    polygon_list = []

    for coco_polygon in segmentation:
        point_list = list(zip(coco_polygon[::2], coco_polygon[1::2]))
        shapely_polygon = Polygon(point_list)
        polygon_list.append(repair_polygon(shapely_polygon))

    shapely_multipolygon = repair_multipolygon(MultiPolygon(polygon_list))
    return shapely_multipolygon


def object_prediction_list_to_torch(object_prediction_list: ObjectPredictionList) -> Any:
    """Convert to torch.Tensor. Requires torch to be installed.

    Returns:
        torch.Tensor of size N x [x1, y1, x2, y2, score, category_id]
    """
    if not is_available("torch"):
        raise ImportError("torch is required for totensor(). Install it with: pip install sahi[torch]")
    import torch

    num_predictions = len(object_prediction_list)
    torch_predictions = torch.zeros([num_predictions, 6], dtype=torch.float32)
    for ind, object_prediction in enumerate(object_prediction_list):
        torch_predictions[ind, :4] = torch.tensor(object_prediction.tolist().bbox.to_xyxy(), dtype=torch.float32)
        torch_predictions[ind, 4] = object_prediction.tolist().score.value
        torch_predictions[ind, 5] = object_prediction.tolist().category.id
    return torch_predictions


def object_prediction_list_to_numpy(object_prediction_list: ObjectPredictionList) -> np.ndarray:
    """Convert an ObjectPredictionList to a numpy array.

    Args:
        object_prediction_list: The predictions to convert.

    Returns:
        np.ndarray of shape (N, 6) with columns
        [x1, y1, x2, y2, score, category_id].
    """
    num_predictions = len(object_prediction_list)
    numpy_predictions = np.zeros([num_predictions, 6], dtype=np.float32)
    for ind, object_prediction in enumerate(object_prediction_list):
        numpy_predictions[ind, :4] = np.array(object_prediction.tolist().bbox.to_xyxy(), dtype=np.float32)
        numpy_predictions[ind, 4] = object_prediction.tolist().score.value
        numpy_predictions[ind, 5] = object_prediction.tolist().category.id
    return numpy_predictions


def calculate_box_union(box1: list[int] | np.ndarray, box2: list[int] | np.ndarray) -> list[int]:
    """Compute the smallest bounding box enclosing both input boxes.

    Args:
        box1: First box as [x1, y1, x2, y2].
        box2: Second box as [x1, y1, x2, y2].

    Returns:
        The union bounding box as [x1, y1, x2, y2].
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    left_top = np.minimum(box1[:2], box2[:2])
    right_bottom = np.maximum(box1[2:], box2[2:])
    return list(np.concatenate((left_top, right_bottom)))


def calculate_area(box: list[int] | np.ndarray) -> float:
    """Compute the area of an axis-aligned bounding box.

    Args:
        box: Bounding box as [x1, y1, x2, y2].

    Returns:
        The area of the box (width * height).
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def calculate_intersection_area(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute the intersection area of two axis-aligned bounding boxes.

    Args:
        box1: First box as np.array([x1, y1, x2, y2]).
        box2: Second box as np.array([x1, y1, x2, y2]).

    Returns:
        The area of the intersection region, or 0 if the boxes do not
        overlap.
    """
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    width_height = (right_bottom - left_top).clip(min=0)
    return width_height[0] * width_height[1]


def calculate_bbox_iou(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
    """Compute Intersection over Union (IoU) between two predictions.

    Args:
        pred1: First object prediction.
        pred2: Second object prediction.

    Returns:
        The IoU value in [0, 1].
    """
    box1 = np.array(pred1.bbox.to_xyxy())
    box2 = np.array(pred2.bbox.to_xyxy())
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    intersect = calculate_intersection_area(box1, box2)
    return intersect / (area1 + area2 - intersect)


def calculate_bbox_ios(pred1: ObjectPrediction, pred2: ObjectPrediction) -> float:
    """Compute Intersection over Smaller (IoS) between two predictions.

    Args:
        pred1: First object prediction.
        pred2: Second object prediction.

    Returns:
        The IoS value in [0, 1], where the denominator is the area of
        the smaller bounding box.
    """
    box1 = np.array(pred1.bbox.to_xyxy())
    box2 = np.array(pred2.bbox.to_xyxy())
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    intersect = calculate_intersection_area(box1, box2)
    smaller_area = np.minimum(area1, area2)
    return intersect / smaller_area


def has_match(
    pred1: ObjectPrediction, pred2: ObjectPrediction, match_type: str = "IOU", match_threshold: float = 0.5
) -> bool:
    """Check whether two predictions overlap above the given threshold.

    Args:
        pred1: First object prediction.
        pred2: Second object prediction.
        match_type: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum overlap to count as a match.

    Returns:
        True if the overlap exceeds match_threshold.

    Raises:
        ValueError: If match_type is not "IOU" or "IOS".
    """
    if match_type == "IOU":
        threshold_condition = calculate_bbox_iou(pred1, pred2) > match_threshold
    elif match_type == "IOS":
        threshold_condition = calculate_bbox_ios(pred1, pred2) > match_threshold
    else:
        raise ValueError()
    return threshold_condition


def get_merged_mask(pred1: ObjectPrediction, pred2: ObjectPrediction) -> Mask:
    """Compute the union of two prediction masks.

    Args:
        pred1: First object prediction with a valid mask.
        pred2: Second object prediction with a valid mask.

    Returns:
        A new Mask representing the geometric union of both masks.
    """
    mask1 = pred1.mask
    mask2 = pred2.mask

    # buffer(0) is a quickhack to fix invalid polygons most of the time
    poly1 = get_shapely_multipolygon(mask1.segmentation).buffer(0)
    poly2 = get_shapely_multipolygon(mask2.segmentation).buffer(0)

    if poly1.is_empty:
        poly1 = coco_segmentation_to_shapely(mask1.segmentation)
    if poly2.is_empty:
        poly2 = coco_segmentation_to_shapely(mask2.segmentation)

    union_poly = poly1.union(poly2)
    if not hasattr(union_poly, "geoms"):
        union_poly = MultiPolygon([union_poly])
    else:
        union_poly = MultiPolygon([g.buffer(0) for g in union_poly.geoms if isinstance(g, Polygon)])
    union = ShapelyAnnotation(multipolygon=union_poly).to_coco_segmentation()
    return Mask(
        segmentation=union,
        full_shape=mask1.full_shape,
        shift_amount=mask1.shift_amount,
    )


def get_merged_score(
    pred1: ObjectPrediction,
    pred2: ObjectPrediction,
) -> float:
    """Return the higher confidence score from two predictions.

    Args:
        pred1: First object prediction.
        pred2: Second object prediction.

    Returns:
        The maximum score value.
    """
    scores: list[float] = [pred.score.value for pred in (pred1, pred2)]
    return max(scores)


def get_merged_bbox(pred1: ObjectPrediction, pred2: ObjectPrediction) -> BoundingBox:
    """Compute the union bounding box of two predictions.

    Args:
        pred1: First object prediction.
        pred2: Second object prediction.

    Returns:
        A BoundingBox enclosing both input bounding boxes.
    """
    box1: list[int] = pred1.bbox.to_xyxy()
    box2: list[int] = pred2.bbox.to_xyxy()
    bbox = BoundingBox(box=calculate_box_union(box1, box2))
    return bbox


def get_merged_category(pred1: ObjectPrediction, pred2: ObjectPrediction) -> Category:
    """Return the category of the higher-scored prediction.

    Args:
        pred1: First object prediction.
        pred2: Second object prediction.

    Returns:
        The Category from whichever prediction has the higher score.
    """
    if pred1.score.value > pred2.score.value:
        return pred1.category
    else:
        return pred2.category


def merge_object_prediction_pair(
    pred1: ObjectPrediction,
    pred2: ObjectPrediction,
) -> ObjectPrediction:
    """Merge two overlapping predictions into a single prediction.

    Combines bounding boxes (union), masks (geometric union), scores
    (maximum), and categories (from the higher-scored prediction).

    Args:
        pred1: First object prediction.
        pred2: Second object prediction.

    Returns:
        A new ObjectPrediction with merged attributes.
    """
    shift_amount = pred1.bbox.shift_amount
    merged_bbox: BoundingBox = get_merged_bbox(pred1, pred2)
    merged_score: float = get_merged_score(pred1, pred2)
    merged_category: Category = get_merged_category(pred1, pred2)
    if pred1.mask and pred2.mask:
        merged_mask: Mask = get_merged_mask(pred1, pred2)
        segmentation = merged_mask.segmentation
        full_shape = merged_mask.full_shape
    else:
        segmentation = None
        full_shape = None
    return ObjectPrediction(
        bbox=merged_bbox.to_xyxy(),
        score=merged_score,
        category_id=merged_category.id,
        category_name=merged_category.name,
        segmentation=segmentation,
        shift_amount=shift_amount,
        full_shape=full_shape,
    )
