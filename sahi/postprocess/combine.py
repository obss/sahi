from __future__ import annotations

import torch
from shapely.geometry import box
from shapely.strtree import STRtree

from sahi.logger import logger
from sahi.postprocess.utils import ObjectPredictionList, has_match, merge_object_prediction_pair
from sahi.prediction import ObjectPrediction
from sahi.utils.package_utils import check_requirements


def batched_nms(predictions: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5):
    """Apply non-maximum suppression to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    scores = predictions[:, 4].squeeze()
    category_ids = predictions[:, 5].squeeze()
    keep_mask = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_indices = nms(predictions[curr_indices], match_metric, match_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    # sort selected indices by their scores
    keep_indices = keep_indices[scores[keep_indices].sort(descending=True)[1]].tolist()
    return keep_indices


def nms(
    predictions: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Optimized non-maximum suppression for axis-aligned bounding boxes using STRTree.

    Args:
        predictions: (tensor) The location preds for the image along with the class
            predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.

    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    # Extract coordinates and scores as tensors
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]
    scores = predictions[:, 4]

    # Calculate areas as tensor (vectorized operation)
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes only once
    boxes = []
    for i in range(len(predictions)):
        boxes.append(
            box(
                x1[i].item(),  # Convert only individual values
                y1[i].item(),
                x2[i].item(),
                y2[i].item(),
            )
        )

    # Sort indices by score (descending) using torch
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    # Build STRtree
    tree = STRtree(boxes)

    keep = []
    suppressed = set()

    for current_idx in sorted_idxs:
        if current_idx in suppressed:
            continue

        keep.append(current_idx)
        current_box = boxes[current_idx]
        current_area = areas[current_idx].item()  # Convert only when needed

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx or candidate_idx in suppressed:
                continue

            # Skip candidates with higher scores (already processed)
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                # Use box coordinates for stable ordering
                current_coords = (
                    x1[current_idx].item(),
                    y1[current_idx].item(),
                    x2[current_idx].item(),
                    y2[current_idx].item(),
                )
                candidate_coords = (
                    x1[candidate_idx].item(),
                    y1[candidate_idx].item(),
                    x2[candidate_idx].item(),
                    y2[candidate_idx].item(),
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError("Invalid match_metric")

            # Suppress if overlap exceeds threshold
            if metric >= match_threshold:
                suppressed.add(candidate_idx)

    return keep


def batched_greedy_nmm(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """Apply greedy version of non-maximum merging per category to avoid detecting too many overlapping bounding boxes
    for a given object.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = greedy_nmm(object_predictions_as_tensor[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()
        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [curr_indices_list[curr_merge_ind] for curr_merge_ind in curr_merge_list]
            keep_to_merge_list[keep] = merge_list
    return keep_to_merge_list


def greedy_nmm(
    object_predictions_as_tensor: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Optimized greedy non-maximum merging for axis-aligned bounding boxes using STRTree.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
    Returns:
        keep_to_merge_list: (dict[int, list[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    # Extract coordinates and scores as tensors
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]
    scores = object_predictions_as_tensor[:, 4]

    # Calculate areas as tensor (vectorized operation)
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes only once
    boxes = []
    for i in range(len(object_predictions_as_tensor)):
        boxes.append(
            box(
                x1[i].item(),  # Convert only individual values
                y1[i].item(),
                x2[i].item(),
                y2[i].item(),
            )
        )

    # Sort indices by score (descending) using torch
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    # Build STRtree
    tree = STRtree(boxes)

    keep_to_merge_list = {}
    suppressed = set()

    for current_idx in sorted_idxs:
        if current_idx in suppressed:
            continue

        current_box = boxes[current_idx]
        current_area = areas[current_idx].item()  # Convert only when needed

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        merge_list = []
        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx or candidate_idx in suppressed:
                continue

            # Only consider candidates with lower or equal score
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                # Use box coordinates for stable ordering
                current_coords = (
                    x1[current_idx].item(),
                    y1[current_idx].item(),
                    x2[current_idx].item(),
                    y2[current_idx].item(),
                )
                candidate_coords = (
                    x1[candidate_idx].item(),
                    y1[candidate_idx].item(),
                    x2[candidate_idx].item(),
                    y2[candidate_idx].item(),
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError("Invalid match_metric")

            # Add to merge list if overlap exceeds threshold
            if metric >= match_threshold:
                merge_list.append(candidate_idx)
                suppressed.add(candidate_idx)

        keep_to_merge_list[int(current_idx)] = [int(idx) for idx in merge_list]

    return keep_to_merge_list


def batched_nmm(
    object_predictions_as_tensor: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """Apply non-maximum merging per category to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = nmm(object_predictions_as_tensor[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()
        for curr_keep, curr_merge_list in curr_keep_to_merge_list.items():
            keep = curr_indices_list[curr_keep]
            merge_list = [curr_indices_list[curr_merge_ind] for curr_merge_ind in curr_merge_list]
            keep_to_merge_list[keep] = merge_list
    return keep_to_merge_list


def nmm(
    object_predictions_as_tensor: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """Apply non-maximum merging to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    # Extract coordinates and scores as tensors
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]
    scores = object_predictions_as_tensor[:, 4]

    # Calculate areas as tensor (vectorized operation)
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes only once
    boxes = []
    for i in range(len(object_predictions_as_tensor)):
        boxes.append(
            box(
                x1[i].item(),  # Convert only individual values
                y1[i].item(),
                x2[i].item(),
                y2[i].item(),
            )
        )

    # Sort indices by score (descending) using torch
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    # Build STRtree
    tree = STRtree(boxes)

    keep_to_merge_list = {}
    merge_to_keep = {}

    for current_idx in sorted_idxs:
        current_box = boxes[current_idx]
        current_area = areas[current_idx].item()  # Convert only when needed

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        matched_box_indices = []
        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx:
                continue

            # Only consider candidates with lower or equal score
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                # Use box coordinates for stable ordering
                current_coords = (
                    x1[current_idx].item(),
                    y1[current_idx].item(),
                    x2[current_idx].item(),
                    y2[current_idx].item(),
                )
                candidate_coords = (
                    x1[candidate_idx].item(),
                    y1[candidate_idx].item(),
                    x2[candidate_idx].item(),
                    y2[candidate_idx].item(),
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError("Invalid match_metric")

            # Add to matched list if overlap exceeds threshold
            if metric >= match_threshold:
                matched_box_indices.append(candidate_idx)

        # Convert current_idx to native Python int
        current_idx_native = int(current_idx)

        # Create keep_ind to merge_ind_list mapping
        if current_idx_native not in merge_to_keep:
            keep_to_merge_list[current_idx_native] = []

            for matched_box_idx in matched_box_indices:
                matched_box_idx_native = int(matched_box_idx)
                if matched_box_idx_native not in merge_to_keep:
                    keep_to_merge_list[current_idx_native].append(matched_box_idx_native)
                    merge_to_keep[matched_box_idx_native] = current_idx_native
        else:
            keep_idx = merge_to_keep[current_idx_native]
            for matched_box_idx in matched_box_indices:
                matched_box_idx_native = int(matched_box_idx)
                if (
                    matched_box_idx_native not in keep_to_merge_list.get(keep_idx, [])
                    and matched_box_idx_native not in merge_to_keep
                ):
                    if keep_idx not in keep_to_merge_list:
                        keep_to_merge_list[keep_idx] = []
                    keep_to_merge_list[keep_idx].append(matched_box_idx_native)
                    merge_to_keep[matched_box_idx_native] = keep_idx

    return keep_to_merge_list


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions."""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ):
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric

        check_requirements(["torch"])

    def __call__(self, predictions: list[ObjectPrediction]):
        raise NotImplementedError()


class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep = nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric
            )
        else:
            keep = batched_nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric
            )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions


class NMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep_to_merge_list = nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(), object_prediction_list[merge_ind].tolist()
                    )
            selected_object_predictions.append(object_prediction_list[keep_ind].tolist())

        return selected_object_predictions


class GreedyNMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep_to_merge_list = greedy_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_greedy_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(), object_prediction_list[merge_ind].tolist()
                    )
            selected_object_predictions.append(object_prediction_list[keep_ind].tolist())

        return selected_object_predictions


class LSNMSPostprocess(PostprocessPredictions):
    # https://github.com/remydubois/lsnms/blob/10b8165893db5bfea4a7cb23e268a502b35883cf/lsnms/nms.py#L62
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        try:
            from lsnms import nms
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please run "pip install lsnms>0.3.1" to install lsnms first for lsnms utilities.'
            )

        if self.match_metric == "IOS":
            NotImplementedError(f"match_metric={self.match_metric} is not supported for LSNMSPostprocess")

        logger.warning("LSNMSPostprocess is experimental and not recommended to use.")

        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_numpy = object_prediction_list.tonumpy()

        boxes = object_predictions_as_numpy[:, :4]
        scores = object_predictions_as_numpy[:, 4]
        class_ids = object_predictions_as_numpy[:, 5].astype("uint8")

        keep = nms(
            boxes, scores, iou_threshold=self.match_threshold, class_ids=None if self.class_agnostic else class_ids
        )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions
