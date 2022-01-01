# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2021.

import logging
from typing import List

import torch

from sahi.postprocess.utils import ObjectPredictionList, has_match, merge_object_prediction_pair
from sahi.prediction import ObjectPrediction

logger = logging.getLogger(__name__)


def batched_nms(predictions: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5):
    scores = predictions[:, 4].squeeze()
    category_ids = predictions[:, 5].squeeze()
    keep_mask = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_indices = nms(predictions[curr_indices], match_metric, match_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def nms(
    predictions: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """
    # we extract coordinates for every
    # prediction box present in P
    x1 = predictions[:, 0]
    y1 = predictions[:, 1]
    x2 = predictions[:, 2]
    y2 = predictions[:, 3]

    # we extract the confidence scores as well
    scores = predictions[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoU of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU less than thresh_iou
        mask = match_metric_value < match_threshold
        order = order[mask]
    return keep


def batched_nmm_torch0(
    object_predictions_as_tensor: torch.tensor,
    object_prediction_list: ObjectPredictionList,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_mask = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_object_predictions = object_prediction_list[curr_indices]
        curr_keep_indices, curr_object_predictions = nmm_torch0(
            object_predictions_as_tensor[curr_indices], curr_object_predictions, match_metric, match_threshold
        )
        keep_mask[curr_indices[curr_keep_indices]] = True
        object_prediction_list[curr_indices] = curr_object_predictions
    keep_indices = torch.where(keep_mask)[0]
    scores = object_prediction_list.totensor()[keep_indices, 4].squeeze()
    return keep_indices[scores.sort(descending=True)[1]], object_prediction_list


def nmm_torch0(
    object_predictions_as_tensor: torch.tensor,
    object_prediction_list: ObjectPredictionList,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        object_predictions_as_list: ObjectPredictionList Object prediction objects
            to be merged.
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    # we extract coordinates for every
    # prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]

    # we extract the confidence scores as well
    scores = object_predictions_as_tensor[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU/IoS less than thresh_iou
        mask = match_metric_value < match_threshold
        matched_box_indices = order[(mask == False).nonzero().flatten()].flip(dims=(0,))
        unmatched_indices = order[(mask == True).nonzero().flatten()]

        # merge matching predictions
        for matched_box_ind in matched_box_indices:
            object_prediction_list[idx] = merge_object_prediction_pair(
                object_prediction_list[idx].tolist(), object_prediction_list[matched_box_ind].tolist()
            )

        # update box pool
        order = unmatched_indices[scores[unmatched_indices].argsort()]
    return keep, object_prediction_list


def batched_greedy_nmm(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply greedy version of non-maximum merging per category to avoid detecting
    too many overlapping bounding boxes for a given object.
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
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply greedy version of non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        object_predictions_as_list: ObjectPredictionList Object prediction objects
            to be merged.
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    keep_to_merge_list = {}

    # we extract coordinates for every
    # prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]

    # we extract the confidence scores as well
    scores = object_predictions_as_tensor[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()

    # initialise an empty list for
    # filtered prediction boxes
    keep = []

    while len(order) > 0:
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]

        # push S in filtered predictions list
        keep.append(idx.tolist())

        # remove S from P
        order = order[:-1]

        # sanity check
        if len(order) == 0:
            break

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=order)
        xx2 = torch.index_select(x2, dim=0, index=order)
        yy1 = torch.index_select(y1, dim=0, index=order)
        yy2 = torch.index_select(y2, dim=0, index=order)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=order)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[idx]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[idx])
            # find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU/IoS less than thresh_iou
        mask = match_metric_value < match_threshold
        matched_box_indices = order[(mask == False).nonzero().flatten()].flip(dims=(0,))
        unmatched_indices = order[(mask == True).nonzero().flatten()]

        # update box pool
        order = unmatched_indices[scores[unmatched_indices].argsort()]

        # create keep_ind to merge_ind_list mapping
        keep_to_merge_list[idx.tolist()] = []

        for matched_box_ind in matched_box_indices.tolist():
            keep_to_merge_list[idx.tolist()].append(matched_box_ind)

    return keep_to_merge_list


def batched_nmm(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum merging per category to avoid detecting too many
    overlapping bounding boxes for a given object.
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
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Apply non-maximum merging to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        object_predictions_as_list: ObjectPredictionList Object prediction objects
            to be merged.
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    keep_to_merge_list = {}
    merge_to_keep = {}

    # we extract coordinates for every
    # prediction box present in P
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]

    # we extract the confidence scores as well
    scores = object_predictions_as_tensor[:, 4]

    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)

    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort(descending=True)

    for ind in range(len(object_predictions_as_tensor)):
        # extract the index of the
        # prediction with highest score
        # we call this prediction S
        pred_ind = order[ind]
        pred_ind = pred_ind.tolist()

        # remove selected pred
        other_pred_inds = order[order != pred_ind]

        # select coordinates of BBoxes according to
        # the indices in order
        xx1 = torch.index_select(x1, dim=0, index=other_pred_inds)
        xx2 = torch.index_select(x2, dim=0, index=other_pred_inds)
        yy1 = torch.index_select(y1, dim=0, index=other_pred_inds)
        yy2 = torch.index_select(y2, dim=0, index=other_pred_inds)

        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[pred_ind])
        yy1 = torch.max(yy1, y1[pred_ind])
        xx2 = torch.min(xx2, x2[pred_ind])
        yy2 = torch.min(yy2, y2[pred_ind])

        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1

        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)

        # find the intersection area
        inter = w * h

        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim=0, index=other_pred_inds)

        if match_metric == "IOU":
            # find the union of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            union = (rem_areas - inter) + areas[pred_ind]
            # find the IoU of every prediction in P with S
            match_metric_value = inter / union

        elif match_metric == "IOS":
            # find the smaller area of every prediction T in P
            # with the prediction S
            # Note that areas[idx] represents area of S
            smaller = torch.min(rem_areas, areas[pred_ind])
            # find the IoS of every prediction in P with S
            match_metric_value = inter / smaller
        else:
            raise ValueError()

        # keep the boxes with IoU/IoS less than thresh_iou
        mask = match_metric_value < match_threshold
        matched_box_indices = other_pred_inds[(mask == False).nonzero().flatten()].flip(dims=(0,))
        unmatched_indices = other_pred_inds[(mask == True).nonzero().flatten()]

        # create keep_ind to merge_ind_list mapping
        if pred_ind not in merge_to_keep:
            keep_to_merge_list[pred_ind] = []

            for matched_box_ind in matched_box_indices.tolist():
                if matched_box_ind not in merge_to_keep:
                    keep_to_merge_list[pred_ind].append(matched_box_ind)
                    merge_to_keep[matched_box_ind] = pred_ind

        else:
            keep = merge_to_keep[pred_ind]
            for matched_box_ind in matched_box_indices.tolist():
                if matched_box_ind not in keep_to_merge_list and matched_box_ind not in merge_to_keep:
                    keep_to_merge_list[keep].append(matched_box_ind)
                    merge_to_keep[matched_box_ind] = keep

    return keep_to_merge_list


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions"""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ):
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric

    def __call__(self):
        NotImplementedError()


class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
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

        selected_object_predictions = object_prediction_list[keep.tolist()].tolist()

        return selected_object_predictions


class NMMPostprocess0(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep, object_prediction_list = nmm_torch0(
                object_predictions_as_torch,
                object_prediction_list,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep, object_prediction_list = batched_nmm_torch0(
                object_predictions_as_torch,
                object_prediction_list,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = object_prediction_list[keep.tolist()].tolist()

        return selected_object_predictions


class NMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: List[ObjectPrediction],
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
        object_predictions: List[ObjectPrediction],
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
        object_predictions: List[ObjectPrediction],
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

        selected_object_predictions = object_prediction_list[keep]

        return selected_object_predictions
