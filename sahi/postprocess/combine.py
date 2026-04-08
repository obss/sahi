from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from sahi.logger import logger
from sahi.postprocess.backends import resolve_backend
from sahi.postprocess.utils import ObjectPredictionList, has_match, merge_object_prediction_pair
from sahi.prediction import ObjectPrediction

# ---------------------------------------------------------------------------
# Backend dispatch
# ---------------------------------------------------------------------------

_BACKEND_MODULE_MAP = {
    "numpy": "sahi.postprocess._numpy_backend",
    "numba": "sahi.postprocess._numba_backend",
    "torchvision": "sahi.postprocess._torchvision_backend",
}

_FUNC_NAME_MAP = {
    "nms": {"numpy": "nms_numpy", "numba": "nms_numba", "torchvision": "nms_torchvision"},
    "greedy_nmm": {"numpy": "greedy_nmm_numpy", "numba": "greedy_nmm_numba", "torchvision": "greedy_nmm_torchvision"},
    "nmm": {"numpy": "nmm_numpy", "numba": "nmm_numba", "torchvision": "nmm_torchvision"},
}

_dispatch_cache: dict[tuple[str, str], Callable[..., Any]] = {}


def _dispatch(func_type: str) -> Callable[..., Any]:
    """Resolve and return the backend-specific function for a given operation type.

    Uses a two-level lookup table: first maps the requested backend name
    (e.g. "numpy") to its module path, then maps the operation type
    (e.g. "nms") to the concrete function name within that module.
    Results are cached per (backend, func_type) pair and invalidated
    when the backend changes.

    Args:
        func_type: The operation type to dispatch. One of "nms",
            "greedy_nmm", or "nmm".

    Returns:
        The callable backend function for the requested operation.
    """
    backend = resolve_backend()
    key = (backend, func_type)
    cached = _dispatch_cache.get(key)
    if cached is not None:
        return cached
    module_path = _BACKEND_MODULE_MAP[backend]
    func_name = _FUNC_NAME_MAP[func_type][backend]
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    _dispatch_cache[key] = func
    return func


# ---------------------------------------------------------------------------
# Batched (per-category) wrapper — shared logic for all batched_* functions
# ---------------------------------------------------------------------------


def _batched_apply(
    predictions: np.ndarray,
    func: Callable[..., Any],
    match_metric: str,
    match_threshold: float,
) -> list[int] | dict[int, list[int]]:
    """Apply a postprocessing function per category and remap indices to global space.

    Works for both suppression functions (returning list[int]) and
    merging functions (returning dict[int, list[int]]).

    Args:
        predictions: Array of shape (N, 6) with columns
            [x1, y1, x2, y2, score, category_id].
        func: The postprocessing function to apply per category (e.g. nms,
            greedy_nmm, nmm).
        match_metric: Metric for overlap computation, "IOU" or "IOS".
        match_threshold: Minimum overlap value to consider a match.

    Returns:
        For NMS-style functions: a list of kept global indices sorted by
        score descending.
        For NMM-style functions: a dict mapping each kept global index to
        a list of merged global indices.
    """
    category_ids = predictions[:, 5]
    scores = predictions[:, 4]

    # Collect results from each category
    all_results = {}
    for category_id in np.unique(category_ids):
        curr_indices = np.where(category_ids == category_id)[0]
        result = func(predictions[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()

        if isinstance(result, list):
            # NMS-style: list of kept indices
            for local_idx in result:
                global_idx = curr_indices_list[local_idx]
                all_results[global_idx] = None  # placeholder for ordering
        else:
            # NMM-style: dict of keeper → merged list
            for local_keep, local_merge_list in result.items():
                global_keep = curr_indices_list[local_keep]
                global_merge = [curr_indices_list[m] for m in local_merge_list]
                all_results[global_keep] = global_merge

    # Determine return type from collected results: NMS returns list,
    # NMM returns dict.  We detect by checking if any value is non-None
    # (NMM stores lists, NMS stores None placeholders).
    is_nms = all(v is None for v in all_results.values())
    if is_nms:
        keep = list(all_results.keys())
        keep.sort(key=lambda i: scores[i], reverse=True)
        return keep
    else:
        return all_results


# ---------------------------------------------------------------------------
# Public API — unchanged signatures
# ---------------------------------------------------------------------------


def nms(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> list[int]:
    """Non-maximum suppression for axis-aligned bounding boxes.

    Dispatches to the resolved backend (numpy, numba, or torchvision).

    Args:
        predictions: Array of shape (N, 6) with columns
            [x1, y1, x2, y2, score, category_id].
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum overlap to suppress a lower-scored box.

    Returns:
        List of indices of the kept predictions, sorted by score descending.
    """
    if len(predictions) == 0:
        return []
    return _dispatch("nms")(predictions, match_metric, match_threshold)


def batched_nms(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> list[int]:
    """Apply non-maximum suppression independently per category.

    Args:
        predictions: Array of shape (N, 6) with columns
            [x1, y1, x2, y2, score, category_id].
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum overlap to suppress a lower-scored box.

    Returns:
        List of indices of the kept predictions, sorted by score descending.
    """
    return _batched_apply(predictions, nms, match_metric, match_threshold)


def greedy_nmm(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Greedy non-maximum merging for axis-aligned bounding boxes.

    Instead of discarding overlapping boxes, merges them into the
    highest-scored box. Dispatches to the resolved backend.

    Args:
        predictions: Array of shape (N, 6) with columns
            [x1, y1, x2, y2, score, category_id].
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum overlap to merge a lower-scored box.

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    return _dispatch("greedy_nmm")(predictions, match_metric, match_threshold)


def batched_greedy_nmm(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Apply greedy non-maximum merging independently per category.

    Args:
        predictions: Array of shape (N, 6) with columns
            [x1, y1, x2, y2, score, category_id].
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum overlap to merge a lower-scored box.

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    return _batched_apply(predictions, greedy_nmm, match_metric, match_threshold)


def nmm(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Non-maximum merging (non-greedy, transitive) for axis-aligned bounding boxes.

    Unlike greedy NMM, this variant allows transitive merging: if box A
    merges with B and B merges with C, all three are merged together.
    Dispatches to the resolved backend.

    Args:
        predictions: Array of shape (N, 6) with columns
            [x1, y1, x2, y2, score, category_id].
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum overlap to merge a lower-scored box.

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    return _dispatch("nmm")(predictions, match_metric, match_threshold)


def batched_nmm(
    predictions: np.ndarray,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
) -> dict[int, list[int]]:
    """Apply non-maximum merging (non-greedy, transitive) independently per category.

    Args:
        predictions: Array of shape (N, 6) with columns
            [x1, y1, x2, y2, score, category_id].
        match_metric: Overlap metric, "IOU" or "IOS".
        match_threshold: Minimum overlap to merge a lower-scored box.

    Returns:
        Dict mapping each kept index to a list of indices merged into it.
    """
    return _batched_apply(predictions, nmm, match_metric, match_threshold)


# ---------------------------------------------------------------------------
# Postprocess classes
# ---------------------------------------------------------------------------


class PostprocessPredictions(ABC):
    """Abstract base class for postprocessing object prediction lists.

    Subclasses implement a specific strategy (NMS, NMM, greedy NMM, etc.)
    to reduce overlapping detections produced by sliced inference.

    Args:
        match_threshold: Minimum overlap value (IoU or IoS) to consider
            two predictions as matching.
        match_metric: Overlap metric, "IOU" or "IOS".
        class_agnostic: If True, apply postprocessing across all
            categories. If False, apply per category independently.
    """

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ) -> None:
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric

    @abstractmethod
    def __call__(self, predictions: list[ObjectPrediction]) -> list[ObjectPrediction]:
        pass


def _apply_merge(
    object_prediction_list: ObjectPredictionList,
    keep_to_merge_list: dict[int, list[int]],
    match_metric: str,
    match_threshold: float,
) -> list[ObjectPrediction]:
    """Apply merge operations using the keep-to-merge mapping.

    Shared merge logic for NMM and GreedyNMM postprocess classes.
    For each kept prediction, iteratively merges all matched predictions
    (bounding boxes, masks, scores, and categories) into it.

    Args:
        object_prediction_list: The full list of object predictions.
        keep_to_merge_list: Dict mapping each kept index to a list of
            indices that should be merged into it.
        match_metric: Overlap metric used for the merge check, "IOU"
            or "IOS".
        match_threshold: Minimum overlap to confirm and apply a merge.

    Returns:
        List of merged ObjectPrediction instances.
    """
    selected = []
    for keep_ind, merge_ind_list in keep_to_merge_list.items():
        for merge_ind in merge_ind_list:
            if has_match(
                object_prediction_list[keep_ind].tolist(),
                object_prediction_list[merge_ind].tolist(),
                match_metric,
                match_threshold,
            ):
                object_prediction_list[keep_ind] = merge_object_prediction_pair(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                )
        selected.append(object_prediction_list[keep_ind].tolist())
    return selected


class NMSPostprocess(PostprocessPredictions):
    """Postprocessor using Non-Maximum Suppression (NMS).

    Keeps the highest-scored prediction among overlapping boxes and
    discards the rest. Does not merge bounding boxes or masks.
    """

    def __call__(self, object_predictions: list[ObjectPrediction]) -> list[ObjectPrediction]:
        object_prediction_list = ObjectPredictionList(object_predictions)
        preds_np = object_prediction_list.tonumpy()
        func = nms if self.class_agnostic else batched_nms
        keep = func(preds_np, match_threshold=self.match_threshold, match_metric=self.match_metric)

        selected = object_prediction_list[keep].tolist()
        if not isinstance(selected, list):
            selected = [selected]
        return selected


class NMMPostprocess(PostprocessPredictions):
    """Postprocessor using Non-Maximum Merging (NMM) with transitive merging.

    Instead of discarding overlapping detections, merges their bounding
    boxes, masks, and scores. Uses non-greedy transitive merging: if A
    overlaps B and B overlaps C, all three are merged even if A does not
    directly overlap C.
    """

    _agnostic_func = staticmethod(nmm)
    _batched_func = staticmethod(batched_nmm)

    def __call__(self, object_predictions: list[ObjectPrediction]) -> list[ObjectPrediction]:
        object_prediction_list = ObjectPredictionList(object_predictions)
        preds_np = object_prediction_list.tonumpy()
        func = self._agnostic_func if self.class_agnostic else self._batched_func
        keep_to_merge = func(preds_np, match_threshold=self.match_threshold, match_metric=self.match_metric)
        return _apply_merge(object_prediction_list, keep_to_merge, self.match_metric, self.match_threshold)


class GreedyNMMPostprocess(NMMPostprocess):
    """Postprocessor using Greedy Non-Maximum Merging (NMM).

    Similar to NMM but uses a greedy strategy: each kept prediction only
    merges boxes that directly overlap with it (no transitive merging).
    This is faster than full NMM and produces tighter merged boxes.
    """

    _agnostic_func = staticmethod(greedy_nmm)
    _batched_func = staticmethod(batched_greedy_nmm)


class LSNMSPostprocess(PostprocessPredictions):
    """Postprocessor using Locality-Sensitive NMS from the ``lsnms`` package.

    Uses a spatial index for fast neighbor lookup, making it efficient for
    large numbers of predictions. Only supports IoU metric (not IoS).
    Requires the ``lsnms`` package (``pip install lsnms>0.3.1``).

    Note:
        This postprocessor is experimental and not recommended for
        production use.
    """

    def __call__(self, object_predictions: list[ObjectPrediction]) -> list[ObjectPrediction]:
        try:
            from lsnms import nms
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please run "pip install lsnms>0.3.1" to install lsnms first for lsnms utilities.'
            )

        if self.match_metric == "IOS":
            raise NotImplementedError(f"match_metric={self.match_metric} is not supported for LSNMSPostprocess")

        logger.warning("LSNMSPostprocess is experimental and not recommended to use.")

        object_prediction_list = ObjectPredictionList(object_predictions)
        preds_np = object_prediction_list.tonumpy()

        boxes = preds_np[:, :4]
        scores = preds_np[:, 4]
        class_ids = preds_np[:, 5].astype("uint8")

        keep = nms(
            boxes, scores, iou_threshold=self.match_threshold, class_ids=None if self.class_agnostic else class_ids
        )

        selected = object_prediction_list[keep].tolist()
        if not isinstance(selected, list):
            selected = [selected]
        return selected
