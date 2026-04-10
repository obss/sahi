from __future__ import annotations

import numpy as np

from sahi.slicers.base import BaseSlicer
from sahi.utils.import_utils import check_requirements


class NumbaSlicer(BaseSlicer):
    """Numba-accelerated image slicing backend.

    Uses a JIT-compiled copy kernel for pixel extraction, which can
    be faster than NumPy for large numbers of small slices due to
    reduced Python overhead.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        check_requirements(["numba"])

        import numba

        @numba.njit
        def _numba_slice(arr: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
            return arr[y0:y1, x0:x1].copy()

        self._numba_slice = _numba_slice

    def _slice_region(
        self, image_arr: np.ndarray, y_min: int, y_max: int, x_min: int, x_max: int
    ) -> np.ndarray:
        return self._numba_slice(image_arr, y_min, y_max, x_min, x_max)
