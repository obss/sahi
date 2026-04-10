from __future__ import annotations

from sahi.slicers.base import BaseSlicer


class NumpySlicer(BaseSlicer):
    """NumPy-based image slicing backend.

    Uses the default ``BaseSlicer`` implementation which slices via
    standard NumPy array indexing.  No additional dependencies required.
    """
