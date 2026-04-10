from sahi.slicers.base import BaseSlicer
from sahi.slicers.dali_slicer import DALISlicer
from sahi.slicers.numba_slicer import NumbaSlicer
from sahi.slicers.numpy_slicer import NumpySlicer

SLICER_BACKENDS = {
    "numpy": NumpySlicer,
    "numba": NumbaSlicer,
    "dali": DALISlicer,
}


def get_slicer(
    backend: str = "numpy",
    slice_height: int | None = 512,
    slice_width: int | None = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    auto_slice_resolution: bool = True,
    **kwargs,
) -> BaseSlicer:
    """Instantiate a slicer backend.

    Args:
        backend: Slicing backend to use. One of "numpy", "numba", "dali".
        slice_height: Height of each slice.
        slice_width: Width of each slice.
        overlap_height_ratio: Fractional overlap in height of each window.
        overlap_width_ratio: Fractional overlap in width of each window.
        auto_slice_resolution: Whether to automatically calculate slice resolution.
        **kwargs: Additional backend-specific arguments.

    Returns:
        BaseSlicer: Slicer instance.
    """
    if backend not in SLICER_BACKENDS:
        raise ValueError(f"Unknown slicer backend: {backend}. Choose from {list(SLICER_BACKENDS.keys())}")

    slicer_class = SLICER_BACKENDS[backend]
    return slicer_class(
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        auto_slice_resolution=auto_slice_resolution,
        **kwargs,
    )
