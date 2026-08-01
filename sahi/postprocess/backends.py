"""Postprocessing backend selection and auto-detection.

Usage:
    from sahi.postprocess.backends import set_postprocess_backend, get_postprocess_backend

    set_postprocess_backend("numba")   # force numba
    set_postprocess_backend("auto")    # auto-detect best available
"""

from __future__ import annotations

from sahi.utils.import_utils import is_available

VALID_BACKENDS = ("auto", "numpy", "numba", "torchvision")

_backend: str = "auto"
_resolved_cache: str | None = None


def set_postprocess_backend(name: str) -> None:
    """Set the postprocessing backend.

    Call once at startup before running any inference.  This function is
    **not** thread-safe.

    Args:
        name: One of "auto", "numpy", "numba", "torchvision".
    """
    global _backend, _resolved_cache
    if name not in VALID_BACKENDS:
        raise ValueError(f"Unknown backend {name!r}. Choose from {VALID_BACKENDS}")
    _backend = name
    _resolved_cache = None  # force re-resolve
    # Invalidate the dispatch function cache in combine module
    try:
        from sahi.postprocess.combine import _dispatch_cache

        _dispatch_cache.clear()
    except ImportError:
        pass


def get_postprocess_backend() -> str:
    """Return the currently configured backend name (may be "auto")."""
    return _backend


def resolve_backend() -> str:
    """Resolve "auto" to a concrete backend, caching the result.

    When the backend is set to "auto", detection follows this priority:

    1. **torchvision** -- selected if torchvision is installed and a GPU is
       available, either CUDA or Apple MPS (GPU-accelerated NMS).
    2. **numba** -- selected if the numba package is installed (JIT-compiled
       loops, faster than pure numpy for large prediction counts).
    3. **numpy** -- always available as the fallback (pure numpy,
       no extra dependencies).

    If the backend was explicitly set via ``set_postprocess_backend``, that
    value is returned directly without auto-detection.

    Returns:
        One of "numpy", "numba", or "torchvision".
    """
    global _resolved_cache
    if _resolved_cache is not None:
        return _resolved_cache

    if _backend != "auto":
        _resolved_cache = _backend
        return _backend

    # Auto-detect: prefer torchvision on CUDA/MPS, then numba, then numpy
    if is_available("torchvision"):
        try:
            import torch

            # torch.mps.is_available() only exists from torch 2.5; torch.backends.mps
            # has been the stable entry point since 1.12 and is False on non-Apple builds.
            if torch.cuda.is_available() or torch.backends.mps.is_available():
                _resolved_cache = "torchvision"
                return _resolved_cache
        except ImportError:
            pass

    if is_available("numba"):
        _resolved_cache = "numba"
        return _resolved_cache

    _resolved_cache = "numpy"
    return _resolved_cache
