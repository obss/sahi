import importlib.metadata as importlib_metadata

try:
    # This will read version from pyproject.toml
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "development"


# Lazy imports — heavy modules (cv2, shapely, requests, tqdm) are only
# loaded when the user actually accesses one of these names.
_LAZY_IMPORTS = {
    "BoundingBox": "sahi.annotation",
    "Category": "sahi.annotation",
    "Mask": "sahi.annotation",
    "AutoDetectionModel": "sahi.auto_model",
    "DetectionModel": "sahi.models.base",
    "ObjectPrediction": "sahi.prediction",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib

        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        # Cache on the module so __getattr__ is not called again
        globals()[name] = value
        return value
    raise AttributeError(f"module 'sahi' has no attribute {name!r}")


__all__ = [
    "AutoDetectionModel",
    "BoundingBox",
    "Category",
    "DetectionModel",
    "Mask",
    "ObjectPrediction",
    "__version__",
]
