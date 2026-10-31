"""Ultralytics/YOLO model constants and utilities for testing."""

from __future__ import annotations

import os
from pathlib import Path

from sahi.utils.file import download_from_url

ASSETS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
YOLO26N_WEIGHTS_URL = f"{ASSETS_URL}/yolo26n.pt"
YOLO26N_SEG_WEIGHTS_URL = f"{ASSETS_URL}/yolo26n-seg.pt"
YOLO26N_OBB_WEIGHTS_URL = f"{ASSETS_URL}/yolo26n-obb.pt"


class UltralyticsConstants:
    """Constants for Ultralytics/YOLO model paths."""

    YOLO26N_MODEL_PATH = "tests/data/models/yolo26n.pt"
    YOLO26N_SEG_MODEL_PATH = "tests/data/models/yolo26n-seg.pt"
    YOLO26N_OBB_MODEL_PATH = "tests/data/models/yolo26n-obb.pt"
    YOLO26N_ONNX_MODEL_PATH = "tests/data/models/yolo26/yolo26n.onnx"


def download_yolo26n_model(destination_path: str | None = None) -> str:
    """Downloads YOLO26n model if not already downloaded."""
    destination_path = destination_path or UltralyticsConstants.YOLO26N_MODEL_PATH
    download_from_url(YOLO26N_WEIGHTS_URL, destination_path)
    return destination_path


def download_yolo26n_seg_model(destination_path: str | None = None) -> str:
    """Downloads YOLO26n-seg model if not already downloaded."""
    destination_path = destination_path or UltralyticsConstants.YOLO26N_SEG_MODEL_PATH
    download_from_url(YOLO26N_SEG_WEIGHTS_URL, destination_path)
    return destination_path


def download_yolo26n_obb_model(destination_path: str | None = None) -> str:
    """Downloads YOLO26n-obb model if not already downloaded."""
    destination_path = destination_path or UltralyticsConstants.YOLO26N_OBB_MODEL_PATH
    download_from_url(YOLO26N_OBB_WEIGHTS_URL, destination_path)
    return destination_path


def download_yolo26n_onnx_model(
    destination_path: str | Path = UltralyticsConstants.YOLO26N_ONNX_MODEL_PATH,
    image_size: int | None = 640,
) -> None:
    """Download YOLO26n weights and export them to ONNX if needed.

    Args:
        destination_path: Path where the ONNX model will be saved.
        image_size: Image size for model export.
    """
    destination_path = Path(destination_path)
    if destination_path.exists():
        return
    model_path = destination_path.parent / (destination_path.stem + ".pt")
    download_yolo26n_model(str(model_path))

    from ultralytics import YOLO

    model = YOLO(model_path)
    model.export(format="onnx", imgsz=image_size)

    # Cleanup temporary files
    for ext in [".pt", ".yaml"]:
        temp_file = destination_path.parent / (destination_path.stem + ext)
        if temp_file.exists():
            os.remove(temp_file)
