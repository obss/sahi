import os
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

YOLOV8N_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
YOLOV8N_SEG_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt"
YOLO11N_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
YOLO11N_SEG_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt"
YOLO11N_OBB_WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt"


class UltralyticsTestConstants:
    YOLOV8N_MODEL_PATH = "tests/data/models/yolov8n.pt"
    YOLOV8N_SEG_MODEL_PATH = "tests/data/models/yolov8n-seg.pt"
    YOLO11N_MODEL_PATH = "tests/data/models/yolo11n.pt"
    YOLO11N_SEG_MODEL_PATH = "tests/data/models/yolo11n-seg.pt"
    YOLO11N_OBB_MODEL_PATH = "tests/data/models/yolo11n-obb.pt"


def download_file(url: str, save_path: str, chunk_size: int = 8192) -> None:
    """
    Downloads a file from a given URL to the specified path.

    Args:
        url: URL to download the file from
        save_path: Path where the file will be saved
        chunk_size: Size of chunks for downloading
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)


def download_yolov8n_model(destination_path: Optional[str] = None) -> str:
    """Downloads YOLOv8n model if not already downloaded."""
    if destination_path is None:
        destination_path = UltralyticsTestConstants.YOLOV8N_MODEL_PATH

    if not os.path.exists(destination_path):
        download_file(YOLOV8N_WEIGHTS_URL, destination_path)
    return destination_path


def download_yolov8n_seg_model(destination_path: Optional[str] = None) -> str:
    """Downloads YOLOv8n-seg model if not already downloaded."""
    if destination_path is None:
        destination_path = UltralyticsTestConstants.YOLOV8N_SEG_MODEL_PATH

    if not os.path.exists(destination_path):
        download_file(YOLOV8N_SEG_WEIGHTS_URL, destination_path)
    return destination_path


def download_yolo11n_model(destination_path: Optional[str] = None) -> str:
    """Downloads YOLO11n model if not already downloaded."""
    if destination_path is None:
        destination_path = UltralyticsTestConstants.YOLO11N_MODEL_PATH

    if not os.path.exists(destination_path):
        download_file(YOLO11N_WEIGHTS_URL, destination_path)
    return destination_path


def download_yolo11n_seg_model(destination_path: Optional[str] = None) -> str:
    """Downloads YOLO11n-seg model if not already downloaded."""
    if destination_path is None:
        destination_path = UltralyticsTestConstants.YOLO11N_SEG_MODEL_PATH

    if not os.path.exists(destination_path):
        download_file(YOLO11N_SEG_WEIGHTS_URL, destination_path)
    return destination_path


def download_yolo11n_obb_model(destination_path: Optional[str] = None) -> str:
    """Downloads YOLO11n-obb model if not already downloaded."""
    if destination_path is None:
        destination_path = UltralyticsTestConstants.YOLO11N_OBB_MODEL_PATH

    if not os.path.exists(destination_path):
        download_file(YOLO11N_OBB_WEIGHTS_URL, destination_path)
    return destination_path


def download_model_weights(model_path: str) -> str:
    """
    Downloads model weights based on the model path.

    Args:
        model_path: Path or name of the model
    Returns:
        Path to the downloaded weights file
    """
    model_name = Path(model_path).stem
    if model_name == "yolov8n":
        return download_yolov8n_model()
    elif model_name == "yolov8n-seg":
        return download_yolov8n_seg_model()
    elif model_name == "yolo11n":
        return download_yolo11n_model()
    elif model_name == "yolo11n-seg":
        return download_yolo11n_seg_model()
    elif model_name == "yolo11n-obb":
        return download_yolo11n_obb_model()
    else:
        raise ValueError(f"Unknown model: {model_name}")
