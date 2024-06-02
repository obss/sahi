import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class Yolov8TestConstants:
    YOLOV8N_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    YOLOV8N_MODEL_PATH = "tests/data/models/yolov8/yolov8n.pt"

    YOLOV8S_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    YOLOV8S_MODEL_PATH = "tests/data/models/yolov8/yolov8s.pt"

    YOLOV8M_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
    YOLOV8M_MODEL_PATH = "tests/data/models/yolov8/yolov8m.pt"

    YOLOV8L_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
    YOLOV8L_MODEL_PATH = "tests/data/models/yolov8/yolov8l.pt"

    YOLOV8X_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt"
    YOLOV8X_MODEL_PATH = "tests/data/models/yolov8/yolov8x.pt"

    YOLOV8N_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-seg.pt"
    YOLOV8N_SEG_MODEL_PATH = "tests/data/models/yolov8/yolov8n-seg.pt"

    YOLOV8S_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-seg.pt"
    YOLOV8S_SEG_MODEL_PATH = "tests/data/models/yolov8/yolov8s-seg.pt"

    YOLOV8M_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt"
    YOLOV8M_SEG_MODEL_PATH = "tests/data/models/yolov8/yolov8m-seg.pt"

    YOLOV8L_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-seg.pt"
    YOLOV8L_SEG_MODEL_PATH = "tests/data/models/yolov8/yolov8l-seg.pt"

    YOLOV8X_SEG_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-seg.pt"
    YOLOV8X_SEG_MODEL_PATH = "tests/data/models/yolov8/yolov8x-seg.pt"


def download_yolov8n_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8N_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8N_MODEL_URL,
            destination_path,
        )


def download_yolov8s_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8S_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8S_MODEL_URL,
            destination_path,
        )


def download_yolov8m_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8M_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8M_MODEL_URL,
            destination_path,
        )


def download_yolov8l_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8L_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8L_MODEL_URL,
            destination_path,
        )


def download_yolov8x_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8X_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8X_MODEL_URL,
            destination_path,
        )


def download_yolov8n_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8N_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8N_SEG_MODEL_URL,
            destination_path,
        )


def download_yolov8s_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8S_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8S_SEG_MODEL_URL,
            destination_path,
        )


def download_yolov8m_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8M_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8M_SEG_MODEL_URL,
            destination_path,
        )


def download_yolov8l_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8L_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8L_SEG_MODEL_URL,
            destination_path,
        )


def download_yolov8x_seg_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = Yolov8TestConstants.YOLOV8X_SEG_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            Yolov8TestConstants.YOLOV8X_SEG_MODEL_URL,
            destination_path,
        )
