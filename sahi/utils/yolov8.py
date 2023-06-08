import urllib.request
from os import path
from pathlib import Path
from typing import Optional
from ultralytics import YOLO


class Yolov8TestConstants:
    YOLOV8N_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    YOLOV8N_MODEL_PATH = "tests/data/models/yolov8/yolov8n.pt"

    YOLOV8S_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
    YOLOV8S_MODEL_PATH = "tests/data/models/yolov8/yolov8s.pt"

    YOLOV8M_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt"
    YOLOV8M_MODEL_PATH = "tests/data/models/yolov8/yolov8m.pt"

    YOLOV8M_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
    YOLOV8M_MODEL_PATH = "tests/data/models/yolov8/yolov8l.pt"


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


def OpenVino_yolov8n_model(yolov8n_model_path: Optional[str] = None):
    if yolov8n_model_path is None:
        yolov8n_model_path = Yolov8TestConstants.YOLOV8S_MODEL_PATH
    
    download_yolov8n_model(yolov8n_model_path)

    destination_path = str(Path(yolov8n_model_path).parent) + "/" + "yolov8n_openvino_model" + "/" + "yolov8n.xml"
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
 
    if not path.exists(destination_path):
        try:
            det_model = YOLO(yolov8n_model_path)
            det_model.export(format="openvino", dynamic=True, half=False)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov8 model path: ", e)


def OpenVino_yolov8s_model(yolov8s_model_path: Optional[str] = None):
    if yolov8s_model_path is None:
        yolov8s_model_path = Yolov8TestConstants.YOLOV8S_MODEL_PATH
    download_yolov8s_model(yolov8s_model_path)

    destination_path = str(Path(yolov8s_model_path).parent) + "/" + "yolov8s_openvino_model" + "/" + "yolov8s.xml"
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        try:
            det_model = YOLO(yolov8s_model_path)
            det_model.export(format="openvino", dynamic=True, half=False)
        except Exception as e:
            raise TypeError("model_path is not a valid yolov8 model path: ", e)
    

