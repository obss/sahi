# OBSS SAHI Tool
# Code written by Kadir Nar, 2022.

import urllib.request
from os import path
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torchvision
from PIL import Image
from torchvision.io.image import read_image


class TorchVisionTestConstants:
    FASTERCNN_CONFIG_ZOO_NAME = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # RETINANET_CONFIG_ZOO_NAME = detection.retinanet_resnet50_fpn(pretrained=True)
    # MASKRCNN_CONFIG_ZOO_NAME = detection.maskrcnn_resnet50_fpn(pretrained=True)

    FASTERCNN_CONFIG_URL = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    FASTERCNN_MODEL_PATH = "tests/data/models/torcvhvision/faster_rcnn.pth"


def download_torchvision_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = TorchVisionTestConstants.FASTERCNN_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            TorchVisionTestConstants.FASTERCNN_CONFIG_URL,
            destination_path,
        )


COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def numpy_to_pil(img):
    return Image.fromarray(img)


def pil_to_numpy(img):
    import numpy as np

    return np.array(img)


def resize_aspect_ratio(img, long_size):
    height, width, channel = img.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc

    return resized
