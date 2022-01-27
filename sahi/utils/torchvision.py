# OBSS SAHI Tool
# Code written by Kadir Nar, 2020.

import cv2
import numpy as np
import torchvision
import urllib.request
from os import path
from pathlib import Path
from typing import Optional


class TorchVisionTestConstants:
    FASTERCNN_CONFIG_ZOO_NAME = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #RETINANET_CONFIG_ZOO_NAME = detection.retinanet_resnet50_fpn(pretrained=True)
    #MASKRCNN_CONFIG_ZOO_NAME = detection.maskrcnn_resnet50_fpn(pretrained=True)

    FASTERCNN_CONFIG_URL = "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    FASTERCNN_MODEL_PATH = "tests/data/models/torcvhvision/faster_rcnn.pt"


def download_torchvision_model(destination_path: Optional[str] = None):
    if destination_path is None:
        destination_path = TorchVisionTestConstants.FASTERCNN_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            TorchVisionTestConstants.FASTERCNN_CONFIG_URL,
            destination_path,
        )


classes = (
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)


def read_image(img):
    if type(img) == str:
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(img) == bytes:
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    elif type(img) == np.ndarray:
        if len(img.shape) == 2:  # grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        elif len(img.shape) == 3 and img.shape[2] == 3:
            img = img

        elif len(img.shape) == 3 and img.shape[2] == 4:  # RGBA
            img = img[:, :, :3]

    return img


def numpy_to_torch(img):
    import torch

    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).float()
    if img.max() > 1:
        img /= 255
    return img


def torch_to_numpy(img):
    img = img.numpy()
    if img.max() > 1:
        img /= 255
    img = img.transpose((1, 2, 0))
    return img


def resize_image(image, long_size, interpolation=cv2.INTER_LINEAR):
    height, width, channel = image.shape

    # set target image size
    target_size = long_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    image = cv2.resize(image, (target_w, target_h), interpolation=interpolation)

    return image
