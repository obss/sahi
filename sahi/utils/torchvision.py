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


def read_image(image, img_size=416):
    if type(image) == str:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif type(image) == bytes:
        nparr = np.frombuffer(image, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    elif type(image) == np.ndarray:
        if len(image.shape) == 2:  # grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = image

        elif len(image.shape) == 3 and image.shape[2] == 4:  # RGBAscale
            image = image[:, :, :3]

    image = cv2.resize(image, (img_size, img_size))
    image = numpy_to_torch(image)
    return image


def numpy_to_torch(image):
    import torch
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float()
    if image.max() > 1:
        image /= 255
    return image
