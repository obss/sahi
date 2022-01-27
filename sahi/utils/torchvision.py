import cv2
import numpy as np
import torch
import torchvision.models.detection as models


class TorchVisionTestConstants:
    fasterrcnn_resnet50 = models.fasterrcnn_resnet50_fpn(pretrained=True)
    retinanet_resnet50 = models.retinanet_resnet50_fpn(pretrained=True)
    maskrcnn_resnet50 = models.maskrcnn_resnet50_fpn(pretrained=True)


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
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image).float()
    if image.max() > 1:
        image /= 255
    return image
