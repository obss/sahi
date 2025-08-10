import torchvision

MODEL_NAME_TO_CONSTRUCTOR = {
    "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "fasterrcnn_mobilenet_v3_large_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn,
    "fasterrcnn_mobilenet_v3_large_320_fpn": torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
    "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
    "ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    "fcos_resnet50_fpn": torchvision.models.detection.fcos_resnet50_fpn,
}
