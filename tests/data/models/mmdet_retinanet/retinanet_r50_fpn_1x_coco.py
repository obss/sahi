"""MMDetection configuration file for retinanet_r50_fpn_1x_coco.py."""

_base_ = ["retinanet_r50_fpn.py", "coco_detection.py", "schedule_1x.py", "default_runtime.py"]
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
