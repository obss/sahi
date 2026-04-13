---
tags:
  - models
  - inference
  - ultralytics
  - mmdetection
  - huggingface
  - torchvision
  - detectron2
  - yolov5
  - roboflow
---

# Model Integrations

SAHI works with any object detection framework through a unified API. Load your
model once with `AutoDetectionModel.from_pretrained()`, then use it with any SAHI
function -- sliced prediction, batch inference, CLI, etc.

## Ultralytics (YOLO)

Supports YOLOv8, YOLO11, YOLO26, and all Ultralytics model variants
including segmentation and oriented bounding box models.

```bash
pip install ultralytics
```

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cuda:0",  # or "cpu"
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

Ultralytics models also support **native GPU batch inference** for faster
processing of multiple slices:

```python
result = get_sliced_prediction(
    "large_image.jpg",
    detection_model,
    slice_height=640,
    slice_width=640,
    batch_size=8,  # process 8 slices at once
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb)

---

## YOLOE

YOLOE models with prompt-free and open-vocabulary detection.

```bash
pip install ultralytics
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yoloe",
    model_path="yoloe-v8l-seg.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb)

---

## YOLO-World (Zero-Shot)

Open-vocabulary detection -- detect objects by text description without
retraining.

```bash
pip install ultralytics
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolo-world",
    model_path="yolov8s-worldv2.pt",
    confidence_threshold=0.1,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

---

## YOLOv5

Classic YOLOv5 models via the `yolov5` pip package.

```bash
pip install yolov5
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov5",
    model_path="yolov5s.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb)

---

## HuggingFace Transformers

Use any object detection model from the HuggingFace Hub (DETR, Deformable DETR,
DETA, etc.).

```bash
pip install transformers timm
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="huggingface",
    model_path="facebook/detr-resnet-50",
    confidence_threshold=0.3,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb)

---

## RT-DETR

Real-Time Detection Transformer for high-accuracy real-time detection.

```bash
pip install transformers timm
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="rtdetr",
    model_path="PekingU/rtdetr_r50vd",
    confidence_threshold=0.3,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb)

---

## TorchVision

Use built-in TorchVision detection models (Faster R-CNN, RetinaNet, FCOS, SSD,
etc.).

```bash
pip install torch torchvision
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="torchvision",
    model_path="fasterrcnn_resnet50_fpn",
    confidence_threshold=0.3,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb)

---

## MMDetection

Supports the full MMDetection model zoo (300+ models).

```bash
pip install mmdet mmcv mmengine
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="mmdet",
    model_path="path/to/checkpoint.pth",
    config_path="path/to/config.py",
    confidence_threshold=0.25,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb)

---

## Detectron2

Use Facebook's Detectron2 models for detection and instance segmentation.

```bash
pip install detectron2
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="detectron2",
    model_path="path/to/model_final.pth",
    config_path="path/to/config.yaml",
    confidence_threshold=0.25,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb)

---

## Roboflow (RF-DETR)

Use Roboflow's RF-DETR models for detection and segmentation.

```bash
pip install rfdetr
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="roboflow",
    model_path="rfdetr-base",
    confidence_threshold=0.3,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb)

---

## Common Parameters

All models accept these parameters in `AutoDetectionModel.from_pretrained()`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_type` | str | Framework name (see sections above) |
| `model_path` | str | Path to weights file or model name |
| `config_path` | str | Config file path (MMDetection, Detectron2) |
| `confidence_threshold` | float | Minimum score to keep a detection (default: 0.25) |
| `device` | str | `"cpu"`, `"cuda:0"`, `"mps"`, etc. |
| `category_mapping` | dict | Map category IDs to names: `{0: "car", 1: "person"}` |
| `category_remapping` | dict | Remap category names after inference |
| `image_size` | int | Override model input resolution |
| `load_at_init` | bool | Load weights immediately (default: True) |

## Using a Pre-loaded Model

If you already have a model instance, pass it directly instead of a path:

```python
from ultralytics import YOLO

yolo_model = YOLO("yolo26n.pt")
# ... customize the model ...

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model=yolo_model,
    confidence_threshold=0.25,
    device="cuda:0",
)
```

## Next Steps

- [How Sliced Inference Works](sliced-inference.md) -- Understand the algorithm
- [Prediction Utilities](../predict.md) -- Advanced prediction options
- [Interactive Notebooks](../notebooks.md) -- Hands-on examples for each framework
