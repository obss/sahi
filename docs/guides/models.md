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

Supports [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26),
[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11),
[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8), and all
Ultralytics model variants including segmentation and oriented bounding box models.

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

Use object detection and zero-shot object detection models from the HuggingFace
Hub (DETR, Deformable DETR, DETA, GroundingDINO, etc.).

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

GroundingDINO models require text-conditioned inference. Use `text_labels` when
the target categories are known, so SAHI can assign stable category ids to those
labels. Additional grounded phrases returned by the processor are appended as
new categories.

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="huggingface",
    model_path="IDEA-Research/grounding-dino-tiny",
    confidence_threshold=0.25,
    text_threshold=0.20,
    text_labels=["car", "truck", "person"],
    device="cuda:0",
)
```

### Zero-shot parameters

In addition to the [common parameters](#common-parameters), zero-shot
(GroundingDINO) models accept:

| Parameter | Type | Description |
| ----------- | ------ | ------------- |
| `text_labels` | `list[str]` | Fixed categories to detect, e.g. `["car", "truck"]`. Each gets a stable category id; phrases outside this list are dropped |
| `text_prompt` | str | Free-form prompt (e.g. `"a car. a truck."`) used when `text_labels` is not set; returned phrases become categories dynamically |
| `text_threshold` | float | Minimum score for matching a box to a text token (default: 0.25) |

HuggingFace object detection notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb)

GroundingDINO zero-shot detection notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_groundingdino.ipynb)

---

## HuggingFace Segmentation

Run segmentation models from the HuggingFace Hub. SAHI returns each segment as
an `ObjectPrediction` with a polygon mask, so sliced inference and
postprocessing work the same as for detection.

| Architecture | `instance` | `semantic` | `panoptic` |
| ------------ | :--------: | :--------: | :--------: |
| MaskFormer   |     ✅     |     ✅     |     ✅     |
| Mask2Former  |     ✅     |     ✅     |     ✅     |
| OneFormer    |     ✅     |     ✅     |     ✅     |

The available heads depend on the checkpoint (e.g.
`facebook/mask2former-swin-tiny-coco-instance` is instance-only). OneFormer
selects the head at inference time, so a single checkpoint serves all three.

```bash
pip install transformers timm
```

```python
from sahi.models.huggingface_segmentation import SegmentationType

detection_model = AutoDetectionModel.from_pretrained(
    model_type="huggingface_segmentation",
    model_path="facebook/mask2former-swin-tiny-coco-instance",
    confidence_threshold=0.5,
    device="cuda:0",
    segmentation_type=SegmentationType.INSTANCE_SEGMENTATION,
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

Switch `segmentation_type` to `SEMANTIC_SEGMENTATION` or
`PANOPTIC_SEGMENTATION` to use the matching head. Note that semantic
segmentation merges every instance of a class into a single mask, so one
`ObjectPrediction` is returned per class rather than per instance.

### Segmentation parameters

In addition to the [common parameters](#common-parameters), this model accepts:

| Parameter | Type | Description |
| ----------- | ------ | ------------- |
| `segmentation_type` | `SegmentationType` | `INSTANCE_SEGMENTATION` (default), `SEMANTIC_SEGMENTATION`, or `PANOPTIC_SEGMENTATION` |
| `min_segment_area` | int | Drop segments smaller than this many pixels (default: 100) |
| `overlap_mask_area_threshold` | float | Merge/discard disconnected parts within a mask (default: 0.8) |
| `label_ids_to_fuse` | `list[int]` | Panoptic only -- fuse all instances of these labels into one segment |
| `token` | str | HuggingFace access token for gated/private models (falls back to `$HF_TOKEN`) |

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
| ----------- | ------ | ------------- |
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
