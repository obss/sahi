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

# 模型集成

SAHI 通过统一的 API 支持任意目标检测框架。使用
`AutoDetectionModel.from_pretrained()` 加载一次模型后，即可将其用于 SAHI
的任意功能，包括切片预测、批量推理和 CLI 等。

## Ultralytics (YOLO)

支持 YOLOv8、YOLO11、YOLO26 和所有 Ultralytics 模型变体，包括分割模型和
旋转边界框模型。

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
    device="cuda:0",  # 或 "cpu"
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

Ultralytics 模型还支持**原生 GPU 批量推理**，可以更快地处理多个切片：

```python
result = get_sliced_prediction(
    "large_image.jpg",
    detection_model,
    slice_height=640,
    slice_width=640,
    batch_size=8,  # 一次处理 8 个切片
)
```

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb)

---

## YOLOE

支持免提示词检测和开放词汇检测的 YOLOE 模型。

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

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb)

---

## YOLO-World（零样本）

开放词汇检测：无需重新训练，即可通过文本描述检测目标。

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

通过 `yolov5` pip 包使用经典 YOLOv5 模型。

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

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb)

---

## HuggingFace Transformers

使用 HuggingFace Hub 中的目标检测和零样本目标检测模型，包括 DETR、
Deformable DETR、DETA 和 GroundingDINO 等。

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

GroundingDINO 模型需要以文本为条件进行推理。如果目标类别已知，请使用
`text_labels`，这样 SAHI 可以为这些标签分配稳定的类别 ID。处理器返回的其他短语
会被过滤。

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

### 零样本参数

除[通用参数](#common-parameters)外，零样本模型（GroundingDINO）还接受以下参数：

| 参数 | 类型 | 说明 |
| ------ | ------ | ------ |
| `text_labels` | `list[str]` | 要检测的固定类别，例如 `["car", "truck"]`。每个类别都会获得稳定的类别 ID；列表外的短语会被过滤 |
| `text_prompt` | str | 未设置 `text_labels` 时使用的自由文本提示词，例如 `"a car. a truck."`；返回的短语会动态成为类别 |
| `text_threshold` | float | 文本 token 与边界框匹配时的最低分数，默认值为 0.25 |

HuggingFace 目标检测 notebook：
[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb)

GroundingDINO 零样本检测 notebook：
[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_groundingdino.ipynb)

---

## HuggingFace 分割

使用 HuggingFace Hub 中的分割模型。SAHI 会将每个分割结果作为带有多边形掩码的
`ObjectPrediction` 返回，因此切片推理和后处理方式与目标检测相同。

| 架构        | `instance` | `semantic` | `panoptic` |
| ----------- | :--------: | :--------: | :--------: |
| MaskFormer  |     ✅     |     ✅     |     ✅     |
| Mask2Former |     ✅     |     ✅     |     ✅     |
| OneFormer   |     ✅     |     ✅     |     ✅     |

可用的任务头取决于 checkpoint。例如，
`facebook/mask2former-swin-tiny-coco-instance` 仅支持实例分割。OneFormer
会在推理时选择任务头，因此同一个 checkpoint 可以支持全部三种任务。

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

将 `segmentation_type` 切换为 `SEMANTIC_SEGMENTATION` 或
`PANOPTIC_SEGMENTATION` 即可使用对应的任务头。请注意，语义分割会将同一类别的所有
实例合并为一个掩码，因此每个类别只会返回一个 `ObjectPrediction`，而不是为每个
实例分别返回。

### 分割参数

除[通用参数](#common-parameters)外，该模型还接受以下参数：

| 参数 | 类型 | 说明 |
| ------ | ------ | ------ |
| `segmentation_type` | `SegmentationType` | `INSTANCE_SEGMENTATION`（默认）、`SEMANTIC_SEGMENTATION` 或 `PANOPTIC_SEGMENTATION` |
| `min_segment_area` | int | 丢弃像素数小于该值的分割结果，默认值为 100 |
| `overlap_mask_area_threshold` | float | 合并或丢弃掩码中不连续区域的阈值，默认值为 0.8 |
| `label_ids_to_fuse` | `list[int]` | 仅用于全景分割：将这些标签的所有实例合并为一个分割结果 |
| `token` | str | 用于受限或私有模型的 HuggingFace 访问令牌；未提供时回退到 `$HF_TOKEN` |

---

## RT-DETR

用于高精度实时检测的 Real-Time Detection Transformer。

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

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb)

---

## TorchVision

使用 TorchVision 内置的目标检测模型，包括 Faster R-CNN、RetinaNet、FCOS 和 SSD
等。

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

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb)

---

## MMDetection

支持完整的 MMDetection 模型库（300 多个模型）。

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

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb)

---

## Detectron2

使用 Facebook 的 Detectron2 模型进行目标检测和实例分割。

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

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb)

---

## Roboflow (RF-DETR)

使用 Roboflow 的 RF-DETR 模型进行目标检测和分割。

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

[![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb)

---

## 通用参数 { #common-parameters }

所有模型都接受以下 `AutoDetectionModel.from_pretrained()` 参数：

| 参数 | 类型 | 说明 |
| ------ | ------ | ------ |
| `model_type` | str | 框架名称，请参见上文各节 |
| `model_path` | str | 权重文件路径或模型名称 |
| `config_path` | str | 配置文件路径，用于 MMDetection 和 Detectron2 |
| `confidence_threshold` | float | 保留检测结果的最低分数，默认值为 0.25 |
| `device` | str | `"cpu"`、`"cuda:0"` 或 `"mps"` 等 |
| `category_mapping` | dict | 将类别 ID 映射到名称，例如 `{0: "car", 1: "person"}` |
| `category_remapping` | dict | 推理后重新映射类别名称 |
| `image_size` | int | 覆盖模型输入分辨率 |
| `load_at_init` | bool | 立即加载权重，默认值为 True |

## 使用预加载模型

如果已经有模型实例，可以直接传入实例，而不是提供路径：

```python
from ultralytics import YOLO

yolo_model = YOLO("yolo26n.pt")
# ... 自定义模型 ...

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model=yolo_model,
    confidence_threshold=0.25,
    device="cuda:0",
)
```

## 下一步

- [切片推理工作原理](sliced-inference.md) -- 了解算法
- [预测工具](../predict.md) -- 高级预测选项
- [交互式 Notebooks](../notebooks.md) -- 每个框架的动手实践示例
