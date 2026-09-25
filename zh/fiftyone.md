---
tags:
  - fiftyone
  - visualization
  - coco
  - dataset
---

# FiftyOne 可视化

[FiftyOne](https://github.com/voxel51/fiftyone) 提供了交互式 UI，用于探索检测结果、比较预测和调试模型性能。

支持版本：`pip install fiftyone>=0.14.2,<0.15.0`

## 探索 COCO 数据集

```python
from sahi.utils.fiftyone import launch_fiftyone_app

# 使用你的 COCO 数据集启动 FiftyOne 应用
session = launch_fiftyone_app(coco_image_dir, coco_json_path)

# 完成后关闭会话
session.close()
```

## 可视化 SAHI 预测结果

运行切片推理并将结果转换为 FiftyOne 格式：

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# 转换为 FiftyOne 检测格式
fiftyone_detections = result.to_fiftyone_detections()
```

## 比较多个检测结果

使用命令行可视化数据集及多个预测结果，按误检排序：

```bash
sahi coco fiftyone \
  --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json cocoresult2.json
```

设置用于 FP/TP 分类的 IOU 阈值：

```bash
sahi coco fiftyone --iou_threshold 0.5 \
  --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json
```

## 一步完成预测和探索

`predict-fiftyone` CLI 命令可直接运行切片推理并在 FiftyOne 中打开结果：

```bash
sahi predict-fiftyone \
  --image_dir images/ \
  --dataset_json_path dataset.json \
  --model_path yolo26n.pt \
  --model_type ultralytics \
  --slice_height 512 \
  --slice_width 512
```
