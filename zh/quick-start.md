---
hide:
  - navigation
tags:
  - getting-started
  - installation
  - inference
  - slicing
---

# 快速开始

SAHI 将大图像切成相互重叠的切片，在每个切片上运行检测器，再把检测结果合并回完整图像。小目标因此保持足够大，可以被检测到，而且无需重新训练。

## 1. 安装

```bash
pip install "sahi[ultralytics]"
```

单独安装的 `sahi` 不包含任何检测器，因此请为所需框架选择对应的 extra：`ultralytics`、`transformers`、`yolov5`、`roboflow`、`torchvision`、`torch`、`onnx`、`numba` 或 `all`。Conda 用户可以运行 `conda install -c conda-forge sahi`，然后单独安装框架。

## 2. 获取预测结果

以下代码可在 CPU 上完整运行。它会下载一张示例图像，Ultralytics 会在首次使用时下载 `yolo26n.pt`。

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import download_from_url

download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cpu",  # 或 "cuda:0"
)

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

替换 `model_type` 和 `model_path` 即可使用其他框架。完整列表请参见[模型集成](guides/models.md)。

## 3. 读取结果

`result` 是一个 `PredictionResult`。所有检测结果都在 `result.object_prediction_list` 中。

```python
for pred in result.object_prediction_list:
    print(pred.category.name, pred.score.value, pred.bbox.to_xyxy())

# 写入 demo_data/prediction_visual.png
result.export_visuals(export_dir="demo_data/")

# COCO 格式的字典，可直接保存为 JSON
coco_predictions = result.to_coco_predictions(image_id=1)
```

如果图像尺寸已经接近模型输入尺寸，请改用 `get_prediction`，完全跳过切片。

## 4. 使用 CLI 执行同样的操作

```bash
sahi predict --model_type ultralytics --model_path yolo26n.pt --source demo_data/ --slice_height 512 --slice_width 512
```

可视化结果写入 `runs/predict/exp`。添加 `--dataset_json_path dataset.json` 还会导出用于评估的 COCO `result.json`。

## 下一步

- [切片推理工作原理](guides/sliced-inference.md)：如何选择切片大小、重叠率和合并策略。
- [模型集成](guides/models.md)：HuggingFace、MMDetection、Detectron2、TorchVision、RT-DETR、RF-DETR 等。
- [预测工具](predict.md)：批量推理、进度条和导出选项。
- [CLI 命令](cli.md)：所有命令和参数。
- [交互式 Notebooks](notebooks.md)：可运行的 Colab 示例。
