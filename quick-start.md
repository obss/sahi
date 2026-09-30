---
hide:
  - navigation
tags:
  - getting-started
  - installation
  - inference
  - slicing
---

# Quick Start

SAHI cuts a large image into overlapping slices, runs your detector on each slice, and merges the detections back onto the full image. Small objects stay big enough to detect, and no retraining is needed.

## 1. Install

```bash
pip install "sahi[ultralytics]"
```

`sahi` on its own ships no detector, so pick the extra for the framework you want: `ultralytics`, `transformers`, `yolov5`, `roboflow`, `torchvision`, `torch`, `onnx`, `numba`, or `all`. Conda users can run `conda install -c conda-forge sahi` and install the framework separately.

## 2. Get a prediction

This runs end to end on CPU. It downloads a sample image, and Ultralytics downloads `yolo26n.pt` on first use.

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
    device="cpu",  # or "cuda:0"
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

Swap `model_type` and `model_path` for any other framework. See [Model Integrations](guides/models.md) for the full list.

## 3. Read the result

`result` is a `PredictionResult`. Every detection lives in `result.object_prediction_list`.

```python
for pred in result.object_prediction_list:
    print(pred.category.name, pred.score.value, pred.bbox.to_xyxy())

# Writes demo_data/prediction_visual.png
result.export_visuals(export_dir="demo_data/")

# COCO-format dicts, ready to dump as JSON
coco_predictions = result.to_coco_predictions(image_id=1)
```

If your image is already close to the model input size, use `get_prediction` instead and skip slicing entirely.

## 4. The same run from the CLI

```bash
sahi predict --model_type ultralytics --model_path yolo26n.pt --source demo_data/ --slice_height 512 --slice_width 512
```

Visuals are written to `runs/predict/exp`. Add `--dataset_json_path dataset.json` to also export a COCO `result.json` for evaluation.

## Next steps

- [How Sliced Inference Works](guides/sliced-inference.md) for choosing slice size, overlap, and the merge strategy.
- [Model Integrations](guides/models.md) for HuggingFace, MMDetection, Detectron2, TorchVision, RT-DETR, RF-DETR, and the rest.
- [Prediction Utilities](predict.md) for batch inference, progress bars, and export options.
- [CLI Commands](cli.md) for every command and flag.
- [Interactive Notebooks](notebooks.md) for runnable Colab examples.
