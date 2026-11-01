---
tags:
  - fiftyone
  - visualization
  - coco
  - dataset
---

# FiftyOne Visualization

[FiftyOne](https://github.com/voxel51/fiftyone) provides an interactive UI for
exploring detection results, comparing predictions, and debugging model
performance.

Supported version: `pip install fiftyone>=0.14.2,<0.15.0`

## Explore a COCO Dataset

```python
from sahi.utils.fiftyone import launch_fiftyone_app

# Launch the FiftyOne app with your COCO dataset
session = launch_fiftyone_app(coco_image_dir, coco_json_path)

# When done, close the session
session.close()
```

## Visualize SAHI Predictions

Run sliced inference and convert results to FiftyOne format:

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

# Convert to FiftyOne detection format
fiftyone_detections = result.to_fiftyone_detections()
```

## Compare Multiple Detection Results

Use the CLI to visualize a dataset alongside multiple prediction results, ordered
by misdetections:

```bash
sahi coco fiftyone \
  --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json cocoresult2.json
```

Set the IOU threshold for FP/TP classification:

```bash
sahi coco fiftyone --iou_threshold 0.5 \
  --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json
```

## Predict and Explore in One Step

The `predict-fiftyone` CLI command runs sliced inference and opens results in
FiftyOne directly:

```bash
sahi predict-fiftyone \
  --image_dir images/ \
  --dataset_json_path dataset.json \
  --model_path yolo26n.pt \
  --model_type ultralytics \
  --slice_height 512 \
  --slice_width 512
```
