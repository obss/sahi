# Quick Start

Welcome to SAHI! This guide will get you up and running with the core features of the library, including installation, performing predictions, and using the command-line interface.

## 1. Installation

Install SAHI using pip. For object detection, it's recommended to also install `ultralytics`.

```bash
pip install sahi ultralytics
```

## 2. Sliced Prediction with Python

Sliced inference is the core feature of SAHI, allowing you to detect small objects in large images. Here's a simple example using the Python API:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Initialize a YOLOv8 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n.pt', # or any other YOLOv8 model
    confidence_threshold=0.25,
    device="cuda:0", # or "cpu"
)

# Run sliced prediction
result = get_sliced_prediction(
    "path/to/your/image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# Export visualizations
result.export_visuals(export_dir="demo_data/")

# Get predictions as a list of objects
predictions = result.object_prediction_list
```

## 3. Prediction with the CLI

SAHI also provides a powerful command-line interface for quick predictions without writing any Python code.

```bash
sahi predict --model_path yolov8n.pt --model_type yolov8 --source /path/to/images/ --slice_height 512 --slice_width 512
```

This command will run sliced inference on all images in the specified directory and save the results.

## Next Steps

You've now seen the basics of SAHI! To dive deeper, check out these resources:

* **Prediction In-Depth**: For advanced prediction options, see the [Prediction Utilities guide](predict.md).
* **Demos**: Explore our interactive notebooks in the [demo directory](../demo/) for hands-on examples with different models.
* **COCO Tools**: Learn how to create, manipulate, and convert datasets in the [COCO Utilities guide](coco.md).
* **All CLI Commands**: See the full list of commands in the [CLI documentation](cli.md).