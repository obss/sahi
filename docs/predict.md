# Prediction Utilities

- Sliced inference:

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init any model
detection_model = AutoDetectionModel.from_pretrained(model_type='mmdet',...) # for MMDetection models
detection_model = AutoDetectionModel.from_pretrained(model_type='ultralytics',...) # for YOLOv8/YOLO11/YOLO12 models
detection_model = AutoDetectionModel.from_pretrained(model_type='huggingface',...) # for HuggingFace detection models
detection_model = AutoDetectionModel.from_pretrained(model_type='torchvision',...) # for Torchvision detection models

# get sliced prediction result
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

```

- Standard inference:

```python
from sahi.predict import get_prediction
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get standard prediction result
result = get_prediction(
    image,
    detection_model,
)

```

- Batch inference:

```python
from sahi.predict import predict
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get batch predict result
result = predict(
    model_type=..., # one of 'ultralytics', 'mmdet', 'huggingface'
    model_path=..., # path to model weight file
    model_config_path=..., # for mmdet models
    model_confidence_threshold=0.5,
    model_device='cpu', # or 'cuda:0'
    source=..., # image or folder path
    no_standard_prediction=True,
    no_sliced_prediction=False,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2, 
    overlap_width_ratio=0.2,
    export_pickle=False,
    export_crop=False,
)

```

- Exclude custom classes on inference:

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# define the class names to exclude from custom model inference
exclude_classes_by_name = ["car"]

# or exclude classes by its custom id
exclude_classes_by_id = [0]

result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
    exclude_classes_by_name = exclude_classes_by_name
    # exclude_classes_by_id = exclude_classes_by_id
)

```

- Visualization parameters and export formats:

```python
from sahi.predict import get_prediction
from sahi import AutoDetectionModel
from PIL import Image

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get prediction result
result = get_prediction(
    image,
    detection_model,
)

# Export with custom visualization parameters
result.export_visuals(
    export_dir="outputs/",
    text_size=1.0,  # Size of the class label text
    rect_th=2,      # Thickness of bounding box lines
    text_th=2,      # Thickness of the text
    hide_labels=False,  # Set True to hide class labels
    hide_conf=False,    # Set True to hide confidence scores
    color=(255, 0, 0),  # Custom color in RGB format (red in this example)
    file_name="custom_visualization",
    export_format="jpg"  # Supports 'jpg' and 'png'
)

# Export as COCO format annotations
coco_annotations = result.to_coco_annotations()
# Example output: [{'image_id': None, 'bbox': [x, y, width, height], 'category_id': 0, 'area': width*height, ...}]

# Export as COCO predictions (includes confidence scores)
coco_predictions = result.to_coco_predictions(image_id=1)
# Example output: [{'image_id': 1, 'bbox': [x, y, width, height], 'score': 0.98, 'category_id': 0, ...}]

# Export as imantics format
imantics_annotations = result.to_imantics_annotations()
# For use with imantics library: https://github.com/jsbroks/imantics

# Export for FiftyOne visualization
fiftyone_detections = result.to_fiftyone_detections()
# For use with FiftyOne: https://github.com/voxel51/fiftyone
```

# Interactive Demos and Examples
Want to see these prediction utilities in action? We have several interactive notebooks that demonstrate different model integrations:

- For YOLOv8/YOLO11/YOLO12 models, explore our [Ultralytics integration notebook](../demo/inference_for_ultralytics.ipynb)
- For YOLOv5 models, check out our [YOLOv5 integration notebook](../demo/inference_for_yolov5.ipynb)
- For MMDetection models, try our [MMDetection integration notebook](../demo/inference_for_mmdetection.ipynb)
- For HuggingFace models, see our [HuggingFace integration notebook](../demo/inference_for_huggingface.ipynb)
- For TorchVision models, explore our [TorchVision integration notebook](../demo/inference_for_torchvision.ipynb)
- For RT-DETR models, check out our [RT-DETR integration notebook](../demo/inference_for_rtdetr.ipynb)

These notebooks provide hands-on examples and allow you to experiment with different parameters and settings.
