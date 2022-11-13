# Prediction Utilities

- Sliced inference:

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init any model
detection_model = AutoDetectionModel.from_pretrained(model_type='mmdet',...) # for MMDetection models
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov5',...) # for YOLOv5 models
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
    model_type=..., # one of 'yolov5', 'mmdet', 'detectron2'
    model_path=..., # path to model weight file
    model_config_path=..., # for detectron2 and mmdet models
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
