# 预测工具

- 切片推理：

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init any model
detection_model = AutoDetectionModel.from_pretrained(model_type='mmdet',...) # for MMDetection models
detection_model = AutoDetectionModel.from_pretrained(model_type='ultralytics',...) # for YOLOv8/YOLO11/YOLO12 models
detection_model = AutoDetectionModel.from_pretrained(model_type='huggingface',...) # for HuggingFace detection models
detection_model = AutoDetectionModel.from_pretrained(model_type='torchvision',...) # for Torchvision detection models
detection_model = AutoDetectionModel.from_pretrained(model_type='rtdetr',...) # for RT-DETR models
detection_model = AutoDetectionModel.from_pretrained(model_type='yoloe',...) # for YOLOE models
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov5',...) # for YOLOv5 models
detection_model = AutoDetectionModel.from_pretrained(model_type='yolo-world',...) # for YOLOWorld models
detection_model = AutoDetectionModel.from_pretrained(model_type='roboflow',...) # for Roboflow RFDETR detection/segmentation models

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

- 基础推理：

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

- 批量推理:

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
    progress_bar=False,
)

```

## 进度条

提供了两个选项来控制和接收切片推理过程中的进度更新：

- `progress_bar` (bool)：设为 True 时，在切片处理过程中显示 tqdm 进度条。适用于终端和 notebook 中的可视化反馈。默认为 False。
- `progress_callback` (callable)：一个回调函数，在每个切片（或切片组）处理完成后被调用。该回调接收两个整数参数：`(current_slice_index, total_slices)`。可用于集成自定义进度报告（例如，更新 GUI 元素或将进度记录到文件）。

使用回调的示例：

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# 初始化模型
detection_model = AutoDetectionModel.from_pretrained(...)

def my_progress_callback(current, total):
    print(f"已处理 {current}/{total} 个切片")

result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    progress_bar=False,           # 禁用 tqdm 进度条
    progress_callback=my_progress_callback,  # 使用回调接收进度更新
)
```

!!! tip "提示"
    - `progress_bar` 和 `progress_callback` 可以同时使用。当两者都提供时，tqdm 进度条会显示，同时回调函数也会在每个切片组处理后被调用。
    - `progress_callback` 使用从 1 开始的索引（即第一次调用为 `(1, total)`）。

- 在推理时排除自定义类别:

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

- 可视化参数与导出格式:

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

# 交互式示例和演示
想要马上看到这些预测工具在使用中的表现？我们有几个集成了不同模型的交互式的notebook：

- 对于 YOLOv8/YOLO11/YOLO12 模型，探索我们的 [Ultralytics集成 notebook](../../demo/inference_for_ultralytics.ipynb)
- 对于 YOLOv5 模型，查看我们的 [YOLOv5集成 notebook](../../demo/inference_for_yolov5.ipynb)
- 对于 MMDetection 模型，尝试我们的 [MMDetection集成 notebook](../../demo/inference_for_mmdetection.ipynb)
- 对于 HuggingFace 模型，查看我们的 [HuggingFace集成 notebook](../../demo/inference_for_huggingface.ipynb)
- 对于 TorchVision 模型, 探索我们的 [TorchVision集成 notebook](../../demo/inference_for_torchvision.ipynb)
- 对于 RT-DETR 模型，查看我们的 [RT-DETR集成 notebook](../../demo/inference_for_rtdetr.ipynb)

这些示例提供了快速上手的例子并且让你能够体验不同的参数和设置。
