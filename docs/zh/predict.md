---
tags:
  - inference
  - slicing
  - batch-inference
  - visualization
  - object-detection
  - small-object-detection
---

# 预测工具

## 切片推理

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# 初始化任意模型
detection_model = AutoDetectionModel.from_pretrained(model_type='mmdet',...) # MMDetection 模型
detection_model = AutoDetectionModel.from_pretrained(model_type='ultralytics',...) # YOLOv8/YOLO11/YOLO26 模型
detection_model = AutoDetectionModel.from_pretrained(model_type='huggingface',...) # HuggingFace 检测模型
detection_model = AutoDetectionModel.from_pretrained(model_type='torchvision',...) # Torchvision 检测模型
detection_model = AutoDetectionModel.from_pretrained(model_type='rtdetr',...) # RT-DETR 模型
detection_model = AutoDetectionModel.from_pretrained(model_type='yoloe',...) # YOLOE 模型
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov5',...) # YOLOv5 模型
detection_model = AutoDetectionModel.from_pretrained(model_type='yolo-world',...) # YOLOWorld 模型
detection_model = AutoDetectionModel.from_pretrained(model_type='roboflow',...) # Roboflow RFDETR 检测/分割模型

# 获取切片预测结果
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

```

## 全图推理

```python
from sahi.predict import get_prediction
from sahi import AutoDetectionModel

# 初始化模型
detection_model = AutoDetectionModel.from_pretrained(...)

# 获取标准预测结果
result = get_prediction(
    image,
    detection_model,
)

```

## 批量推理

### 对文件夹或文件列表进行批量预测

使用高级 `predict` 函数对多张图像一次性执行切片推理，并自动导出结果：

```python
from sahi.predict import predict
from sahi import AutoDetectionModel

# 初始化模型
detection_model = AutoDetectionModel.from_pretrained(...)

# 获取批量预测结果
result = predict(
    model_type=..., # 'ultralytics', 'mmdet', 'huggingface' 之一
    model_path=..., # 模型权重文件路径
    model_config_path=..., # 用于 mmdet 模型
    model_confidence_threshold=0.5,
    model_device='cpu', # 或 'cuda:0'
    source=..., # 图像或文件夹路径
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

### 底层批量推理 API

`perform_batch_inference` 允许你在单次调用中对多张图像运行模型，并获取每张图像的预测列表。Ultralytics YOLO 模型使用原生 GPU 批处理；所有其他模型回退到逐张图像的顺序推理，但使用相同的 API。

```python
import cv2
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)

# 加载一批图像为 numpy 数组 (H, W, C)，RGB 格式
images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]

# 运行批量推理（Ultralytics 使用原生 GPU 批处理）
detection_model.perform_batch_inference(images)

# 提供每张图像的偏移量和完整图像尺寸（当图像不是切片时使用 [[0, 0]] 默认值）
shift_amount_list = [[0, 0]] * len(images)
full_shape_list   = [[img.shape[0], img.shape[1]] for img in images]

detection_model.convert_original_predictions(
    shift_amount=shift_amount_list,
    full_shape=full_shape_list,
)

# 访问每张图像的预测结果
for i, preds in enumerate(detection_model.object_prediction_list_per_image):
    print(f"Image {i}: {len(preds)} detections")
    for pred in preds:
        print(pred.category.name, pred.score.value, pred.bbox.to_xyxy())
```

!!! note "单图像兼容性"
    现有的 `object_prediction_list` 属性保持不变，返回第一张图像的预测结果，因此使用 `perform_inference` + `convert_original_predictions` + `object_prediction_list` 的代码无需修改即可继续工作。

## 进度条

提供了两个选项来控制和接收切片推理过程中的进度更新：

- `progress_bar`
  (bool)：设为 True 时，在切片处理过程中显示 tqdm 进度条。适用于终端和 notebook 中的可视化反馈。默认为 False。
- `progress_callback`
  (callable)：一个回调函数，在每个切片（或切片组）处理完成后被调用。该回调接收两个整数参数：`(current_slice_index, total_slices)`。可用于集成自定义进度报告（例如，更新 GUI 元素或将进度记录到文件）。

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

## 在推理时排除自定义类别

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# 初始化模型
detection_model = AutoDetectionModel.from_pretrained(...)

# 定义要排除的类别名称
exclude_classes_by_name = ["car"]

# 或通过自定义 id 排除类别
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

## 可视化参数与导出格式

```python
from sahi.predict import get_prediction
from sahi import AutoDetectionModel
from PIL import Image

# 初始化模型
detection_model = AutoDetectionModel.from_pretrained(...)

# 获取预测结果
result = get_prediction(
    image,
    detection_model,
)

# 使用自定义可视化参数导出
result.export_visuals(
    export_dir="outputs/",
    text_size=1.0,  # 类别标签文字大小
    rect_th=2,      # 边界框线条粗细
    text_th=2,      # 文字粗细
    hide_labels=False,  # 设为 True 隐藏类别标签
    hide_conf=False,    # 设为 True 隐藏置信度分数
    color=(255, 0, 0),  # 自定义 RGB 颜色（此例为红色）
    file_name="custom_visualization",
    export_format="jpg"  # 支持 'jpg' 和 'png'
)

# 导出为 COCO 格式标注
coco_annotations = result.to_coco_annotations()
# 示例输出: [{'image_id': None, 'bbox': [x, y, width, height], 'category_id': 0, 'area': width*height, ...}]

# 导出为 COCO 预测格式（包含置信度分数）
coco_predictions = result.to_coco_predictions(image_id=1)
# 示例输出: [{'image_id': 1, 'bbox': [x, y, width, height], 'score': 0.98, 'category_id': 0, ...}]

# 导出为 imantics 格式
imantics_annotations = result.to_imantics_annotations()
# 用于 imantics 库: https://github.com/jsbroks/imantics

# 导出用于 FiftyOne 可视化
fiftyone_detections = result.to_fiftyone_detections()
# 用于 FiftyOne: https://github.com/voxel51/fiftyone
```

!!! tip "交互式示例"
    想要查看这些预测工具的实际效果？请查阅我们的[交互式 notebooks](notebooks.md)，其中包含每个支持框架的动手实践示例。
