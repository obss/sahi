# Fiftyone 工具

- 通过 FiftyOne 应用来探索COCO数据集：

支持版本： `pip install fiftyone>=0.14.2<0.15.0`

```python
from sahi.utils.fiftyone import launch_fiftyone_app

# launch fiftyone app:
session = launch_fiftyone_app(coco_image_dir, coco_json_path)

# close fiftyone app:
session.close()
```

- 将预测结果转换为 FiftyOne 检测格式：

```python
from sahi import get_sliced_prediction

# perform sliced prediction
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

# convert detections into fiftyone detection format
fiftyone_detections = result.to_fiftyone_detections()
```

- 在 Fiftyone UI 中探索检测结果：

```bash
sahi coco fiftyone --image_dir dir/to/images --dataset_json_path dataset.json cocoresult1.json cocoresult2.json
```

该操作将打开一个 FiftyOne 应用，用于可视化给定的数据集和 2 个检测结果

使用 `--iou_threshold 0.5` 参数来指定用于判断 FP/TP 的 IOU 阈值。
