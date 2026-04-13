---
tags:
  - coco
  - dataset
  - annotation
  - slicing
  - evaluation
---

# COCO 工具

SAHI 提供了一整套用于创建、操作和转换 COCO 格式数据集的工具。

## 创建数据集

```python
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation, CocoPrediction
from sahi.utils.file import save_json

# 初始化 COCO 数据集并添加类别
coco = Coco()
coco.add_category(CocoCategory(id=0, name="human"))
coco.add_category(CocoCategory(id=1, name="vehicle"))

# 创建图像条目
coco_image = CocoImage(file_name="image1.jpg", height=1080, width=1920)

# 添加真实标注
coco_image.add_annotation(
    CocoAnnotation(bbox=[x_min, y_min, width, height], category_id=0, category_name="human")
)
coco_image.add_annotation(
    CocoAnnotation(bbox=[x_min, y_min, width, height], category_id=1, category_name="vehicle")
)

# 添加模型预测结果（带置信度分数）
coco_image.add_prediction(
    CocoPrediction(score=0.86, bbox=[x_min, y_min, width, height], category_id=0, category_name="human")
)

# 将图像添加到数据集
coco.add_image(coco_image)

# 导出为 JSON
save_json(coco.json, "coco_dataset.json")

# 导出 COCO 结果格式的预测
save_json(coco.prediction_array, "coco_predictions.json")
```

### 使用 pycocotools 评估

```python
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

coco_gt = COCO(annotation_file="coco_dataset.json")
coco_dt = coco_gt.loadRes("coco_predictions.json")

evaluator = COCOeval(coco_gt, coco_dt, "bbox")
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
```

---

## 加载数据集

```python
from sahi.utils.coco import Coco

coco = Coco.from_coco_dict_or_path("coco.json")
```

---

## 切片图像和标注

将大图像及其 COCO 标注切片为更小的网格块：

```python
from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="coco.json",
    image_dir="source/coco/image/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

---

## 拆分为训练集/验证集

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

result = coco.split_coco_as_train_val(train_split_rate=0.85)

save_json(result["train_coco"].json, "train_split.json")
save_json(result["val_coco"].json, "val_split.json")
```

---

## 合并数据集

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco_1 = Coco.from_coco_dict_or_path("coco1.json", image_dir="images_1/")
coco_2 = Coco.from_coco_dict_or_path("coco2.json", image_dir="images_2/")

coco_1.merge(coco_2)

save_json(coco_1.json, "merged_coco.json")
```

---

## 筛选和更新

### 按类别筛选

选择特定类别并重新映射其 ID：

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

desired_name2id = {"big_vehicle": 1, "car": 2, "human": 3}
coco.update_categories(desired_name2id)

save_json(coco.json, "updated_coco.json")
```

### 按标注面积筛选

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

# 按最小面积筛选
area_filtered = coco.get_area_filtered_coco(min=50)

# 按最小和最大面积筛选
area_filtered = coco.get_area_filtered_coco(min=50, max_val=10000)

# 按类别设置面积区间
intervals = {
    "human": {"min": 20, "max": 10000},
    "vehicle": {"min": 50, "max": 15000},
}
area_filtered = coco.get_area_filtered_coco(intervals_per_category=intervals)

save_json(area_filtered.json, "area_filtered_coco.json")
```

### 保留无标注的图像

默认情况下，没有标注的图像会被排除。如需保留：

```python
coco = Coco.from_coco_dict_or_path("coco.json", ignore_negative_samples=False)
```

### 将边界框裁剪到图像尺寸范围内

```python
# 加载时裁剪
coco = Coco.from_coco_dict_or_path("coco.json", clip_bboxes_to_img_dims=True)

# 或对已有对象进行裁剪
coco = coco.get_coco_with_clipped_bboxes()
```

---

## 采样

### 下采样

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

# 保留 1/10 的图像
subsampled = coco.get_subsampled_coco(subsample_ratio=10)

# 仅对包含特定类别的图像进行下采样
subsampled = coco.get_subsampled_coco(subsample_ratio=10, category_id=0)

# 将负样本（无标注图像）缩减为 1/10
subsampled = coco.get_subsampled_coco(subsample_ratio=10, category_id=-1)

save_json(subsampled.json, "subsampled_coco.json")
```

### 上采样

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

# 将每个样本重复 10 次
upsampled = coco.get_upsampled_coco(upsample_ratio=10)

# 仅对包含特定类别的图像进行上采样
upsampled = coco.get_upsampled_coco(upsample_ratio=10, category_id=0)

save_json(upsampled.json, "upsampled_coco.json")
```

---

## 转换为 YOLO 格式

### 单数据集自动拆分

```python
from sahi.utils.coco import Coco

coco = Coco.from_coco_dict_or_path("coco.json", image_dir="coco_images/")

coco.export_as_yolo(output_dir="output/folder/dir", train_split_rate=0.85)
```

### 预拆分的训练集/验证集

```python
from sahi.utils.coco import Coco, export_coco_as_yolo

train_coco = Coco.from_coco_dict_or_path("train_coco.json", image_dir="coco_images/")
val_coco = Coco.from_coco_dict_or_path("val_coco.json", image_dir="coco_images/")

data_yml_path = export_coco_as_yolo(
    output_dir="output/folder/dir",
    train_coco=train_coco,
    val_coco=val_coco,
)
```

---

## 数据集统计

```python
from sahi.utils.coco import Coco

coco = Coco.from_coco_dict_or_path("coco.json")

print(coco.stats)
# {
#   'num_images': 6471,
#   'num_annotations': 343204,
#   'num_categories': 2,
#   'num_negative_images': 0,
#   'num_images_per_category': {'human': 5684, 'vehicle': 6323},
#   'num_annotations_per_category': {'human': 106396, 'vehicle': 236808},
#   'min_num_annotations_in_image': 1,
#   'max_num_annotations_in_image': 902,
#   'avg_num_annotations_in_image': 53.04,
#   'min_annotation_area': 3,
#   'max_annotation_area': 328640,
#   'avg_annotation_area': 2448.41,
#   'min_annotation_area_per_category': {'human': 3, 'vehicle': 3},
#   'max_annotation_area_per_category': {'human': 72670, 'vehicle': 328640},
# }
```

---

## 清理无效结果

从 COCO 结果 JSON 中移除无效预测：

```python
from sahi.utils.coco import remove_invalid_coco_results
from sahi.utils.file import save_json

coco_results = remove_invalid_coco_results("coco_result.json")
save_json(coco_results, "fixed_coco_result.json")

# 同时过滤掉超出图像尺寸的边界框
coco_results = remove_invalid_coco_results("coco_result.json", "coco_dataset.json")
```

---

## 更多资源

- [交互式 notebooks](../notebooks.md) -- 包含 COCO 数据集切片的动手实践示例
- [CLI 文档](cli.md) -- COCO 数据集的命令行操作
