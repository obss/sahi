# COCO 工具

<details closed>
<summary>
<big><b>创建 COCO 数据集：</b></big>
</summary>

- 导入需要的类：

```python
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
```

- 初始化 COCO 对象：

```python
coco = Coco()
```

- 添加从 id 0 开始的类别：

```python
coco.add_category(CocoCategory(id=0, name='human'))
coco.add_category(CocoCategory(id=1, name='vehicle'))
```

- 创建COCO图像：

```python
coco_image = CocoImage(file_name="image1.jpg", height=1080, width=1920)
```

- 在COCO图像上添加标注：

```python
coco_image.add_annotation(
  CocoAnnotation(
    bbox=[x_min, y_min, width, height],
    category_id=0,
    category_name='human'
  )
)
coco_image.add_annotation(
  CocoAnnotation(
    bbox=[x_min, y_min, width, height],
    category_id=1,
    category_name='vehicle'
  )
)
```

- 为 COCO 图像添加预测结果：

```python
coco_image.add_prediction(
  CocoPrediction(
    score=0.864434,
    bbox=[x_min, y_min, width, height],
    category_id=0,
    category_name='human'
  )
)
coco_image.add_prediction(
  CocoPrediction(
    score=0.653424,
    bbox=[x_min, y_min, width, height],
    category_id=1,
    category_name='vehicle'
  )
)
```

- 将 COCO 图像添加到 COCO 对象中：

```python
coco.add_image(coco_image)
```

- 在添加了所有的图像后，将COCO对象转换为COCO数据集的json格式：

```python
coco_json = coco.json
```

- 你可以把它导出为 json 文件：

```python
from sahi.utils.file import save_json

save_json(coco_json, "coco_dataset.json")
```

- 你也可以导出COCO预测格式的预测结果数组并且将它保存为json文件：

```python
from sahi.utils.file import save_json

predictions_array = coco.prediction_array
save_json = save_json(predictions_array, "coco_predictions.json")
```

- 此预测数组可用于通过官方 pycocotool API 获取标准的 COCO 评估指标：

```python
# note:- pycocotools 需要另外安装
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

coco_ground_truth = COCO(annotation_file="coco_dataset.json")
coco_predictions = coco_ground_truth.loadRes("coco_predictions.json")

coco_evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()
```

</details>

<details closed>
<summary>
<big><b>将 COCO 数据集的图像和标注分割成片：</b></big>
</summary>

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

</details>

<details closed>
<summary>
<big><b>将 COCO 数据集分割为训练/验证集：</b></big>
</summary>

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

# 指定 COCO 数据集路径
coco_path = "coco.json"

# 初始化 COCO 对象
coco = Coco.from_coco_dict_or_path(coco_path)

# 将 COCO 数据集分割为 85% 的训练集 和 15% 的验证集
result = coco.split_coco_as_train_val(
  train_split_rate=0.85
)

# 导出分割完的文件
save_json(result["train_coco"].json, "train_split.json")
save_json(result["val_coco"].json, "val_split.json")
```

</details>

<details closed>
<summary>
<big><b>按类别筛选/更新 COCO 数据集：</b></big>
</summary>

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json


# 通过指定 COCO 数据集和图像文件夹路径初始化 COCO 对象
coco = Coco.from_coco_dict_or_path("coco.json")

# 只选择3个类别并将其映射到不同 id 值
desired_name2id = {
  "big_vehicle": 1,
  "car": 2,
  "human": 3
}
coco.update_categories(desired_name2id)

# 导出更新和筛选好的 COCO 数据集
save_json(coco.json, "updated_coco.json")
```

</details>

<details closed>
<summary>
<big><b>通过标注区域筛选 COCO 数据集：</b></big>
</summary>

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

# 通过指定 COCO 数据集和图像文件夹路径初始化 COCO 对象
coco = Coco.from_coco_dict_or_path("coco.json")

# 过滤掉包含面积小于50的标注图像
area_filtered_coco = coco.get_area_filtered_coco(min=50)
# 过滤掉包含面积小于50大于10000的标注图像
area_filtered_coco = coco.get_area_filtered_coco(min=50, max_val=10000)
过滤掉
# 根据每个类别独立的面积区间来筛选图像
intervals_per_category = {
  "human": {"min": 20, "max": 10000},
  "vehicle": {"min": 50, "max": 15000},
}
area_filtered_coco = coco.get_area_filtered_coco(intervals_per_category=intervals_per_category)

# 导出筛选好的 COCO 数据集
save_json(area_filtered_coco.json, "area_filtered_coco.json")
```

</details>

<details closed>
<summary>
<big><b>过滤掉不包含任何标注的图像：</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# 如果你希望在 JSON 和 YOLO 导出中保留没有标注的图像，请将 ignore_negative_samples 设置为 False
coco = Coco.from_coco_dict_or_path("coco.json", ignore_negative_samples=False)

```

</details>

<details closed>
<summary>
<big><b>合并 COCO 数据集文件：</b></big>
</summary>

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

# 通过指定 COCO 数据集路径和图像文件夹目录来初始化 COCO 对象
coco_1 = Coco.from_coco_dict_or_path("coco1.json", image_dir="images_1/")
coco_2 = Coco.from_coco_dict_or_path("coco2.json", image_dir="images_2/")

# 合并 COCO 数据集
coco_1.merge(coco_2)

# 导出合并后的 COCO 数据集
save_json(coco_1.json, "merged_coco.json")
```

</details>

<details closed>
<summary>
<big><b>将 COCO 数据集转换为 ultralytics/YOLO 格式：</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# 初始化 COCO 对象
coco = Coco.from_coco_dict_or_path("coco.json", image_dir="coco_images/")

# 将转换后的 YOLO 格式数据集导出到指定的 output_dir 中，并按 85% 训练集 / 15% 验证集划分。
coco.export_as_yolo(
  output_dir="output/folder/dir",
  train_split_rate=0.85
)
```

</details>

<details closed>
<summary>
<big><b>将训练/验证 COCO 数据集转换为 ultralytics/YOLO 格式：</b></big>
</summary>

```python
from sahi.utils.coco import Coco, export_coco_as_yolo

# 初始化 COCO 对象
train_coco = Coco.from_coco_dict_or_path("train_coco.json", image_dir="coco_images/")
val_coco = Coco.from_coco_dict_or_path("val_coco.json", image_dir="coco_images/")

# 将转换后的 YOLO 格式数据集导出到指定的 output_dir 中，并按照给定的训练集/验证集划分。
data_yml_path = export_coco_as_yolo(
  output_dir="output/folder/dir",
  train_coco=train_coco,
  val_coco=val_coco
)
```

</details>

<details closed>
<summary>
<big><b>对 COCO 数据集文件进行抽样：</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# 指定 COCO 数据集路径
coco_path = "coco.json"

# 初始化 COCO 对象
coco = Coco.from_coco_dict_or_path(coco_path)

# 创建一个包含全部图像十分之一的 COCO 对象
subsampled_coco = coco.get_subsampled_coco(subsample_ratio=10)

# 导出抽样后的 COCO 数据集
save_json(subsampled_coco.json, "subsampled_coco.json")

# bonus: 基于仅含“第一个类别”的图像构建 COCO 对象，样本规模为总图像数的 1/10
subsampled_coco = coco.get_subsampled_coco(subsample_ratio=10, category_id=0)

# bonus2: 构建 COCO 对象，并将负样本（无标注图像）数量抽样 1/10
subsampled_coco = coco.get_subsampled_coco(subsample_ratio=10, category_id=-1)
```
</details>

<details closed>
<summary>
<big><b>上采样 COCO 数据集文件：</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# 指定 COCO 数据集路径
coco_path = "coco.json"

# 初始化 COCO 对象
coco = Coco.from_coco_dict_or_path(coco_path)

# 创建一个 COCO 对象，其中每个样本都会被重复 10 次
upsampled_coco = coco.get_upsampled_coco(upsample_ratio=10)

# 导出上采样后的数据集
save_json(upsampled_coco.json, "upsampled_coco.json")

# bonus: 构建 COCO 对象，并将包含“第一个类别”的图像重复采样 10 次
subsampled_coco = coco.get_subsampled_coco(upsample_ratio=10, category_id=0)

# bonus2: 构建 COCO 对象，并将负样本上采样 10 倍
upsampled_coco = coco.get_upsampled_coco(upsample_ratio=10, category_id=-1)
```
</details>

<details closed>
<summary>
<big><b>获取数据集统计信息：</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# 初始化 COCO 对象
coco = Coco.from_coco_dict_or_path("coco.json")

# 获取数据集的统计信息
coco.stats
{
  'num_images': 6471,
  'num_annotations': 343204,
  'num_categories': 2,
  'num_negative_images': 0,
  'num_images_per_category': {'human': 5684, 'vehicle': 6323},
  'num_annotations_per_category': {'human': 106396, 'vehicle': 236808},
  'min_num_annotations_in_image': 1,
  'max_num_annotations_in_image': 902,
  'avg_num_annotations_in_image': 53.037243084530985,
  'min_annotation_area': 3,
  'max_annotation_area': 328640,
  'avg_annotation_area': 2448.405738278109,
  'min_annotation_area_per_category': {'human': 3, 'vehicle': 3},
  'max_annotation_area_per_category': {'human': 72670, 'vehicle': 328640},
}

```
</details>

<details closed>
<summary>
<big><b>移除无效的 COCO 结果：</b></big>
</summary>

```python
from sahi.utils.file import save_json
from sahi.utils.coco import remove_invalid_coco_results

# 从 COCO 结果的 JSON 中移除无效预测
coco_results = remove_invalid_coco_results("coco_result.json")

# 导出经过处理的 COCO 结果
save_json(coco_results, "fixed_coco_result.json")

# bonus: 通过提供 COCO 数据集路径，从 COCO 结果 JSON 中移除无效预测
# 同时过滤掉超出图像高度和宽度范围的 bbox 结果
coco_results = remove_invalid_coco_results("coco_result.json", "coco_dataset.json")
```
</details>

<details closed>
<summary>
<big><b>获取边界框经过裁剪的 COCO 数据：</b></big>
</summary>

- 导入必要的类：

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json
```
用法：

```python
# 将越界的边界框裁剪到图像的宽度和高度范围内
coco = Coco.from_coco_dict_or_path(coco_path, clip_bboxes_to_img_dims=True)
```
或者，

```python
# 应用到你已经创建好的 COCO 对象
coco = coco.get_coco_with_clipped_bboxes()
```

- 导出裁剪后有边界框的 COCO 对象:

```python
save_json(coco.json, "coco.json")
```
</details>

# 交互式示例和附加资源

想要马上看到 COCO 工具在使用中的表现？这里有一些有用的资源：

- 关于 COCO 数据集切片的动手实践示例，查看我们的 [切片示例 notebook](../../demo/slicing.ipynb)
- 要了解如何使用 COCO 数据集进行预测与可视化，探索我们在 [demo 目录](../../demo/) 中特定模型的notebook
- 关于 COCO 数据集的命令行操作, 参考我们的 [CLI 文档](cli.md)

这些资源提供了实用的例子和详细的解释，来帮助你用 SAHI 来高效的处理 COCO 数据集。
