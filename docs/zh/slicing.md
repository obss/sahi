# 切片

::: sahi.slicing

## 切片工具

- 对一张图片进行切片操作：

```python
from sahi.slicing import slice_image

slice_image_result = slice_image(
    image=image_path,
    output_file_name=output_file_name,
    output_dir=output_dir,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

- 对一个 COCO 格式数据集进行切片操作：

```python
from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=image_dir,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

# 交互式示例

想要体验不同的切片参数并查看其效果？请查阅我们的 [交互式 Jupyter notebook](../demo/slicing.ipynb) ，其中展示了这些切片操作的实际应用。
