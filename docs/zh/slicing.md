---
tags:
  - slicing
  - api-reference
  - coco
  - dataset
  - small-object-detection
---

# 切片

## 切片

::: sahi.slicing

### 切片工具

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

- 流式读取超大图像的切片，无需整幅解码（`get_sliced_prediction` 内部即使用此方式；
  安装 `sahi[bigimage]` 可启用基于 libvips 的行带读取器，否则会回退为整幅解码）：

```python
from sahi.slicing import SliceImageStream

stream = SliceImageStream(
    image=image_path,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
for batch_images, batch_starts in stream.iter_batches(batch_size=8):
    ...  # 任意时刻内存中最多只驻留一个行带
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

### 交互式示例

想要体验不同的切片参数并查看其效果？请查阅我们的[交互式 notebooks](notebooks.md)，其中展示了这些切片操作的实际应用。
