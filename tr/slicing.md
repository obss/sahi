---
tags:
  - slicing
  - api-reference
  - coco
  - dataset
  - small-object-detection
---

# Slicing

## Slicing

::: sahi.slicing

### Slicing Araçları

- Bir görseli dilimleyin:

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

- COCO formatındaki bir veri kümesini dilimleyin:

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

### Etkileşimli İnceleme

Farklı dilimleme parametreleriyle denemeler yapıp etkilerini görmek ister misiniz? Uygulamalı örnekler için [etkileşimli notebook'larımıza](notebooks.md) göz atın.
