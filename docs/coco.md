---
tags:
  - coco
  - dataset
  - annotation
  - slicing
  - evaluation
---

# COCO Utilities

SAHI provides a full suite of tools for creating, manipulating, and converting
COCO format datasets.

## Creating a Dataset

```python
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation, CocoPrediction
from sahi.utils.file import save_json

# Initialize a COCO dataset and add categories
coco = Coco()
coco.add_category(CocoCategory(id=0, name="human"))
coco.add_category(CocoCategory(id=1, name="vehicle"))

# Create an image entry
coco_image = CocoImage(file_name="image1.jpg", height=1080, width=1920)

# Add ground-truth annotations
coco_image.add_annotation(
    CocoAnnotation(bbox=[x_min, y_min, width, height], category_id=0, category_name="human")
)
coco_image.add_annotation(
    CocoAnnotation(bbox=[x_min, y_min, width, height], category_id=1, category_name="vehicle")
)

# Add model predictions (with confidence scores)
coco_image.add_prediction(
    CocoPrediction(score=0.86, bbox=[x_min, y_min, width, height], category_id=0, category_name="human")
)

# Add the image to the dataset
coco.add_image(coco_image)

# Export as JSON
save_json(coco.json, "coco_dataset.json")

# Export predictions in COCO result format
save_json(coco.prediction_array, "coco_predictions.json")
```

### Evaluating with pycocotools

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

## Loading a Dataset

```python
from sahi.utils.coco import Coco

coco = Coco.from_coco_dict_or_path("coco.json")
```

---

## Slicing Images and Annotations

Slice large images and their COCO annotations into a grid of smaller tiles:

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

## Splitting into Train/Val

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

result = coco.split_coco_as_train_val(train_split_rate=0.85)

save_json(result["train_coco"].json, "train_split.json")
save_json(result["val_coco"].json, "val_split.json")
```

---

## Merging Datasets

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco_1 = Coco.from_coco_dict_or_path("coco1.json", image_dir="images_1/")
coco_2 = Coco.from_coco_dict_or_path("coco2.json", image_dir="images_2/")

coco_1.merge(coco_2)

save_json(coco_1.json, "merged_coco.json")
```

---

## Filtering and Updating

### By categories

Select specific categories and remap their IDs:

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

desired_name2id = {"big_vehicle": 1, "car": 2, "human": 3}
coco.update_categories(desired_name2id)

save_json(coco.json, "updated_coco.json")
```

### By annotation area

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

# Filter by minimum area
area_filtered = coco.get_area_filtered_coco(min=50)

# Filter by min and max area
area_filtered = coco.get_area_filtered_coco(min=50, max_val=10000)

# Per-category area intervals
intervals = {
    "human": {"min": 20, "max": 10000},
    "vehicle": {"min": 50, "max": 15000},
}
area_filtered = coco.get_area_filtered_coco(intervals_per_category=intervals)

save_json(area_filtered.json, "area_filtered_coco.json")
```

### Keep images without annotations

By default, images without annotations are excluded. To keep them:

```python
coco = Coco.from_coco_dict_or_path("coco.json", ignore_negative_samples=False)
```

### Clip bounding boxes to image dimensions

```python
# On load
coco = Coco.from_coco_dict_or_path("coco.json", clip_bboxes_to_img_dims=True)

# Or on an existing object
coco = coco.get_coco_with_clipped_bboxes()
```

---

## Sampling

### Subsample

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

# Keep 1/10 of images
subsampled = coco.get_subsampled_coco(subsample_ratio=10)

# Subsample only images containing a specific category
subsampled = coco.get_subsampled_coco(subsample_ratio=10, category_id=0)

# Reduce negative samples (images without annotations) to 1/10
subsampled = coco.get_subsampled_coco(subsample_ratio=10, category_id=-1)

save_json(subsampled.json, "subsampled_coco.json")
```

### Upsample

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

# Repeat each sample 10 times
upsampled = coco.get_upsampled_coco(upsample_ratio=10)

# Upsample only images containing a specific category
upsampled = coco.get_upsampled_coco(upsample_ratio=10, category_id=0)

save_json(upsampled.json, "upsampled_coco.json")
```

---

## Converting to YOLO Format

### Single dataset with auto-split

```python
from sahi.utils.coco import Coco

coco = Coco.from_coco_dict_or_path("coco.json", image_dir="coco_images/")

coco.export_as_yolo(output_dir="output/folder/dir", train_split_rate=0.85)
```

### Pre-split train/val datasets

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

## Dataset Statistics

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

## Cleaning Invalid Results

Remove invalid predictions from a COCO results JSON:

```python
from sahi.utils.coco import remove_invalid_coco_results
from sahi.utils.file import save_json

coco_results = remove_invalid_coco_results("coco_result.json")
save_json(coco_results, "fixed_coco_result.json")

# Also filter out bboxes exceeding image dimensions
coco_results = remove_invalid_coco_results("coco_result.json", "coco_dataset.json")
```

---

## Additional Resources

- [Interactive notebooks](notebooks.md) -- Hands-on examples including COCO
  dataset slicing
- [CLI documentation](cli.md) -- Command-line operations for COCO datasets
