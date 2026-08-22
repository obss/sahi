---
tags:
  - coco
  - dataset
  - annotation
  - slicing
  - evaluation
---

# COCO Araçları

SAHI, COCO formatındaki veri kümelerini oluşturmak, işlemek ve dönüştürmek için kapsamlı bir araç seti sağlar.

## Veri Kümesi Oluşturma

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

### pycocotools ile Değerlendirme (Evaluation)

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

## Veri Kümesi Yükleme

```python
from sahi.utils.coco import Coco

coco = Coco.from_coco_dict_or_path("coco.json")
```

---

## Görselleri ve Annotations'ları Dilimleme (Slicing)

Büyük görselleri ve COCO annotation'larını daha küçük dilimlerden (tiles) oluşan bir ızgaraya dilimleyin:

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

## Train/Val Olarak Bölme

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

result = coco.split_coco_as_train_val(train_split_rate=0.85)

save_json(result["train_coco"].json, "train_split.json")
save_json(result["val_coco"].json, "val_split.json")
```

---

## Veri Kümelerini Birleştirme (Merging)

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco_1 = Coco.from_coco_dict_or_path("coco1.json", image_dir="images_1/")
coco_2 = Coco.from_coco_dict_or_path("coco2.json", image_dir="images_2/")

coco_1.merge(coco_2)

save_json(coco_1.json, "merged_coco.json")
```

---

## Filtreleme ve Güncelleme

### Kategorilere göre

Belirli kategorileri seçin ve ID'lerini yeniden haritalandırın:

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

coco = Coco.from_coco_dict_or_path("coco.json")

desired_name2id = {"big_vehicle": 1, "car": 2, "human": 3}
coco.update_categories(desired_name2id)

save_json(coco.json, "updated_coco.json")
```

### Annotation alanına (area) göre

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

### Annotation içermeyen görselleri koruma

Varsayılan olarak annotation içermeyen görseller hariç tutulur. Bunları korumak için:

```python
coco = Coco.from_coco_dict_or_path("coco.json", ignore_negative_samples=False)
```

### Bounding box'ları görsel boyutlarına kırpma (clipping)

```python
# On load
coco = Coco.from_coco_dict_or_path("coco.json", clip_bboxes_to_img_dims=True)

# Or on an existing object
coco = coco.get_coco_with_clipped_bboxes()
```

---

## Örnekleme (Sampling)

### Alt Örnekleme (Subsample)

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

### Üst Örnekleme (Upsample)

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

## YOLO Formatına Dönüştürme

### Otomatik bölmeli tek veri kümesi

```python
from sahi.utils.coco import Coco

coco = Coco.from_coco_dict_or_path("coco.json", image_dir="coco_images/")

coco.export_as_yolo(output_dir="output/folder/dir", train_split_rate=0.85)
```

### Önceden bölünmüş train/val veri kümeleri

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

## Veri Kümesi İstatistikleri

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

## Geçersiz Sonuçları Temizleme

Bir COCO sonuçları JSON dosyasındaki geçersiz prediction'ları kaldırın:

```python
from sahi.utils.coco import remove_invalid_coco_results
from sahi.utils.file import save_json

coco_results = remove_invalid_coco_results("coco_result.json")
save_json(coco_results, "fixed_coco_result.json")

# Also filter out bboxes exceeding image dimensions
coco_results = remove_invalid_coco_results("coco_result.json", "coco_dataset.json")
```

---

## Ek Kaynaklar

- [Etkileşimli Notebook'lar](notebooks.md): COCO veri kümesi dilimleme dahil uygulamalı örnekler
- [CLI dokümantasyonu](cli.md): COCO veri kümeleri için komut satırı operasyonları
