---
tags:
  - fiftyone
  - visualization
  - coco
  - dataset
---

# FiftyOne Görselleştirmesi

[FiftyOne](https://github.com/voxel51/fiftyone), tespit sonuçlarını incelemek, tahminleri karşılaştırmak ve model performansında hata ayıklamak (debug) için etkileşimli bir kullanıcı arayüzü sunar.

Desteklenen sürüm: `pip install fiftyone>=0.14.2,<0.15.0`

## COCO Veri Kümesini İnceleme

```python
from sahi.utils.fiftyone import launch_fiftyone_app

# Launch the FiftyOne app with your COCO dataset
session = launch_fiftyone_app(coco_image_dir, coco_json_path)

# When done, close the session
session.close()
```

## SAHI Prediction'larını Görselleştirme

Sliced Inference çalıştırın ve sonuçları FiftyOne formatına dönüştürün:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Convert to FiftyOne detection format
fiftyone_detections = result.to_fiftyone_detections()
```

## Birden Fazla Tespit Sonucunu Karşılaştırma

Bir veri kümesini yanlış tespitlere göre sıralanmış birden fazla prediction sonucuyla birlikte görselleştirmek için CLI'yi kullanın:

```bash
sahi coco fiftyone \
  --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json cocoresult2.json
```

FP/TP sınıflandırması için IOU eşiğini belirleyin:

```bash
sahi coco fiftyone --iou_threshold 0.5 \
  --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json
```

## Tek Adımda Tahmin Yürütme ve İnceleme

`predict-fiftyone` CLI komutu Sliced Inference çalıştırır ve sonuçları doğrudan FiftyOne uygulamasında açar:

```bash
sahi predict-fiftyone \
  --image_dir images/ \
  --dataset_json_path dataset.json \
  --model_path yolo26n.pt \
  --model_type ultralytics \
  --slice_height 512 \
  --slice_width 512
```
