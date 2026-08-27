---
tags:
  - models
  - inference
  - ultralytics
  - mmdetection
  - huggingface
  - torchvision
  - detectron2
  - yolov5
  - roboflow
---

# Model Entegrasyonları

SAHI, birleştirilmiş (tekil) bir API üzerinden tüm object detection kütüphaneleriyle çalışır. Modelinizi `AutoDetectionModel.from_pretrained()` ile bir kez yükleyin, ardından istediğiniz tüm SAHI fonksiyonlarıyla (Sliced Prediction, batch inference, CLI vb.) kullanın.

## Ultralytics (YOLO)

[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11), [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8) ve segmentation ile oriented bounding box modelleri dahil tüm Ultralytics varyantlarını destekler.

```bash
pip install ultralytics
```

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cuda:0",  # or "cpu"
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

Ultralytics modelleri ayrıca birden fazla dilimi daha hızlı işlemek için **yerel GPU batch inference** özelliğini destekler:

```python
result = get_sliced_prediction(
    "large_image.jpg",
    detection_model,
    slice_height=640,
    slice_width=640,
    batch_size=8,  # process 8 slices at once
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb)

---

## YOLOE

Prompt-free ve open-vocabulary tespiti destekleyen YOLOE modelleri.

```bash
pip install ultralytics
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yoloe",
    model_path="yoloe-v8l-seg.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb)

---

## YOLO-World (Zero-Shot)

Open-vocabulary tespit: yeniden eğitim gerektirmeden metin açıklamasıyla (text description) nesneleri tespit edin.

```bash
pip install ultralytics
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolo-world",
    model_path="yolov8s-worldv2.pt",
    confidence_threshold=0.1,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

---

## YOLOv5

`yolov5` pip paketi üzerinden klasik YOLOv5 modelleri.

```bash
pip install yolov5
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="yolov5",
    model_path="yolov5s.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb)

---

## HuggingFace Transformers

HuggingFace Hub'dan object detection ve zero-shot object detection modellerini kullanın (DETR, Deformable DETR, DETA, GroundingDINO vb.).

```bash
pip install transformers timm
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="huggingface",
    model_path="facebook/detr-resnet-50",
    confidence_threshold=0.3,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

GroundingDINO modelleri metin koşullu (text-conditioned) inference gerektirir. Hedef kategoriler bilindiğinde `text_labels` kullanın; böylece SAHI bu etiketlere sabit kategori id'leri atayabilir. İşlemci tarafından döndürülen ek grounded ifadeler yeni kategoriler olarak eklenir.

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="huggingface",
    model_path="IDEA-Research/grounding-dino-tiny",
    confidence_threshold=0.25,
    text_threshold=0.20,
    text_labels=["car", "truck", "person"],
    device="cuda:0",
)
```

### Zero-shot parametreleri

[Ortak parametrelere](#ortak-parametreler) ek olarak zero-shot (GroundingDINO) modelleri şunları kabul eder:

| Parametre | Tip | Açıklama |
| ----------- | ------ | ------------- |
| `text_labels` | `list[str]` | Tespit edilecek sabit kategoriler, örn. `["car", "truck"]`. Her biri sabit bir kategori id'si alır; bu liste dışındaki ifadeler elenir |
| `text_prompt` | str | `text_labels` ayarlanmadığında kullanılan serbest metin istemi (örn. `"a car. a truck."`); döndürülen ifadeler dinamik kategoriler haline gelir |
| `text_threshold` | float | Bir kutuyu bir metin token'ı ile eşleştirmek için minimum skor (varsayılan: 0.25) |

HuggingFace object detection notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb)

GroundingDINO zero-shot detection notebook:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_groundingdino.ipynb)

---

## HuggingFace Segmentation

HuggingFace Hub'dan segmentation modellerini çalıştırın. SAHI her bir segmenti poligon maskeli bir `ObjectPrediction` olarak döndürür; böylece sliced inference ve postprocessing işlemleri detection ile aynı şekilde çalışır.

| Mimarisi    | `instance` | `semantic` | `panoptic` |
| ----------- | :--------: | :--------: | :--------: |
| MaskFormer  |     ✅     |     ✅     |     ✅     |
| Mask2Former |     ✅     |     ✅     |     ✅     |
| OneFormer   |     ✅     |     ✅     |     ✅     |

Mevcut başlıklar (heads) kontrol noktasına (checkpoint) bağlıdır (örn. `facebook/mask2former-swin-tiny-coco-instance` yalnızca instance destekler). OneFormer inference anında başlığı seçer, dolayısıyla tek bir checkpoint her üçü için de hizmet verebilir.

```bash
pip install transformers timm
```

```python
from sahi.models.huggingface_segmentation import SegmentationType

detection_model = AutoDetectionModel.from_pretrained(
    model_type="huggingface_segmentation",
    model_path="facebook/mask2former-swin-tiny-coco-instance",
    confidence_threshold=0.5,
    device="cuda:0",
    segmentation_type=SegmentationType.INSTANCE_SEGMENTATION,
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

Uygun başlığı kullanmak için `segmentation_type` seçeneğini `SEMANTIC_SEGMENTATION` veya `PANOPTIC_SEGMENTATION` olarak değiştirin. Semantic segmentation sınıfın tüm örneklerini tek bir maskede birleştirir, dolayısıyla nesne başına değil sınıf başına bir `ObjectPrediction` döndürülür.

### Segmentation parametreleri

[Ortak parametrelere](#ortak-parametreler) ek olarak bu model şunları kabul eder:

| Parametre | Tip | Açıklama |
| ----------- | ------ | ------------- |
| `segmentation_type` | `SegmentationType` | `INSTANCE_SEGMENTATION` (varsayılan), `SEMANTIC_SEGMENTATION` veya `PANOPTIC_SEGMENTATION` |
| `min_segment_area` | int | Bu piksel sayısından küçük segmentleri ele (varsayılan: 100) |
| `overlap_mask_area_threshold` | float | Bir maske içindeki bağlantısız parçaları birleştir/at (varsayılan: 0.8) |
| `label_ids_to_fuse` | `list[int]` | Yalnızca panoptic, bu etiketlerin tüm örneklerini tek bir segmentte birleştirir |
| `token` | str | Korumalı/özel (gated/private) modeller için HuggingFace erişim token'ı (`$HF_TOKEN` değerine düşer) |

---

## RT-DETR

Yüksek doğrulukta gerçek zamanlı tespit için Real-Time Detection Transformer.

```bash
pip install transformers timm
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="rtdetr",
    model_path="PekingU/rtdetr_r50vd",
    confidence_threshold=0.3,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb)

---

## TorchVision

Dahili TorchVision tespit modellerini kullanın (Faster R-CNN, RetinaNet, FCOS, SSD vb.).

```bash
pip install torch torchvision
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="torchvision",
    model_path="fasterrcnn_resnet50_fpn",
    confidence_threshold=0.3,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb)

---

## MMDetection

Tüm MMDetection model havuzunu (300+ model) destekler.

```bash
pip install mmdet mmcv mmengine
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="mmdet",
    model_path="path/to/checkpoint.pth",
    config_path="path/to/config.py",
    confidence_threshold=0.25,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb)

---

## Detectron2

Detection ve instance segmentation için Facebook'un Detectron2 modellerini kullanın.

```bash
pip install detectron2
```

```python
detection_model = AutoDetectionModel.from_pretrained(
    model_type="detectron2",
    model_path="path/to/model_final.pth",
    config_path="path/to/config.yaml",
    confidence_threshold=0.25,
    device="cuda:0",
)

result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb)

---

## Roboflow (RF-DETR)

Detection ve segmentation için Roboflow'un RF-DETR modellerini kullanın.

```bash
pip install rfdetr
```

Modeli `model` argümanı ile geçin. Düz metin bir dize Roboflow Universe model id'si olarak değerlendirilir ve bir API anahtarı gerektirir; bir RF-DETR sınıf adı ise yerel bir model seçer.

=== "Roboflow Universe (API key)"

    ```python
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model="rfdetr-base",  # Universe model id
        api_key="YOUR_API_KEY",  # or set ROBOFLOW_API_KEY
        confidence_threshold=0.3,
        device="cuda:0",
    )
    ```

=== "Local weights (no API key)"

    ```python
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="roboflow",
        model="RFDETRSegMedium",  # RF-DETR class name, or the class/instance itself
        model_path="checkpoint.pth",  # your locally trained weights
        category_mapping={"0": "cat", "1": "dog"},  # required for custom classes
        image_size=640,  # must match the training resolution
        confidence_threshold=0.3,
        device="cuda:0",
    )
    ```

```python
result = get_sliced_prediction(
    "image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
)
```

Kullanılabilir RF-DETR modelleri: `RFDETRBase`, `RFDETRNano`, `RFDETRSmall`, `RFDETRMedium`, `RFDETRLarge`, `RFDETRSegNano`, `RFDETRSegSmall`, `RFDETRSegMedium`, `RFDETRSegLarge`, `RFDETRSegXLarge`, `RFDETRSeg2XLarge`.

`category_mapping` değeri `num_classes` parametresini belirler, bu nedenle özel bir model buna ihtiyaç duyar aksi takdirde başlık checkpoint ile eşleşmez. `image_size` yalnızca `model_path` ayarlandığında uygulanır.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb)

---

## Ortak Parametreler

Tüm modeller `AutoDetectionModel.from_pretrained()` içinde şu parametreleri kabul eder:

| Parametre | Tip | Açıklama |
| ----------- | ------ | ------------- |
| `model_type` | str | Framework adı (yukarıdaki bölümlere bakın) |
| `model_path` | str | Ağırlık dosyasının yolu veya model adı |
| `config_path` | str | Konfigürasyon dosyası yolu (MMDetection, Detectron2) |
| `confidence_threshold` | float | Bir tespiti tutmak için minimum skor (varsayılan: 0.25) |
| `device` | str | `"cpu"`, `"cuda:0"`, `"mps"` vb. |
| `category_mapping` | dict | Kategori ID'lerini isimlere haritalar: `{0: "car", 1: "person"}` |
| `category_remapping` | dict | Inference sonrasında kategori isimlerini yeniden haritalar |
| `image_size` | int | Model girdi çözünürlüğünü geçersiz kılar |
| `load_at_init` | bool | Ağırlıkları hemen yükler (varsayılan: True) |

## Önceden Yüklenmiş Model Kullanma

Zaten bir model örneğiniz (instance) varsa, bir yol yerine doğrudan onu geçin:

```python
from ultralytics import YOLO

yolo_model = YOLO("yolo26n.pt")
# ... customize the model ...

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model=yolo_model,
    confidence_threshold=0.25,
    device="cuda:0",
)
```

## Sonraki Adımlar

- [Sliced Inference Nasıl Çalışır](sliced-inference.md): Algoritmayı anlayın
- [Prediction Araçları](../predict.md): Gelişmiş prediction seçenekleri
- [Etkileşimli Notebook'lar](../notebooks.md): Her framework için uygulamalı örnekler
