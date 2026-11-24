---
hide:
  - navigation
tags:
  - getting-started
  - installation
  - inference
  - slicing
---

# Hızlı Başlangıç

SAHI büyük bir görseli örtüşen dilimlere ayırır, dedektörünüzü her dilimde çalıştırır ve tespitleri tam görsel üzerinde yeniden birleştirir. Küçük nesneler tespit edilebilecek kadar büyük kalır ve yeniden eğitim gerekmez.

## 1. Kurulum

```bash
pip install "sahi[ultralytics]"
```

`sahi` tek başına bir dedektörle gelmez, bu yüzden istediğiniz framework için ilgili ekstrayı seçin: `ultralytics`, `transformers`, `yolov5`, `roboflow`, `torchvision`, `torch`, `onnx`, `numba` veya `all`. Conda kullanıcıları `conda install -c conda-forge sahi` çalıştırıp framework'ü ayrıca kurabilir.

## 2. Prediction alın

Bu örnek CPU üzerinde baştan sona çalışır. Örnek bir görsel indirir ve Ultralytics ilk kullanımda `yolo26n.pt` dosyasını indirir.

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.file import download_from_url

download_from_url(
    "https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg",
    "demo_data/small-vehicles1.jpeg",
)

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cpu",  # or "cuda:0"
)

result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

Başka bir framework için `model_type` ve `model_path` değerlerini değiştirin. Tam liste için [Model Entegrasyonları](guides/models.md) sayfasına bakın.

## 3. Sonucu okuyun

`result` bir `PredictionResult` nesnesidir. Her tespit `result.object_prediction_list` içinde bulunur.

```python
for pred in result.object_prediction_list:
    print(pred.category.name, pred.score.value, pred.bbox.to_xyxy())

# Writes demo_data/prediction_visual.png
result.export_visuals(export_dir="demo_data/")

# COCO-format dicts, ready to dump as JSON
coco_predictions = result.to_coco_predictions(image_id=1)
```

Görseliniz zaten model girdi boyutuna yakınsa, bunun yerine `get_prediction` kullanın ve dilimlemeyi tamamen atlayın.

## 4. Aynı işlem CLI ile

```bash
sahi predict --model_type ultralytics --model_path yolo26n.pt --source demo_data/ --slice_height 512 --slice_width 512
```

Görseller `runs/predict/exp` dizinine yazılır. Değerlendirme için ayrıca bir COCO `result.json` dosyası dışa aktarmak isterseniz `--dataset_json_path dataset.json` ekleyin.

## Sonraki Adımlar

- [Sliced Inference Nasıl Çalışır](guides/sliced-inference.md): dilim boyutu, örtüşme ve birleştirme stratejisini seçmek için.
- [Model Entegrasyonları](guides/models.md): HuggingFace, MMDetection, Detectron2, TorchVision, RT-DETR, RF-DETR ve diğerleri için.
- [Prediction Araçları](predict.md): toplu (batch) inference, ilerleme çubukları ve dışa aktarma seçenekleri için.
- [CLI Komutları](cli.md): tüm komutlar ve parametreler için.
- [Etkileşimli Notebook'lar](notebooks.md): çalıştırılabilir Colab örnekleri için.
