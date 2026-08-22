---
hide:
  - navigation
tags:
  - getting-started
  - installation
  - inference
  - slicing
  - postprocessing
---

# Hızlı Başlangıç

SAHI (Slicing Aided Hyper Inference), büyük görselleri örtüşen dilimlere (tiles) ayırarak, her dilimde dedektörünüzü çalıştırıp sonuçları birleştirerek büyük görsellerdeki küçük nesneleri tespit eder. Yeniden eğitim (retraining) gerektirmeden tüm tespit modelleriyle çalışır.

<div align="center">
  <img width="700" alt="sliced inference" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif">
</div>

## Kurulum

[![PyPI - Version](https://img.shields.io/pypi/v/sahi?logo=pypi&logoColor=white)](https://pypi.org/project/sahi/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sahi?logo=condaforge)](https://anaconda.org/conda-forge/sahi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sahi?logo=python&logoColor=gold)](https://pypi.org/project/sahi/)

```bash
pip install sahi
```

Object detection için ayrıca bir framework'e ihtiyacınız olacaktır. En yaygın tercih Ultralytics'tir:

```bash
pip install ultralytics
```

??? note "Diğer kurulum yöntemleri"

    **Conda:**

    [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/sahi.svg)](https://anaconda.org/conda-forge/sahi)
    [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/sahi.svg)](https://anaconda.org/conda-forge/sahi)

    ```bash
    conda install -c conda-forge sahi
    ```

    !!! note
        CUDA ortamında kurulum yapıyorsanız, `ultralytics`, `pytorch` ve `pytorch-cuda` paketlerini aynı komutta yüklemeniz tavsiye edilir:
        ```bash
        conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
        ```

    **Kaynak koddan:**
    ```bash
    pip install git+https://github.com/obss/sahi.git@main
    ```

    **Geliştirme (düzenlenebilir):**
    ```bash
    git clone https://github.com/obss/sahi
    cd sahi
    pip install -e .
    ```

Bağımlılıkların tam listesi için [pyproject.toml](https://github.com/obss/sahi/blob/main/pyproject.toml) dosyasına bakabilirsiniz.

## Python ile Sliced Prediction

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Load a model (works with any supported framework)
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cuda:0",  # or "cpu"
)

# Run sliced prediction
result = get_sliced_prediction(
    "path/to/your/image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# Export visualizations
result.export_visuals(export_dir="demo_data/")

# Access individual predictions
for pred in result.object_prediction_list:
    print(pred.category.name, pred.score.value, pred.bbox.to_xyxy())
```

## CLI ile Prediction

Python kodu yazmadan Sliced Inference çalıştırın:

```bash
sahi predict \
  --model_path yolo26n.pt \
  --model_type ultralytics \
  --source /path/to/images/ \
  --slice_height 512 \
  --slice_width 512
```

Sonuçlar varsayılan olarak `runs/predict/exp` dizinine kaydedilir.

## Postprocessing Backend Seçimi

Dilimleme sonrasında SAHI, örtüşen prediction'ları NMS veya NMM ile birleştirir. Mevcut en iyi backend otomatik olarak seçilir:

| Backend | Ne zaman seçilir | Kurulum |
| --------- | -------------- | --------- |
| **torchvision** | CUDA veya Apple MPS GPU + torchvision mevcut olduğunda | `pip install torch torchvision` |
| **numba** | numba yüklü, GPU yok | `pip install numba` |
| **numpy** | Her zaman mevcut (fallback) | Gerekmez |

Seçimi manuel olarak geçersiz kılma:

```python
from sahi.postprocess.backends import set_postprocess_backend

set_postprocess_backend("numpy")       # always available
set_postprocess_backend("numba")       # JIT-compiled
set_postprocess_backend("torchvision") # GPU-accelerated
set_postprocess_backend("auto")        # restore auto-detection
```

## Sonraki Adımlar

- [Sliced Inference Nasıl Çalışır](guides/sliced-inference.md): Algoritmayı, parametre ipuçlarını ve ne zaman kullanılacağını anlayın
- [Model Entegrasyonları](guides/models.md): SAHI'yi Ultralytics, HuggingFace, MMDetection, TorchVision, Detectron2 ve daha fazlası ile kullanın
- [Prediction Araçları](predict.md): Toplu (batch) inference, ilerleme takibi, görselleştirme seçenekleri
- [COCO Araçları](coco.md): COCO veri kümelerini oluşturun, dilimleyin, birleştirin ve dönüştürün
- [CLI Komutları](cli.md): Tam CLI referansı
- [Etkileşimli Notebook'lar](notebooks.md): Tüm framework'ler için uygulamalı Colab notebook'ları
