---
tags:
  - postprocessing
  - nms
  - nmm
  - gpu
  - api-reference
---

# Postprocessing Backend'leri

SAHI'nin postprocessing (NMS, NMM) işlemleri üç değiştirilebilir backend üzerinde çalışabilir. Doğru backend tercihi donanımınıza ve yüklü paketlerinize bağlıdır.

## Backend genel bakış

| Backend         | En iyi kullanım alanı                                                        | Ek bağımlılık                   |
| --------------- | ---------------------------------------------------------------------------- | ------------------------------- |
| **numpy**       | Yalnızca CPU ortamları, küçük/orta prediction sayıları                       | Yok (her zaman mevcut)          |
| **numba**       | Yüksek prediction sayılı CPU; ilk çağrıda ~1 sn JIT ısınması, ardından hızlı | `pip install numba`             |
| **torchvision** | CUDA veya Apple MPS GPU mevcut olduğunda; büyük batch'ler için en hızlısı    | `pip install torch torchvision` |

## Otomatik Algılama (varsayılan)

SAHI varsayılan olarak çalışma anında (runtime) mevcut en iyi backend'i otomatik olarak seçer:

1. **torchvision**: `torchvision` yüklüyse _ve_ bir GPU mevcutsa
   (CUDA veya Apple Silicon üzerinde Apple MPS).
2. **numba**: `numba` paketi yüklüyse.
3. **numpy**: son çare (fallback) olarak her zaman mevcuttur.

```python
from sahi.postprocess.backends import get_postprocess_backend

# Check which backend was resolved (triggers auto-detection)
print(get_postprocess_backend())  # "auto" until first postprocessing call
```

## Belirli Bir Backend'e Zorlama

Bir backend'i sabitlemek için inference çalıştırmadan önce `set_postprocess_backend` kullanın:

```python
from sahi.postprocess.backends import set_postprocess_backend

# Force pure-numpy (no extra deps, works everywhere)
set_postprocess_backend("numpy")

# Force numba JIT (install with: pip install numba)
set_postprocess_backend("numba")

# Force torchvision GPU (install with: pip install torch torchvision)
set_postprocess_backend("torchvision")

# Restore auto-detection
set_postprocess_backend("auto")
```

Bu çağrı, `get_sliced_prediction` tarafından dahili olarak tetiklenenler de dahil olmak üzere mevcut süreçteki sonraki tüm NMS/NMM operasyonlarını etkiler.

### Örnek: Tam bir inference çalışması için backend sabitleme

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.postprocess.backends import set_postprocess_backend

# CUDA veya Apple Silicon makinelerde GPU hızlandırmalı son işlemeyi kullan
set_postprocess_backend("torchvision")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cuda:0",
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

## Postprocessing Fonksiyonlarını Doğrudan Kullanma

Her üç backend de aynı dizi (array) kuralını paylaşır: `[x1, y1, x2, y2, score, category_id]` sütunlarına sahip bir `(N, 6)` numpy dizisi.

### NMS (baskılama)

```python
import numpy as np
from sahi.postprocess.combine import nms, batched_nms

predictions = np.array([
    [100, 100, 200, 200, 0.95, 0],
    [105, 105, 205, 205, 0.80, 0],
    [300, 300, 400, 400, 0.90, 1],
])

# Global NMS, tüm kategoriler birlikte yarışır
keep = nms(predictions, match_metric="IOU", match_threshold=0.5)
print(predictions[keep])

# Kategori bazında NMS, sınıf 0 ve sınıf 1 bağımsız değerlendirilir
keep = batched_nms(predictions, match_metric="IOU", match_threshold=0.5)
print(predictions[keep])
```

### NMM (birleştirme)

NMM, örtüşen kutuları elenmek yerine birleştirir:

```python
from sahi.postprocess.combine import greedy_nmm, nmm, batched_greedy_nmm

# Greedy NMM: each kept box merges only its direct neighbours (fast, tight boxes)
keep_to_merge = greedy_nmm(predictions, match_metric="IOU", match_threshold=0.5)
# {kept_index: [merged_index, ...], ...}

# Full NMM: transitive merging (A merges B, B merges C → A gets all three)
keep_to_merge = nmm(predictions, match_metric="IOU", match_threshold=0.5)

# Per-category greedy NMM
keep_to_merge = batched_greedy_nmm(predictions, match_threshold=0.5)
```

### IoS metriği

Hem NMS hem de NMM, bir kutu diğerinden çok daha küçük olduğunda faydalı olan `match_metric="IOS"` (Intersection over Smaller area) parametresini destekler:

```python
keep = nms(predictions, match_metric="IOS", match_threshold=0.5)
```

## Postprocess Sınıfları

Yüksek seviyeli sınıflar, SAHI'nin `ObjectPrediction` listeleriyle entegre olur ve `postprocess_type` argümanı aracılığıyla `get_sliced_prediction` tarafından kullanılır:

```python
from sahi.postprocess.combine import NMSPostprocess, NMMPostprocess, GreedyNMMPostprocess

# NMS, en iyi kutuyu tutar, geri kalanını atar
postprocessor = NMSPostprocess(
    match_threshold=0.5,
    match_metric="IOU",
    class_agnostic=True,   # False → kategori bazında
)
filtered = postprocessor(object_prediction_list)

# Greedy NMM, örtüşen kutuları birleştirir (hızlı)
postprocessor = GreedyNMMPostprocess(match_threshold=0.5)
merged = postprocessor(object_prediction_list)

# Full NMM, geçişli birleştirme
postprocessor = NMMPostprocess(match_threshold=0.5)
merged = postprocessor(object_prediction_list)
```

`class_agnostic=False` parametresinin geçilmesi her postprocessor'ın kategori başına bağımsız olarak çalışmasını sağlar, böylece bir "car" tahmini asla bir "person" tahminini bastırmaz.

## API referansı

::: sahi.postprocess.backends

::: sahi.postprocess.combine
