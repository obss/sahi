---
tags:
  - postprocessing
  - nms
  - nmm
  - gpu
  - api-reference
---

# Postprocessing Backends

SAHI's postprocessing (NMS, NMM) can run on three interchangeable backends. The
right backend depends on your hardware and installed packages.

## Backend overview

| Backend         | Best for                                                                   | Extra dependency                       |
| --------------- | -------------------------------------------------------------------------- | -------------------------------------- |
| **numpy**       | CPU-only environments, small/medium prediction counts                      | None (always available)                |
| **numba**       | CPU with large prediction counts; ~1 s JIT warmup on first call, then fast | `pip install numba`                    |
| **torchvision** | CUDA GPU available; fastest for large batches                              | `pip install torch torchvision` + CUDA |

## Auto-detection (default)

By default SAHI automatically picks the best available backend at runtime:

1. **torchvision** — if `torchvision` is installed _and_ a CUDA GPU is present.
2. **numba** — if the `numba` package is installed.
3. **numpy** — always available as the final fallback.

```python
from sahi.postprocess.backends import get_postprocess_backend

# Check which backend was resolved (triggers auto-detection)
print(get_postprocess_backend())  # "auto" until first postprocessing call
```

## Forcing a specific backend

Use `set_postprocess_backend` before running inference to pin a backend:

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

This call affects all subsequent NMS/NMM operations in the current process,
including those triggered internally by `get_sliced_prediction`.

### Example: pinning the backend for a full inference run

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.postprocess.backends import set_postprocess_backend

# Use GPU-accelerated postprocessing when running on a CUDA machine
set_postprocess_backend("torchvision")

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11n.pt",
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

## Using postprocessing functions directly

All three backends share the same array convention: an `(N, 6)` numpy array with
columns `[x1, y1, x2, y2, score, category_id]`.

### NMS (suppression)

```python
import numpy as np
from sahi.postprocess.combine import nms, batched_nms

predictions = np.array([
    [100, 100, 200, 200, 0.95, 0],
    [105, 105, 205, 205, 0.80, 0],
    [300, 300, 400, 400, 0.90, 1],
])

# Global NMS — all categories compete together
keep = nms(predictions, match_metric="IOU", match_threshold=0.5)
print(predictions[keep])

# Per-category NMS — class 0 and class 1 are treated independently
keep = batched_nms(predictions, match_metric="IOU", match_threshold=0.5)
print(predictions[keep])
```

### NMM (merging)

Instead of discarding overlapping boxes, NMM merges them:

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

### IoS metric

Both NMS and NMM support `match_metric="IOS"` (Intersection over Smaller area),
which is useful when one box is much smaller than another:

```python
keep = nms(predictions, match_metric="IOS", match_threshold=0.5)
```

## Postprocess classes

High-level classes integrate with SAHI's `ObjectPrediction` lists and are used
by `get_sliced_prediction` via the `postprocess_type` argument:

```python
from sahi.postprocess.combine import NMSPostprocess, NMMPostprocess, GreedyNMMPostprocess

# NMS — keep the best box, discard the rest
postprocessor = NMSPostprocess(
    match_threshold=0.5,
    match_metric="IOU",
    class_agnostic=True,   # False → per-category
)
filtered = postprocessor(object_prediction_list)

# Greedy NMM — merge overlapping boxes (fast)
postprocessor = GreedyNMMPostprocess(match_threshold=0.5)
merged = postprocessor(object_prediction_list)

# Full NMM — transitive merging
postprocessor = NMMPostprocess(match_threshold=0.5)
merged = postprocessor(object_prediction_list)
```

Passing `class_agnostic=False` makes each postprocessor run independently per
category, so a "car" prediction will never suppress a "person" prediction.

## API reference

::: sahi.postprocess.backends

::: sahi.postprocess.combine
