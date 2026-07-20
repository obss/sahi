---
tags:
  - postprocessing
  - nms
  - nmm
  - gpu
  - api-reference
---

# 后处理后端

SAHI 的后处理操作（NMS、NMM）可以在三种可互换的后端上运行。合适的后端取决于
您的硬件和已安装的软件包。

## 后端概览

| 后端            | 适用场景                                                                | 额外依赖                               |
| --------------- | ----------------------------------------------------------------------- | -------------------------------------- |
| **numpy**       | 仅使用 CPU 的环境，预测结果数量较少或中等                               | 无（始终可用）                         |
| **numba**       | 使用 CPU 且预测结果数量较多；首次调用需要约 1 秒进行 JIT 预热，之后较快 | `pip install numba`                    |
| **torchvision** | CUDA GPU 可用；处理大型批次时速度最快                                   | `pip install torch torchvision` + CUDA |

## 自动检测（默认）

默认情况下，SAHI 会在运行时自动选择最佳可用后端：

1. **torchvision** -- 已安装 `torchvision` 且存在 CUDA GPU 时使用。
2. **numba** -- 已安装 `numba` 软件包时使用。
3. **numpy** -- 始终可用，作为最终兜底方案。

```python
from sahi.postprocess.backends import get_postprocess_backend

# 检查解析出的后端（会触发自动检测）
print(get_postprocess_backend())  # 首次执行后处理操作前为 "auto"
```

## 强制指定后端

在运行推理前使用 `set_postprocess_backend` 固定后端：

```python
from sahi.postprocess.backends import set_postprocess_backend

# 强制使用纯 numpy（无额外依赖，可在所有环境中运行）
set_postprocess_backend("numpy")

# 强制使用 numba JIT（安装命令：pip install numba）
set_postprocess_backend("numba")

# 强制使用 torchvision GPU（安装命令：pip install torch torchvision）
set_postprocess_backend("torchvision")

# 恢复自动检测
set_postprocess_backend("auto")
```

该调用会影响当前进程中后续的所有 NMS/NMM 操作，包括
`get_sliced_prediction` 内部触发的操作。

### 示例：为完整推理流程固定后端

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.postprocess.backends import set_postprocess_backend

# 在 CUDA 环境中使用 GPU 加速后处理
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

## 直接使用后处理函数

三个后端遵循相同的数组约定：使用形状为 `(N, 6)` 的 numpy 数组，列顺序为
`[x1, y1, x2, y2, score, category_id]`。

### NMS（抑制）

```python
import numpy as np
from sahi.postprocess.combine import nms, batched_nms

predictions = np.array([
    [100, 100, 200, 200, 0.95, 0],
    [105, 105, 205, 205, 0.80, 0],
    [300, 300, 400, 400, 0.90, 1],
])

# 全局 NMS：所有类别一起参与比较
keep = nms(predictions, match_metric="IOU", match_threshold=0.5)
print(predictions[keep])

# 按类别执行 NMS：类别 0 和类别 1 分别独立处理
keep = batched_nms(predictions, match_metric="IOU", match_threshold=0.5)
print(predictions[keep])
```

### NMM（合并）

NMM 不会丢弃重叠框，而是将其合并：

```python
from sahi.postprocess.combine import greedy_nmm, nmm, batched_greedy_nmm

# 贪心 NMM：每个保留框只合并与其直接相邻的框（速度快，边界框紧凑）
keep_to_merge = greedy_nmm(predictions, match_metric="IOU", match_threshold=0.5)
# {kept_index: [merged_index, ...], ...}

# 完整 NMM：传递式合并（A 合并 B，B 合并 C，则 A 会获得全部三个框）
keep_to_merge = nmm(predictions, match_metric="IOU", match_threshold=0.5)

# 按类别执行贪心 NMM
keep_to_merge = batched_greedy_nmm(predictions, match_threshold=0.5)
```

### IoS 指标

NMS 和 NMM 都支持 `match_metric="IOS"`（Intersection over Smaller area，
交集与较小面积之比）。当一个边界框远小于另一个边界框时，该指标非常实用：

```python
keep = nms(predictions, match_metric="IOS", match_threshold=0.5)
```

## 后处理类

高级类可以与 SAHI 的 `ObjectPrediction` 列表集成，并由
`get_sliced_prediction` 通过 `postprocess_type` 参数使用：

```python
from sahi.postprocess.combine import NMSPostprocess, NMMPostprocess, GreedyNMMPostprocess

# NMS：保留最佳边界框，丢弃其余边界框
postprocessor = NMSPostprocess(
    match_threshold=0.5,
    match_metric="IOU",
    class_agnostic=True,   # False 表示按类别处理
)
filtered = postprocessor(object_prediction_list)

# 贪心 NMM：合并重叠边界框（速度快）
postprocessor = GreedyNMMPostprocess(match_threshold=0.5)
merged = postprocessor(object_prediction_list)

# 完整 NMM：传递式合并
postprocessor = NMMPostprocess(match_threshold=0.5)
merged = postprocessor(object_prediction_list)
```

传入 `class_agnostic=False` 后，每个后处理器都会按类别独立运行，因此 `"car"`
预测结果不会抑制 `"person"` 预测结果。

## API 参考

::: sahi.postprocess.backends

::: sahi.postprocess.combine
