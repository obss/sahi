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

# 快速开始

SAHI（Slicing Aided Hyper Inference，切片辅助高效推理）通过将大图像切片为重叠的网格块，在每个块上运行检测器，然后合并结果，来检测大图像中的小目标。它适用于任何检测模型，无需重新训练。

<div align="center">
  <img width="700" alt="sliced inference" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif">
</div>

## 安装

[![PyPI - Version](https://img.shields.io/pypi/v/sahi?logo=pypi&logoColor=white)](https://pypi.org/project/sahi/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/sahi?logo=condaforge)](https://anaconda.org/conda-forge/sahi)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sahi?logo=python&logoColor=gold)](https://pypi.org/project/sahi/)

```bash
pip install sahi
```

进行目标检测还需要安装一个检测框架。最常用的选择是 Ultralytics：

```bash
pip install ultralytics
```

??? note "其他安装方式"

    **Conda：**

    [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/sahi.svg)](https://anaconda.org/conda-forge/sahi)
    [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/sahi.svg)](https://anaconda.org/conda-forge/sahi)

    ```bash
    conda install -c conda-forge sahi
    ```

    !!! note
        如果你在 CUDA 环境下安装，建议在同一条命令中安装 `ultralytics`、`pytorch` 和 `pytorch-cuda`：
        ```bash
        conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
        ```

    **从源码安装：**
    ```bash
    pip install git+https://github.com/obss/sahi.git@main
    ```

    **开发模式（可编辑）：**
    ```bash
    git clone https://github.com/obss/sahi
    cd sahi
    pip install -e .
    ```

完整依赖列表请参见
[pyproject.toml](https://github.com/obss/sahi/blob/main/pyproject.toml)。

## 使用 Python 的切片预测

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 加载模型（适用于任何支持的框架）
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cuda:0",  # 或 "cpu"
)

# 执行切片预测
result = get_sliced_prediction(
    "path/to/your/image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)

# 导出可视化结果
result.export_visuals(export_dir="demo_data/")

# 访问各个预测结果
for pred in result.object_prediction_list:
    print(pred.category.name, pred.score.value, pred.bbox.to_xyxy())
```

## 使用命令行进行预测

无需编写 Python 代码即可执行切片推理：

```bash
sahi predict \
  --model_path yolo26n.pt \
  --model_type ultralytics \
  --source /path/to/images/ \
  --slice_height 512 \
  --slice_width 512
```

结果默认保存到 `runs/predict/exp`。

## 选择后处理后端

切片后，SAHI 使用 NMS 或 NMM 合并重叠的预测结果。系统会自动选择最佳可用后端：

| 后端 | 选择条件 | 安装方式 |
|------|---------|---------|
| **torchvision** | CUDA GPU + torchvision 可用 | `pip install torch torchvision` |
| **numba** | 已安装 numba，无 GPU | `pip install numba` |
| **numpy** | 始终可用（兜底方案） | -- |

手动指定后端：

```python
from sahi.postprocess.backends import set_postprocess_backend

set_postprocess_backend("numpy")       # 始终可用
set_postprocess_backend("numba")       # JIT 编译加速
set_postprocess_backend("torchvision") # GPU 加速
set_postprocess_backend("auto")        # 恢复自动检测
```

## 下一步

- [切片推理工作原理](../guides/sliced-inference.md) -- 了解算法、调优技巧和使用场景
- [模型集成](../guides/models.md) -- 将 SAHI 与 Ultralytics、HuggingFace、MMDetection、TorchVision、Detectron2 等配合使用
- [预测工具](predict.md) -- 批量推理、进度跟踪、可视化选项
- [COCO 工具](coco.md) -- 创建、切片、合并和转换 COCO 数据集
- [CLI 命令](cli.md) -- 完整 CLI 参考
- [交互式 Notebooks](../notebooks.md) -- 每个框架的动手实践 Colab notebooks
