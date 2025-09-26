---
hide:
  - navigation
---

# 快速开始

欢迎来到SAHI！本指南将帮助你快速上手库的核心功能，包括安装、进行预测以及使用命令行工具。

# 1. 安装

```
# 安装 SAHI

通过 pip 安装 SAHI。如果你需要做目标检测，建议同时安装’ultralytics‘。
```
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sahi?logo=python&logoColor=gold)

## 1.1 Pip 安装（推荐）

安装或更新 `sahi`：

```bash
pip install sahi
```

更多详情请访问 [Python Package Index (PyPI)](https://pypi.org/project/sahi/)。  

[![PyPI - Version](https://img.shields.io/pypi/v/sahi?logo=pypi&logoColor=white)](https://pypi.org/project/sahi/) [![Downloads](https://static.pepy.tech/badge/sahi)](https://www.pepy.tech/projects/sahi)

你也可以直接从 GitHub 安装最新开发版本（确保已安装 Git 命令行工具）：

```bash
pip install git+https://github.com/obss/sahi.git@main
```

## 1.2 Conda 安装

Conda 也可以作为 pip 的替代工具：

```bash
conda install -c conda-forge sahi
```

更多详情请访问 [Anaconda](https://anaconda.org/conda-forge/sahi)。  | 更新 Conda 包的 feedstock 仓库：[GitHub](https://github.com/conda-forge/sahi-feedstock/)。

> ⚠️ 注意  
> 如果你在 CUDA 环境下安装，建议同时安装 `ultralytics`、`pytorch` 和 `pytorch-cuda`，这样 Conda 可以自动解决依赖冲突。或者在必要时最后安装 `pytorch-cuda` 来覆盖 CPU 版本的 `pytorch`：
>
> ```bash
> conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
> ```

## 1.3 Git 克隆（开发或尝试最新版）

如果你想参与开发或尝试最新源码：

```bash
git clone https://github.com/obss/sahi
cd sahi
pip install -e .
```

<p align="left">
  <a href="https://github.com/obss/sahi">
    <img src="https://img.shields.io/github/last-commit/obss/sahi?logo=github" alt="GitHub last commit">
  </a>
  <a href="https://github.com/obss/sahi/commits/main">
    <img src="https://img.shields.io/github/commit-activity/t/obss/sahi" alt="GitHub commit activity">
  </a>
  <a href="https://github.com/obss/sahi/graphs/contributors">
    <img src="https://img.shields.io/github/contributors/obss/sahi" alt="GitHub contributors">
  </a>
</p>

更多依赖信息请查看 [pyproject.toml](https://github.com/obss/sahi/blob/main/pyproject.toml)。

# 2. 使用 Python 的切片预测

切片推理（Sliced Inference）是 SAHI 的核心功能，可以在大图像中检测小目标。以下是使用 Python API 的简单示例：

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# 初始化 YOLOv8 模型
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n.pt', # 或其他 YOLOv8 模型
    confidence_threshold=0.25,
    device="cuda:0", # 或 "cpu"
)

# 执行切片预测
result = get_sliced_prediction(
    "path/to/your/image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# 导出可视化结果
result.export_visuals(export_dir="demo_data/")

# 获取预测对象列表
predictions = result.object_prediction_list
```
# 3. 使用命令行的切片预测

SAHI 还提供了功能强大的命令行工具，可以无需编写任何 Python 代码即可快速进行预测。

```bash
sahi predict --model_path yolov8n.pt --model_type yolov8 --source /path/to/images/ --slice_height 512 --slice_width 512
```
该命令会对指定目录下的所有图片进行切片推理，并保存预测结果。

# 下一步

你现在已经了解了 SAHI 的基础用法！想要深入学习，可以参考以下资源：

* **预测深入**：更多高级预测选项，请参见[预测工具指南](predict.md)。
* **演示示例**：在 [demo目录](../../demo/) 中查看交互式 notebook，实践不同模型的使用。
* **COCO 工具**： 学习如何创建、操作和转换数据集，请参见[COCO 工具指南](coco.md)。
* **所有 CLI 命令**： 完整命令列表请参见[CLI 文档](cli.md)。