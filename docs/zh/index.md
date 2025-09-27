---
hide:
  - navigation
  - toc
---


<div align="center">
<h1>
  SAHI: 切片辅助超推理
</h1>

<h4>
  用于大规模目标检测和实例分割的轻量级视觉库
</h4>

<h4>
    <img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif">
</h4>

<div>
    <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi" alt="下载量"></a>
    <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi/month" alt="月下载量"></a>
    <a href="https://github.com/obss/sahi/blob/main/LICENSE.md"><img src="https://img.shields.io/pypi/l/sahi" alt="许可证"></a>
    <a href="https://badge.fury.io/py/sahi"><img src="https://badge.fury.io/py/sahi.svg" alt="PyPI 版本"></a>
    <a href="https://anaconda.org/conda-forge/sahi"><img src="https://anaconda.org/conda-forge/sahi/badges/version.svg" alt="Conda 版本"></a>
    <a href="https://github.com/obss/sahi/actions/workflows/ci.yml"><img src="https://github.com/obss/sahi/actions/workflows/ci.yml/badge.svg" alt="持续集成"></a>
  <br>
    <a href="https://context7.com/obss/sahi"><img src="https://img.shields.io/badge/Context7%20MCP-Indexed-blue" alt="Context7 MCP"></a>
    <a href="https://context7.com/obss/sahi/llms.txt"><img src="https://img.shields.io/badge/llms.txt-✓-brightgreen" alt="llms.txt"></a>
    <a href="https://ieeexplore.ieee.org/document/9897990"><img src="https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg" alt="DOI"></a>
    <a href="https://arxiv.org/abs/2202.06934"><img src="https://img.shields.io/badge/arXiv-2202.06934-b31b1b.svg" alt="arXiv"></a>
    <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="在 Colab 打开"></a>
    <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>
</div>
</div>

## 什么是 SAHI？

SAHI（Slicing Aided Hyper Inference，切片辅助超推理）是一个提供了通用的切片辅助推理与微调流程，专门用于小目标检测的开源框架。
在监控等应用场景中，检测小目标或远处物体是一项重大挑战，因为它们仅由少量像素表示，缺乏足够的细节，传统检测器难以识别。

SAHI 通过一种独特的方法解决了这一问题，该方法可以与任意目标检测器结合使用，而无需额外的微调。  
在 Visdrone 和 xView 航拍目标检测数据集上的实验评估表明，SAHI 可以使 FCOS、VFNet 和 TOOD 检测器的 AP 分别提高 6.8%、5.1% 和 5.3%。在结合切片辅助微调后，精度可进一步提升，累计提升分别达到 12.7%、13.4% 和 14.5%。该技术已成功集成到 Detectron2、MMDetection 和 YOLOv5 等模型中。

<div class="grid cards" markdown>

- ⏱️ **快速开始**

    ***

    使用 pip 安装 `sahi`，几分钟即可上手。

    ***

    [➡️ 快速开始](quick-start.md)

- 🖼️ **预测**

    ***

    使用 SAHI 对新图片、视频和流进行预测。

    ***

    [➡️ 了解更多](predict.md)

- ✂️ **切片**

    ***

    学习如何对大图像和数据集进行切片以进行推理。

    ***

    [➡️ 了解更多](slicing.md)

- 🗂️ **COCO 工具**

    ***

    处理 COCO 格式数据集，包括创建、拆分和过滤。

    ***

    [➡️ 了解更多](coco.md)

- 💻 **命令行工具**

    ***

    通过命令行使用 SAHI 进行预测和数据集操作。

    ***

    [➡️ 了解更多](cli.md)

</div>

## 交互式示例

所有文档文件都配有 [demo 目录](../notebooks/) 中的交互式 Jupyter notebook：

<div class="grid cards" markdown>

- 📓 **切片**

    ***

    切片操作演示。

    ***

    [➡️ 打开 Notebook](../notebooks/slicing.ipynb/)

- 📓 **Ultralytics**

    ***

    YOLOv8/YOLO11/YOLO12 集成。

    ***

    [➡️ 打开 Notebook](../notebooks/inference_for_ultralytics.ipynb)

- 📓 **YOLOv5**

    ***

    YOLOv5 集成。

    ***

    [➡️ 打开 Notebook](../notebooks/inference_for_yolov5.ipynb)

- 📓 **MMDetection**

    ***

    MMDetection 集成。

    ***

    [➡️ 打开 Notebook](../notebooks/inference_for_mmdetection.ipynb)

- 📓 **HuggingFace**

    ***

    HuggingFace 模型集成。

    ***

    [➡️ 打开 Notebook](../notebooks/inference_for_huggingface.ipynb)

- 📓 **TorchVision**

    ***

    TorchVision 模型集成。

    ***

    [➡️ 打开 Notebook](../notebooks/inference_for_torchvision.ipynb)

- 📓 **RT-DETR**

    ***

    RT-DETR 集成。

    ***

    [➡️ 打开 Notebook](../notebooks/inference_for_rtdetr.ipynb)

</div>
