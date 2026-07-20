---
tags:
  - notebooks
  - demos
  - interactive
  - colab
---

# 交互式 Notebooks

这些 Jupyter notebooks 展示了如何将 SAHI 与不同的检测框架配合使用。每个 notebook
都可以直接在 Google Colab 中运行，也可以从 GitHub 上的
[demo 目录](https://github.com/obss/sahi/tree/main/demo)克隆到本地。

## 推理 Notebooks

| Notebook | 框架 | 模型 | 链接 |
| ---------- | ------ | ------ | ------ |
| **Ultralytics** | ultralytics | YOLOv8、YOLO11、YOLO26 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) |
| **YOLOE** | ultralytics | YOLOE 变体 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb) |
| **YOLOv5** | yolov5 | YOLOv5 变体 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb) |
| **HuggingFace** | huggingface | DETR、Deformable DETR、DETA | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb) |
| **GroundingDINO** | huggingface | GroundingDINO 零样本检测 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_groundingdino.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_groundingdino.ipynb) |
| **RT-DETR** | rtdetr | RT-DETR 变体 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb) |
| **MMDetection** | mmdet | 300 多个检测模型 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb) |
| **Detectron2** | detectron2 | Detectron2 模型 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb) |
| **TorchVision** | torchvision | Faster R-CNN、RetinaNet、FCOS、SSD | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb) |
| **Roboflow** | roboflow | RF-DETR | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb) |

## 工具 Notebooks

| Notebook | 说明 | 链接 |
| ---------- | ------ | ------ |
| **切片** | 图像和 COCO 数据集切片操作 | [![在 Colab 中打开](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/slicing.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/slicing.ipynb) |

## 在本地运行

克隆仓库并使用 Jupyter 运行 notebooks：

```bash
git clone https://github.com/obss/sahi.git
cd sahi
pip install -e ".[dev]"
jupyter notebook demo/
```
