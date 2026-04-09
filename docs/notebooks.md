---
tags:
  - notebooks
  - demos
  - interactive
  - colab
---

# Interactive Notebooks

Hands-on Jupyter notebooks demonstrating SAHI with different detection
frameworks. Each notebook can be run directly in Google Colab or cloned from the
[demo directory](https://github.com/obss/sahi/tree/main/demo) on GitHub.

## Inference Notebooks

| Notebook | Framework | Models | Links |
|----------|-----------|--------|-------|
| **Ultralytics** | ultralytics | YOLOv8, YOLO11, YOLO26 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) |
| **YOLOE** | ultralytics | YOLOE variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb) |
| **YOLOv5** | yolov5 | YOLOv5 variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb) |
| **HuggingFace** | huggingface | DETR, Deformable DETR, DETA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb) |
| **RT-DETR** | rtdetr | RT-DETR variants | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb) |
| **MMDetection** | mmdet | 300+ detection models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb) |
| **Detectron2** | detectron2 | Detectron2 models | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb) |
| **TorchVision** | torchvision | Faster R-CNN, RetinaNet, FCOS, SSD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb) |
| **Roboflow** | roboflow | RF-DETR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb) |

## Utility Notebooks

| Notebook | Description | Links |
|----------|-------------|-------|
| **Slicing** | Image and COCO dataset slicing operations | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/slicing.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/slicing.ipynb) |

## Running Locally

Clone the repository and run notebooks with Jupyter:

```bash
git clone https://github.com/obss/sahi.git
cd sahi
pip install -e ".[dev]"
jupyter notebook demo/
```
