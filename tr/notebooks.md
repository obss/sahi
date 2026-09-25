---
hide:
  - navigation
tags:
  - notebooks
  - demos
  - interactive
  - colab
---

# Etkileşimli Notebook'lar

SAHI'yi farklı tespit framework'leri ile gösteren uygulamalı Jupyter notebook'ları. Her notebook doğrudan Google Colab üzerinde çalıştırılabilir veya GitHub üzerindeki [demo dizininden](https://github.com/obss/sahi/tree/main/demo) klonlanabilir.

## Inference Notebook'ları

| Notebook | Framework | Modeller | Bağlantılar |
| ---------- | ----------- | -------- | ------- |
| **Ultralytics** | ultralytics | YOLOv8, YOLO11, YOLO26 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) |
| **YOLOE** | ultralytics | YOLOE varyantları | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb) |
| **YOLOv5** | yolov5 | YOLOv5 varyantları | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb) |
| **HuggingFace** | huggingface | DETR, Deformable DETR, DETA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb) |
| **GroundingDINO** | huggingface | GroundingDINO zero-shot detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_groundingdino.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_groundingdino.ipynb) |
| **RT-DETR** | rtdetr | RT-DETR varyantları | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb) |
| **MMDetection** | mmdet | 300+ detection modeli | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb) |
| **Detectron2** | detectron2 | Detectron2 modelleri | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb) |
| **TorchVision** | torchvision | Faster R-CNN, RetinaNet, FCOS, SSD | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb) |
| **Roboflow** | roboflow | RF-DETR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb) |

## Yardımcı Araç Notebook'ları

| Notebook | Açıklama | Bağlantılar |
| ---------- | ------------- | ------- |
| **Slicing** | Görsel ve COCO veri kümesi dilimleme operasyonları | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/slicing.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/slicing.ipynb) |
| **Toplu (Batch) Dilimlenmiş Inference** | Dilimlenmiş inference için batch boyutu, hız ve TensorRT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_with_batch_slicing.ipynb) [![GitHub](https://img.shields.io/badge/GitHub-source-black?logo=github)](https://github.com/obss/sahi/blob/main/demo/inference_with_batch_slicing.ipynb) |

## Yerel Olarak Çalıştırma

Depoyu (repository) klonlayın ve notebook'ları Jupyter ile çalıştırın:

```bash
git clone https://github.com/obss/sahi.git
cd sahi
pip install -e ".[dev]"
jupyter notebook demo/
```
