---
hide:
  - navigation
  - toc
---


<div align="center">
<h1>
  SAHI: Slicing Aided Hyper Inference
</h1>

<h4>
  A lightweight vision library for performing large scale object detection & instance segmentation
</h4>

<h4>
    <img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif">
</h4>

<div>
    <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi" alt="downloads"></a>
    <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi/month" alt="downloads"></a>
    <a href="https://github.com/obss/sahi/blob/main/LICENSE.md"><img src="https://img.shields.io/pypi/l/sahi" alt="License"></a>
    <a href="https://badge.fury.io/py/sahi"><img src="https://badge.fury.io/py/sahi.svg" alt="pypi version"></a>
    <a href="https://anaconda.org/conda-forge/sahi"><img src="https://anaconda.org/conda-forge/sahi/badges/version.svg" alt="conda version"></a>
    <a href="https://github.com/obss/sahi/actions/workflows/ci.yml"><img src="https://github.com/obss/sahi/actions/workflows/ci.yml/badge.svg" alt="Continious Integration"></a>
  <br>
    <a href="https://context7.com/obss/sahi"><img src="https://img.shields.io/badge/Context7%20MCP-Indexed-blue" alt="Context7 MCP"></a>
    <a href="https://context7.com/obss/sahi/llms.txt"><img src="https://img.shields.io/badge/llms.txt-✓-brightgreen" alt="llms.txt"></a>
    <a href="https://ieeexplore.ieee.org/document/9897990"><img src="https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg" alt="ci"></a>
    <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>
</div>
</div>

Welcome to the SAHI documentation! This directory contains detailed guides and tutorials for using SAHI's various features. Below is an overview of each documentation file and what you'll find in it.

## Core Documentation Files

### [Prediction Utilities](predict.md)

- Detailed guide for performing object detection inference
- Standard and sliced inference examples
- Batch prediction usage
- Class exclusion during inference
- Visualization parameters and export formats
- Interactive examples with various model integrations (YOLO11, MMDetection, etc.)

### [Slicing Utilities](slicing.md)

- Guide for slicing large images and datasets
- Image slicing examples
- COCO dataset slicing examples
- Interactive demo notebook reference

### [COCO Utilities](coco.md)

- Comprehensive guide for working with COCO format datasets
- Dataset creation and manipulation
- Slicing COCO datasets
- Dataset splitting (train/val)
- Category filtering and updates
- Area-based filtering
- Dataset merging
- Format conversion (COCO ↔ YOLO)
- Dataset sampling utilities
- Statistics calculation
- Result validation

### [CLI Commands](cli.md)

- Complete reference for SAHI command-line interface
- Prediction commands
- FiftyOne integration
- COCO dataset operations
- Environment information
- Version checking
- Custom script usage

### [FiftyOne Integration](fiftyone.md)
- Guide for visualizing and analyzing predictions with FiftyOne
- Dataset visualization
- Result exploration
- Interactive analysis

## Interactive Examples

All documentation files are complemented by interactive Jupyter notebooks in the [demo directory](/notebooks/):

- `slicing.ipynb` - Slicing operations demonstration
- `inference_for_ultralytics.ipynb` - YOLOv8/YOLO11/YOLO12 integration
- `inference_for_yolov5.ipynb` - YOLOv5 integration
- `inference_for_mmdetection.ipynb` - MMDetection integration
- `inference_for_huggingface.ipynb` - HuggingFace models integration
- `inference_for_torchvision.ipynb` - TorchVision models integration
- `inference_for_rtdetr.ipynb` - RT-DETR integration

## Getting Started

If you're new to SAHI:

1. Start with the [prediction utilities](predict.md) to understand basic inference
2. Explore the [slicing utilities](slicing.md) to learn about processing large images
3. Check out the [CLI commands](cli.md) for command-line usage
4. Dive into [COCO utilities](coco.md) for dataset operations
5. Try the interactive notebooks in the [demo directory](../demo/) for hands-on experience
