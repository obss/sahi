# SAHI Documentation

Welcome to the SAHI documentation! This directory contains detailed guides and tutorials for using SAHI's various features. Below is an overview of each documentation file and what you'll find in it.

## Core Documentation Files

### [Prediction Utilities](predict.md)
- Detailed guide for performing object detection inference
- Standard and sliced inference examples
- Batch prediction usage
- Class exclusion during inference
- Visualization parameters and export formats
- Interactive examples with various model integrations (YOLOv8, MMDetection, etc.)

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

All documentation files are complemented by interactive Jupyter notebooks in the [demo directory](../demo/):
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
