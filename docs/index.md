---
hide:
  - navigation
  - toc
tags:
  - getting-started
  - object-detection
  - small-object-detection
  - slicing
  - instance-segmentation
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
    <a href="https://github.com/obss/sahi/actions/workflows/ci.yml"><img src="https://github.com/obss/sahi/actions/workflows/ci.yml/badge.svg" alt="Continuous Integration"></a>
  <br>
    <a href="https://context7.com/obss/sahi"><img src="https://img.shields.io/badge/Context7%20MCP-Indexed-blue" alt="Context7 MCP"></a>
    <a href="https://context7.com/obss/sahi/llms.txt"><img src="https://img.shields.io/badge/llms.txt-✓-brightgreen" alt="llms.txt"></a>
    <a href="https://ieeexplore.ieee.org/document/9897990"><img src="https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg" alt="ci"></a>
    <a href="https://arxiv.org/abs/2202.06934"><img src="https://img.shields.io/badge/arXiv-2202.06934-b31b1b.svg" alt="arXiv"></a>
    <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>
</div>
</div>

## What is SAHI?

SAHI (Slicing Aided Hyper Inference) is an open-source framework that provides a
generic slicing-aided inference and fine-tuning pipeline for small object
detection. Detecting small objects and those far from the camera is a major
challenge in surveillance applications, as they are represented by a small
number of pixels and lack sufficient detail for conventional detectors.

SAHI addresses this by applying a unique methodology that can be used with any
object detector without requiring additional fine-tuning. Experimental
evaluations on the Visdrone and xView aerial object detection datasets show that
SAHI can increase object detection AP by up to 6.8% for FCOS, 5.1% for VFNet,
and 5.3% for TOOD detectors. With slicing-aided fine-tuning, the accuracy can be
further improved, resulting in a cumulative increase of 12.7%, 13.4%, and 14.5%
AP, respectively. The technique has been successfully integrated with
Ultralytics (YOLOv8, YOLO11, YOLO26), HuggingFace Transformers,
RT-DETR, TorchVision, MMDetection, Detectron2, YOLOv5, YOLOE, YOLO-World, and
Roboflow RF-DETR models.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Getting Started__

    ---

    Install `sahi` with pip and get up and running in minutes.

    [:octicons-arrow-right-24: Quickstart](quick-start.md)

-   :material-lightbulb-outline:{ .lg .middle } __How It Works__

    ---

    Understand the slicing algorithm, when to use it, and how to tune it.

    [:octicons-arrow-right-24: Sliced Inference](guides/sliced-inference.md)

-   :material-puzzle-outline:{ .lg .middle } __Model Integrations__

    ---

    Use SAHI with Ultralytics, HuggingFace, MMDetection, TorchVision, and more.

    [:octicons-arrow-right-24: All models](guides/models.md)

-   :material-image:{ .lg .middle } __Predict__

    ---

    Predict on new images, videos, and streams with SAHI.

    [:octicons-arrow-right-24: Learn more](predict.md)

-   :material-content-cut:{ .lg .middle } __Slicing__

    ---

    Learn how to slice large images and datasets for inference.

    [:octicons-arrow-right-24: Learn more](slicing.md)

-   :material-database:{ .lg .middle } __COCO Utilities__

    ---

    Work with COCO format datasets, including creation, splitting, and filtering.

    [:octicons-arrow-right-24: Learn more](coco.md)

-   :material-console:{ .lg .middle } __CLI Commands__

    ---

    Use SAHI from the command-line for prediction and dataset operations.

    [:octicons-arrow-right-24: Learn more](cli.md)

-   :material-eye:{ .lg .middle } __FiftyOne__

    ---

    Interactively explore and compare detection results.

    [:octicons-arrow-right-24: Learn more](fiftyone.md)

-   :material-notebook:{ .lg .middle } __Notebooks__

    ---

    Hands-on Colab notebooks for every supported framework.

    [:octicons-arrow-right-24: Browse notebooks](notebooks.md)

</div>
