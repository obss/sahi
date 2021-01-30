# SAHI: Slicing Aided Hyper Inference

[![PyPI version](https://badge.fury.io/py/sahi.svg)](https://badge.fury.io/py/sahi)
[![Conda version](https://anaconda.org/obss/sahi/badges/version.svg)](https://anaconda.org/obss/sahi)
[![CI](https://github.com/obss/sahi/workflows/CI/badge.svg)](https://github.com/obss/sahi/actions?query=event%3Apush+branch%3Amain+is%3Acompleted+workflow%3ACI)

A vision library for performing sliced inference on large images/small objects

<img width="700" alt="teaser" src="./demo/sliced_inference.gif">

## Overview

Object detection and instance segmentation are by far the most important fields of applications in Computer Vision. However, detection of small objects and inference on large images are still major issues in practical usage. Here comes the SAHI to help developers overcome these real-world problems.

## Getting started

### Blogpost

Check the [official SAHI blog post](https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80).


### Installation

- Install sahi using conda:

```console
conda install -c obss sahi
```

- Install sahi using pip:

```console
pip install sahi
```

- Install your desired version of pytorch and torchvision:
```console
pip install torch torchvision
```

- Install your desired detection framework (such as mmdet):
```console
pip install mmdet
```

## Usage

- Sliced inference:
```python
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

```
Refer to [inference notebook](demo/inference.ipynb) for detailed usage.

- Slice an image:
```python
from sahi.slicing import slice_image

slice_image_result, num_total_invalid_segmentation = slice_image(
    image=image_path,
    output_file_name=output_file_name,
    output_dir=output_dir,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

- Slice a coco formatted dataset:
```python
from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path=coco_annotation_file_path,
    image_dir=image_dir,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

## Adding new detection framework support

sahi library currently only supports [MMDetection models](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md). However it is easy to add new frameworks.

All you need to do is, creating a new class in [model.py](sahi/model.py) that implements [DetectionModel class](https://github.com/obss/sahi/blob/651f8e6cdb20467815748764bb198dd50241ab2b/sahi/model.py#L10). You can take the [MMDetection wrapper](https://github.com/obss/sahi/blob/651f8e6cdb20467815748764bb198dd50241ab2b/sahi/model.py#L164) as a reference.
