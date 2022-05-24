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
    <a href="https://doi.org/10.48550/arXiv.2202.06934"><img src="https://img.shields.io/badge/arXiv-2202.06934-b31b1b.svg" alt="ci"></a>
    <br>
    <a href="https://badge.fury.io/py/sahi"><img src="https://badge.fury.io/py/sahi.svg" alt="pypi version"></a>
    <a href="https://anaconda.org/conda-forge/sahi"><img src="https://anaconda.org/conda-forge/sahi/badges/version.svg" alt="conda version"></a>
    <a href="https://github.com/obss/sahi/actions?query=event%3Apush+branch%3Amain+is%3Acompleted+workflow%3ACI"><img src="https://github.com/obss/sahi/workflows/CI/badge.svg" alt="ci"></a>
    <br>
    <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>
    <br>
    
</div>
</div>

## <div align="center">Overview</div>

Object detection and instance segmentation are by far the most important fields of applications in Computer Vision. However, detection of small objects and inference on large images are still major issues in practical usage. Here comes the SAHI to help developers overcome these real-world problems with many vision utilities.

| Command  | Description  |
|---|---|
| [predict](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage)  | perform sliced/standard video/image prediction using any [yolov5](https://github.com/ultralytics/yolov5)/[mmdet](https://github.com/open-mmlab/mmdetection)/[detectron2](https://github.com/facebookresearch/detectron2) model |
| [predict-fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-fiftyone-command-usage)  | perform sliced/standard prediction using any [yolov5](https://github.com/ultralytics/yolov5)/[mmdet](https://github.com/open-mmlab/mmdetection)/[detectron2](https://github.com/facebookresearch/detectron2) model and explore results in [fiftyone app](https://github.com/voxel51/fiftyone) |
| [coco slice](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-slice-command-usage)  | automatically slice COCO annotation and image files |
| [coco fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-fiftyone-command-usage)  | explore multiple prediction results on your COCO dataset with [fiftyone ui](https://github.com/voxel51/fiftyone) ordered by number of misdetections |
| [coco evaluate](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-evaluate-command-usage)  | evaluate classwise COCO AP and AR for given predictions and ground truth |
| [coco analyse](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-analyse-command-usage)  | calcualate and export many error analysis plots |
| [coco yolov5](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-yolov5-command-usage)  | automatically convert any COCO dataset to [yolov5](https://github.com/ultralytics/yolov5) format |

## <div align="center">Quick Start Examples</div>

[Check this link for a list of competitions that SAHI made us win 🚀](https://github.com/obss/sahi/issues/384)

### Tutorials

- [Introduction to SAHI](https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80)

- [Official paper](https://arxiv.org/abs/2202.06934) (NEW)

- [Video inference support is live](https://github.com/obss/sahi/issues/457) (NEW)

- [Kaggle notebook](https://www.kaggle.com/remekkinas/sahi-slicing-aided-hyper-inference-yv5-and-yx) (NEW)

- [Satellite object detection](https://blog.ml6.eu/how-to-detect-small-objects-in-very-large-images-70234bab0f98) (NEW)

- [Error analysis plots & evaluation](https://github.com/obss/sahi/issues/356) (NEW)

- [Interactive result visualization and inspection](https://github.com/obss/sahi/issues/357) (NEW)

- [COCO dataset conversion](https://medium.com/codable/convert-any-dataset-to-coco-object-detection-format-with-sahi-95349e1fe2b7)

- [Slicing operation notebook](demo/slicing.ipynb)

- `YOLOX` + `SAHI` demo: <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="sahi-yolox"></a> (RECOMMENDED)

- `YOLOv5` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolov5"></a>

- `MMDetection` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-mmdetection"></a>

- `Detectron2` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-detectron2"></a> (NEW)

<a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img width="600" src="https://user-images.githubusercontent.com/34196005/144092739-c1d9bade-a128-4346-947f-424ce00e5c4f.gif" alt="sahi-yolox"></a> 


</details>

### Installation

<img width="700" alt="sahi-installation" src="https://user-images.githubusercontent.com/34196005/149311602-b44e6fe1-f496-40f2-a7ae-5ea1f66e1550.gif">


<details closed>
<summary>
<big><b>Installation details:</b></big>
</summary>

- Install `sahi` using pip:

```console
pip install sahi
```

- On Windows, `Shapely` needs to be installed via Conda:

```console
conda install -c conda-forge shapely
```

- Install your desired version of pytorch and torchvision:

```console
conda install pytorch=1.10.2 torchvision=0.11.3 cudatoolkit=11.3 -c pytorch
```
  
- Install your desired detection framework (yolov5):

```console
pip install yolov5
```

- Install your desired detection framework (mmdet):

```console
pip install mmcv-full==1.4.4 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```

```console
pip install mmdet==2.21.0
```

- Install your desired detection framework (detectron2):

```console
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

</details>

### Framework Agnostic Sliced/Standard Prediction

<img width="700" alt="sahi-predict" src="https://user-images.githubusercontent.com/34196005/149310540-e32f504c-6c9e-4691-8afd-59f3a1a457f0.gif">

Find detailed info on `sahi predict` command at [cli.md](docs/cli.md#predict-command-usage).

Find detailed info on video inference at [video inference tutorial](https://github.com/obss/sahi/issues/457).

Find detailed info on image/dataset slicing utilities at [slicing.md](docs/slicing.md).

### Error Analysis Plots & Evaluation

<img width="700" alt="sahi-analyse" src="https://user-images.githubusercontent.com/34196005/149537858-22b2e274-04e8-4e10-8139-6bdcea32feab.gif">

Find detailed info at [Error Analysis Plots & Evaluation](https://github.com/obss/sahi/issues/356).

### Interactive Visualization & Inspection

<img width="700" alt="sahi-fiftyone" src="https://user-images.githubusercontent.com/34196005/149321540-e6ddd5f3-36dc-4267-8574-a985dd0c6578.gif">

Find detailed info at [Interactive Result Visualization and Inspection](https://github.com/obss/sahi/issues/357).

### Other utilities

Find detailed info on COCO utilities (yolov5 conversion, slicing, subsampling, filtering, merging, splitting) at [coco.md](docs/coco.md).

Find detailed info on MOT utilities (ground truth dataset creation, exporting tracker metrics in mot challenge format) at [mot.md](docs/mot.md).

## <div align="center">Citation</div>

If you use this package in your work, please cite it as:

```
@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={arXiv preprint arXiv:2202.06934},
  year={2022}
}
```

```
@software{obss2021sahi,
  author       = {Akyon, Fatih Cagatay and Cengiz, Cemil and Altinuc, Sinan Onur and Cavusoglu, Devrim and Sahin, Kadir and Eryuksel, Ogulcan},
  title        = {{SAHI: A lightweight vision library for performing large scale object detection and instance segmentation}},
  month        = nov,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5718950},
  url          = {https://doi.org/10.5281/zenodo.5718950}
}
```

## <div align="center">Contributing</div>

`sahi` library currently supports all [YOLOv5 models](https://github.com/ultralytics/yolov5/releases) and [MMDetection models](https://github.com/open-mmlab/mmdetection/blob/master/docs/model_zoo.md). Moreover, it is easy to add new frameworks.

All you need to do is, creating a new class in [model.py](sahi/model.py) that implements [DetectionModel class](https://github.com/obss/sahi/blob/21ecb285aa6bf93c2a00605dfb9b138f19d8d62d/sahi/model.py#L21). You can take the [MMDetection wrapper](https://github.com/obss/sahi/blob/21ecb285aa6bf93c2a00605dfb9b138f19d8d62d/sahi/model.py#L177) or [YOLOv5 wrapper](https://github.com/obss/sahi/blob/21ecb285aa6bf93c2a00605dfb9b138f19d8d62d/sahi/model.py#L388) as a reference.

Before opening a PR:

- Install required development packages:

```bash
pip install -e ."[dev]"
```

- Reformat with black and isort:

```bash
black . --config pyproject.toml
isort .
```

## <div align="center">Contributors</div>

<div align="center">

<a align="left" href="https://github.com/fcakyon" target="_blank">Fatih Cagatay Akyon</a>

<a align="left" href="https://github.com/sinanonur" target="_blank">Sinan Onur Altinuc</a>

<a align="left" href="https://github.com/devrimcavusoglu" target="_blank">Devrim Cavusoglu</a>

<a align="left" href="https://github.com/cemilcengiz" target="_blank">Cemil Cengiz</a>

<a align="left" href="https://github.com/oulcan" target="_blank">Ogulcan Eryuksel</a>

<a align="left" href="https://github.com/kadirnar" target="_blank">Kadir Nar</a>

<a align="left" href="https://github.com/madenburak" target="_blank">Burak Maden</a>

<a align="left" href="https://github.com/ssahinnkadir" target="_blank">Kadir Sahin</a>

<a align="left" href="https://github.com/weiji14" target="_blank">Wei Ji</a>

</div>
