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
    <br>
    <a href="https://badge.fury.io/py/sahi"><img src="https://badge.fury.io/py/sahi.svg" alt="pypi version"></a>
    <a href="https://anaconda.org/conda-forge/sahi"><img src="https://anaconda.org/conda-forge/sahi/badges/version.svg" alt="conda version"></a>
    <a href="https://github.com/obss/sahi/actions/workflows/ci.yml"><img src="https://github.com/obss/sahi/actions/workflows/ci.yml/badge.svg" alt="Continious Integration"></a>
  <br>
    <a href="https://ieeexplore.ieee.org/document/9897990"><img src="https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg" alt="ci"></a>
  <br>
    <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>

</div>
</div>

## <div align="center">Overview</div>

Object detection and instance segmentation are by far the most important applications in Computer Vision. However, the detection of small objects and inference on large images still need to be improved in practical usage. Here comes the SAHI to help developers overcome these real-world problems with many vision utilities.

| Command  | Description  |
|---|---|
| [predict](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage)  | perform sliced/standard video/image prediction using any [ultralytics](https://github.com/ultralytics/ultralytics)/[mmdet](https://github.com/open-mmlab/mmdetection)/[huggingface](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads)/[torchvision](https://pytorch.org/vision/stable/models.html#object-detection) model |
| [predict-fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-fiftyone-command-usage)  | perform sliced/standard prediction using any [ultralytics](https://github.com/ultralytics/ultralytics)/[mmdet](https://github.com/open-mmlab/mmdetection)/[huggingface](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads)/[torchvision](https://pytorch.org/vision/stable/models.html#object-detection) model and explore results in [fiftyone app](https://github.com/voxel51/fiftyone) |
| [coco slice](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-slice-command-usage)  | automatically slice COCO annotation and image files |
| [coco fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-fiftyone-command-usage)  | explore multiple prediction results on your COCO dataset with [fiftyone ui](https://github.com/voxel51/fiftyone) ordered by number of misdetections |
| [coco evaluate](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-evaluate-command-usage)  | evaluate classwise COCO AP and AR for given predictions and ground truth |
| [coco analyse](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-analyse-command-usage)  | calculate and export many error analysis plots |
| [coco yolo](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-yolo-command-usage)  | automatically convert any COCO dataset to [ultralytics](https://github.com/ultralytics/ultralytics) format |

## <div align="center">Quick Start Examples</div>

[📜 List of publications that cite SAHI (currently 300+)](https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites=14065474760484865747&scipsc=&q=&scisbd=1)

[🏆 List of competition winners that used SAHI](https://github.com/obss/sahi/discussions/688)

### Tutorials

- [Introduction to SAHI](https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80)

- [Official paper](https://ieeexplore.ieee.org/document/9897990) (ICIP 2022 oral)

- [Pretrained weights and ICIP 2022 paper files](https://github.com/fcakyon/small-object-detection-benchmark)

- [2025 Video Tutorial](https://www.youtube.com/watch?v=ILqMBah5ZvI) (RECOMMENDED)

- [Visualizing and Evaluating SAHI predictions with FiftyOne](https://voxel51.com/blog/how-to-detect-small-objects/)

- ['Exploring SAHI' Research Article from 'learnopencv.com'](https://learnopencv.com/slicing-aided-hyper-inference/)

- [Slicing Aided Hyper Inference Explained by Encord](https://encord.com/blog/slicing-aided-hyper-inference-explained/)

- ['VIDEO TUTORIAL: Slicing Aided Hyper Inference for Small Object Detection - SAHI'](https://www.youtube.com/watch?v=UuOjJKxn-M8&t=270s)

- [Video inference support is live](https://github.com/obss/sahi/discussions/626)

- [Kaggle notebook](https://www.kaggle.com/remekkinas/sahi-slicing-aided-hyper-inference-yv5-and-yx)

- [Satellite object detection](https://blog.ml6.eu/how-to-detect-small-objects-in-very-large-images-70234bab0f98)

- [Error analysis plots & evaluation](https://github.com/obss/sahi/discussions/622) (RECOMMENDED)

- [Interactive result visualization and inspection](https://github.com/obss/sahi/discussions/624) (RECOMMENDED)

- [COCO dataset conversion](https://medium.com/codable/convert-any-dataset-to-coco-object-detection-format-with-sahi-95349e1fe2b7)

- [Slicing operation notebook](demo/slicing.ipynb)

- `YOLOX` + `SAHI` demo: <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="sahi-yolox"></a>

- `YOLO12` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo12"></a> (NEW)

- `YOLO11-OBB` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo11-obb"></a> (NEW)

- `YOLO11` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo11"></a>

- `RT-DETR v2` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-rtdetrv2"></a> (NEW)

- `RT-DETR` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-rtdetr"></a> (NEW)

- `HuggingFace` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-huggingface"></a>

- `YOLOv5` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolov5"></a>

- `MMDetection` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-mmdetection"></a>

- `TorchVision` + `SAHI` walkthrough: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-torchvision"></a>

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
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
```

(torch 2.1.2 is required for mmdet support):

```console
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

- Install your desired detection framework (yolov5):

```console
pip install yolov5==7.0.14 sahi==0.11.21
```

- Install your desired detection framework (ultralytics):

```console
pip install ultralytics>=8.3.86
```

- Install your desired detection framework (mmdet):

```console
pip install mim
mim install mmdet==3.3.0
```

- Install your desired detection framework (huggingface):

```console
pip install transformers>=4.42.0 timm
```

</details>

### Framework Agnostic Sliced/Standard Prediction

<img width="700" alt="sahi-predict" src="https://user-images.githubusercontent.com/34196005/149310540-e32f504c-6c9e-4691-8afd-59f3a1a457f0.gif">

Find detailed info on `sahi predict` command at [cli.md](docs/cli.md#predict-command-usage).

Find detailed info on video inference at [video inference tutorial](https://github.com/obss/sahi/discussions/626).

Find detailed info on image/dataset slicing utilities at [slicing.md](docs/slicing.md).

### Error Analysis Plots & Evaluation

<img width="700" alt="sahi-analyse" src="https://user-images.githubusercontent.com/34196005/149537858-22b2e274-04e8-4e10-8139-6bdcea32feab.gif">

Find detailed info at [Error Analysis Plots & Evaluation](https://github.com/obss/sahi/discussions/622).

### Interactive Visualization & Inspection

<img width="700" alt="sahi-fiftyone" src="https://user-images.githubusercontent.com/34196005/149321540-e6ddd5f3-36dc-4267-8574-a985dd0c6578.gif">

Find detailed info at [Interactive Result Visualization and Inspection](https://github.com/obss/sahi/discussions/624).

### Other utilities

Find detailed info on COCO utilities (yolov5 conversion, slicing, subsampling, filtering, merging, splitting) at [coco.md](docs/coco.md).

## <div align="center">Citation</div>

If you use this package in your work, please cite it as:

```bibtex
@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={2022 IEEE International Conference on Image Processing (ICIP)},
  doi={10.1109/ICIP46576.2022.9897990},
  pages={966-970},
  year={2022}
}
```

```bibtex
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

### Add new frameworks

`sahi` library currently supports all [Ultralytics (YOLOv8/v10/v11/RTDETR) models](https://github.com/ultralytics/ultralytics), [MMDetection models](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/model_zoo.md), [Detectron2 models](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md), and [HuggingFace object detection models](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads). Moreover, it is easy to add new frameworks.

All you need to do is, create a new .py file under [sahi/models/](https://github.com/obss/sahi/tree/main/sahi/models) folder and create a new class in that .py file that implements [DetectionModel class](https://github.com/obss/sahi/blob/aaeb57c39780a5a32c4de2848e54df9a874df58b/sahi/models/base.py#L12). You can take the [MMDetection wrapper](https://github.com/obss/sahi/blob/aaeb57c39780a5a32c4de2848e54df9a874df58b/sahi/models/mmdet.py#L91) or [YOLOv5 wrapper](https://github.com/obss/sahi/blob/7e48bdb6afda26f977b763abdd7d8c9c170636bd/sahi/models/yolov5.py#L17) as a reference.

### Open a Pull Request

- Install the [uv package manager](https://docs.astral.sh/uv/) on your system.
- Install [pre-commit](https://pre-commit.com/) hooks with `uv run pre-commit install`.

## <div align="center">Contributors</div>

<div align="center">

<a align="left" href="https://github.com/fcakyon" target="_blank">Fatih Cagatay Akyon</a>

<a align="left" href="https://github.com/sinanonur" target="_blank">Sinan Onur Altinuc</a>

<a align="left" href="https://github.com/devrimcavusoglu" target="_blank">Devrim Cavusoglu</a>

<a align="left" href="https://github.com/cemilcengiz" target="_blank">Cemil Cengiz</a>

<a align="left" href="https://github.com/oulcan" target="_blank">Ogulcan Eryuksel</a>

<a align="left" href="https://github.com/kadirnar" target="_blank">Kadir Nar</a>

<a align="left" href="https://github.com/Dronakurl" target="_blank">Dronakurl</a>

<a align="left" href="https://github.com/madenburak" target="_blank">Burak Maden</a>

<a align="left" href="https://github.com/PushpakBhoge" target="_blank">Pushpak Bhoge</a>

<a align="left" href="https://github.com/mcvarer" target="_blank">M. Can V.</a>

<a align="left" href="https://github.com/ChristofferEdlund" target="_blank">Christoffer Edlund</a>

<a align="left" href="https://github.com/ishworii" target="_blank">Ishwor</a>

<a align="left" href="https://github.com/mecevit" target="_blank">Mehmet Ecevit</a>

<a align="left" href="https://github.com/ssahinnkadir" target="_blank">Kadir Sahin</a>

<a align="left" href="https://github.com/weypro" target="_blank">Wey</a>

<a align="left" href="https://github.com/youngjae-avikus" target="_blank">Youngjae</a>

<a align="left" href="https://github.com/tureckova" target="_blank">Alzbeta Tureckova</a>

<a align="left" href="https://github.com/s-aiueo32" target="_blank">So Uchida</a>

<a align="left" href="https://github.com/developer0hye" target="_blank">Yonghye Kwon</a>

<a align="left" href="https://github.com/aphilas" target="_blank">Neville</a>

<a align="left" href="https://github.com/mayrajeo" target="_blank">Janne Mäyrä</a>

<a align="left" href="https://github.com/christofferedlund" target="_blank">Christoffer Edlund</a>

<a align="left" href="https://github.com/ilkermanap" target="_blank">Ilker Manap</a>

<a align="left" href="https://github.com/nguyenthean" target="_blank">Nguyễn Thế An</a>

<a align="left" href="https://github.com/weiji14" target="_blank">Wei Ji</a>

<a align="left" href="https://github.com/aynursusuz" target="_blank">Aynur Susuz</a>

<a align="left" href="https://github.com/pranavdurai10" target="_blank">Pranav Durai</a>

<a align="left" href="https://github.com/lakshaymehra" target="_blank">Lakshay Mehra</a>

<a align="left" href="https://github.com/karl-joan" target="_blank">Karl-Joan Alesma</a>

<a align="left" href="https://github.com/jacobmarks" target="_blank">Jacob Marks</a>

<a align="left" href="https://github.com/williamlung" target="_blank">William Lung</a>

<a align="left" href="https://github.com/amoghdhaliwal" target="_blank">Amogh Dhaliwal</a>

</div>
