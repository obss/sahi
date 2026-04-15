<div align="center">
<h1>
  SAHI: Slicing Aided Hyper Inference
</h1>

<h4>
  A lightweight vision library for performing large scale object detection & instance segmentation
</h4>

<h4>
    <img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sahi-sliced-inference-overview.avif">
</h4>

<!-- Downloads & Version -->
<div>
  <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi" alt="Total Downloads"></a>
  <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi/month" alt="Monthly Downloads"></a>
  <a href="https://badge.fury.io/py/sahi"><img src="https://badge.fury.io/py/sahi.svg" alt="PyPI Version"></a>
  <a href="https://anaconda.org/conda-forge/sahi"><img src="https://anaconda.org/conda-forge/sahi/badges/version.svg" alt="Conda Version"></a>
  <a href="https://github.com/obss/sahi/blob/main/LICENSE.md"><img src="https://img.shields.io/pypi/l/sahi" alt="License"></a>
</div>

<!-- CI & Quality -->
<div>
  <a href="https://github.com/obss/sahi/actions/workflows/ci.yml"><img src="https://github.com/obss/sahi/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://security.snyk.io/package/pip/sahi"><img src="https://img.shields.io/badge/Snyk_security-monitored-8A2BE2" alt="Known Vulnerabilities"></a>
  <a href="https://www.codefactor.io/repository/github/onuralpszr/sahi"><img src="https://www.codefactor.io/repository/github/onuralpszr/sahi/badge" alt="CodeFactor"></a>
  <a href="https://ieeexplore.ieee.org/document/9897990"><img src="https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg" alt="DOI"></a>
</div>

<!-- AI & Docs -->
<div>
  <a href="https://context7.com/obss/sahi"><img src="https://img.shields.io/badge/Context7%20MCP-Indexed-blue" alt="Context7 MCP"></a>
  <a href="https://context7.com/obss/sahi/llms.txt"><img src="https://img.shields.io/badge/llms.txt-✓-brightgreen" alt="llms.txt"></a>
  <a href="https://deepwiki.com/obss/sahi"><img src="https://img.shields.io/badge/DeepWiki-obss%2Fsahi-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==" alt="DeepWiki"></a>
  <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>
</div>

</div>

## <div align="center">Overview</div>

SAHI helps developers overcome real-world challenges in object detection by
enabling **sliced inference** for detecting small objects in large images. It
supports various popular detection models and provides easy-to-use APIs.

<div align="center">

🌐 [English](README.md) | 🇨🇳 [简体中文](docs/zh/README.md)

</div>

| Command                                                                                               | Description                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [predict](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage)                   | Perform sliced/standard video/image prediction using any [ultralytics](https://github.com/ultralytics/ultralytics) / [mmdet](https://github.com/open-mmlab/mmdetection) / [huggingface](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads) / [torchvision](https://pytorch.org/vision/stable/models.html#object-detection) model — see [CLI guide](docs/cli.md#predict-command-usage) |
| [predict-fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-fiftyone-command-usage) | Perform sliced/standard prediction using any supported model and explore results in [fiftyone app](https://github.com/voxel51/fiftyone) — [learn more](docs/fiftyone.md)                                                                                                                                                                                                                                       |
| [coco slice](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-slice-command-usage)             | Automatically slice COCO annotation and image files — see [slicing utilities](docs/slicing.md)                                                                                                                                                                                                                                                                                                                 |
| [coco fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-fiftyone-command-usage)       | Explore multiple prediction results on your COCO dataset with [fiftyone ui](https://github.com/voxel51/fiftyone) ordered by number of misdetections                                                                                                                                                                                                                                                            |
| [coco evaluate](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-evaluate-command-usage)       | Evaluate classwise COCO AP and AR for given predictions and ground truth — check [COCO utilities](docs/coco.md)                                                                                                                                                                                                                                                                                                |
| [coco analyse](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-analyse-command-usage)         | Calculate and export many error analysis plots — see the [complete guide](docs/README.md)                                                                                                                                                                                                                                                                                                                      |
| [coco yolo](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-yolo-command-usage)               | Automatically convert any COCO dataset to [ultralytics](https://github.com/ultralytics/ultralytics) format                                                                                                                                                                                                                                                                                                     |

### Approved by the Community

[📜 List of publications that cite SAHI (currently 600+)](https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites=14065474760484865747&scipsc=&q=&scisbd=1)

[🏆 List of competition winners that used SAHI](https://github.com/obss/sahi/discussions/688)

### Approved by AI Tools

SAHI's documentation is
[indexed in Context7 MCP](https://context7.com/obss/sahi), providing AI coding
assistants with up-to-date, version-specific code examples and API references.
We also provide an [llms.txt](https://context7.com/obss/sahi/llms.txt) file
following the emerging standard for AI-readable documentation. To integrate SAHI
docs with your AI development workflow, check out the
[Context7 MCP installation guide](https://github.com/upstash/context7#%EF%B8%8F-installation).

## <div align="center">Installation</div>

### Basic Installation

```bash
pip install sahi
```

<details closed>
<summary>
<big><b>Detailed Installation (Click to open)</b></big>
</summary>

- Install your desired version of pytorch and torchvision:

```console
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```

(torch 2.1.2 is required for mmdet support):

```console
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

- Install your desired detection framework (ultralytics):

```console
pip install ultralytics>=8.3.161
```

- Install your desired detection framework (huggingface):

```console
pip install transformers>=4.49.0 timm
```

- Install your desired detection framework (yolov5):

```console
pip install yolov5==7.0.14 sahi==0.11.21
```

- Install your desired detection framework (mmdet):

```console
pip install mim
mim install mmdet==3.3.0
```

- Install your desired detection framework (roboflow):

```console
pip install inference>=0.51.5 rfdetr>=1.6.2
```

</details>

## <div align="center">Quick Start</div>

### Learning Resources

| Resource                                                                                                                                            | Type       |
| --------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| [Introduction to SAHI](https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80) | Blog Post  |
| [2025 Video Tutorial](https://www.youtube.com/watch?v=ILqMBah5ZvI) ⭐                                                                               | Video      |
| [Official Paper](https://ieeexplore.ieee.org/document/9897990) (ICIP 2022 oral)                                                                     | Paper      |
| [Pretrained Weights & ICIP 2022 Paper Files](https://github.com/fcakyon/small-object-detection-benchmark)                                           | Benchmark  |
| [Visualizing and Evaluating SAHI Predictions with FiftyOne](https://voxel51.com/blog/how-to-detect-small-objects/)                                  | Blog Post  |
| [Exploring SAHI – learnopencv.com](https://learnopencv.com/slicing-aided-hyper-inference/)                                                          | Article    |
| [Slicing Aided Hyper Inference Explained by Encord](https://encord.com/blog/slicing-aided-hyper-inference-explained/)                               | Article    |
| [Video Tutorial: SAHI for Small Object Detection](https://www.youtube.com/watch?v=UuOJKxn-M8&t=270s)                                                | Video      |
| [Satellite Object Detection](https://blog.ml6.eu/how-to-detect-small-objects-in-very-large-images-70234bab0f98)                                     | Blog Post  |
| [COCO Dataset Conversion](https://medium.com/codable/convert-any-dataset-to-coco-object-detection-format-with-sahi-95349e1fe2b7)                    | Blog Post  |
| [Kaggle Notebook](https://www.kaggle.com/remekkinas/sahi-slicing-aided-hyper-inference-yv5-and-yx)                                                  | Notebook   |
| [Error Analysis Plots & Evaluation](https://github.com/obss/sahi/discussions/622) ⭐                                                                | Discussion |
| [Interactive Result Visualization and Inspection](https://github.com/obss/sahi/discussions/624) ⭐                                                  | Discussion |
| [Video Inference Support](https://github.com/obss/sahi/discussions/626)                                                                             | Discussion |
| [Slicing Operation Notebook](demo/slicing.ipynb)                                                                                                    | Notebook   |
| [Complete Documentation](docs/README.md)                                                                                                            | Docs       |

### Notebooks & Demos

| Framework          | Notebook                                                                                                                                                                        | Demo                                                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YOLO12             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) | —                                                                                                                                                         |
| YOLO11             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) | —                                                                                                                                                         |
| YOLO11-OBB         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb) | —                                                                                                                                                         |
| Roboflow / RF-DETR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb)    | —                                                                                                                                                         |
| RT-DETR v2         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb) | —                                                                                                                                                         |
| RT-DETR            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb)      | —                                                                                                                                                         |
| HuggingFace        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb) | —                                                                                                                                                         |
| YOLOv5             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb)      | —                                                                                                                                                         |
| MMDetection        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb) | —                                                                                                                                                         |
| TorchVision        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb) | —                                                                                                                                                         |
| YOLOX              | —                                                                                                                                                                               | [![HuggingFace Spaces](https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg)](https://huggingface.co/spaces/fcakyon/sahi-yolox) |

<a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img width="600" src="https://user-images.githubusercontent.com/34196005/144092739-c1d9bade-a128-4346-947f-424ce00e5c4f.gif" alt="sahi-yolox"></a>

### Framework Agnostic Sliced/Standard Prediction

<img width="700" alt="sahi-predict" src="https://user-images.githubusercontent.com/34196005/149310540-e32f504c-6c9e-4691-8afd-59f3a1a457f0.gif">

Find detailed info on using `sahi predict` command in the
[CLI documentation](docs/cli.md#predict-command-usage) and explore the
[prediction API](docs/predict.md) for advanced usage.

Find detailed info on video inference at
[video inference tutorial](https://github.com/obss/sahi/discussions/626).

### Error Analysis Plots & Evaluation

<img width="700" alt="sahi-analyse" src="https://user-images.githubusercontent.com/34196005/149537858-22b2e274-04e8-4e10-8139-6bdcea32feab.gif">

Find detailed info at
[Error Analysis Plots & Evaluation](https://github.com/obss/sahi/discussions/622).

### Interactive Visualization & Inspection

<img width="700" alt="sahi-fiftyone" src="https://user-images.githubusercontent.com/34196005/149321540-e6dd5f3-36dc-4267-8574-a985dd0c6578.gif">

Explore [FiftyOne integration](docs/fiftyone.md) for interactive visualization
and inspection.

### Other Utilities

Check the [comprehensive COCO utilities guide](docs/coco.md) for YOLO
conversion, dataset slicing, subsampling, filtering, merging, and splitting
operations. Learn more about the [slicing utilities](docs/slicing.md) for
detailed control over image and dataset slicing parameters.

## <div align="center">Citation</div>

If you use this package in your work, please cite as:

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

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md)
to get started. Thank you 🙏 to all our contributors!

<p align="center">
    <a href="https://github.com/obss/sahi/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=obss/sahi" />
    </a>
</p>
