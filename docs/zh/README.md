<div align="center">
<h1>
  SAHI: 切片辅助高效推理
</h1>

<h4>
  一个轻量级的计算机视觉库，可实现大规模的目标检测和实例分割
</h4>

<h4>
    <img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sahi-sliced-inference-overview.avif">
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
    <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>
    <a href="https://deepwiki.com/obss/sahi"><img src="https://img.shields.io/badge/DeepWiki-obss%2Fsahi-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==" alt="Sliced/tiled inference DeepWiki"></a>
    <a href="https://squidfunk.github.io/mkdocs-material/"><img src="https://img.shields.io/badge/Material_for_MkDocs-526CFE?logo=MaterialForMkDocs&logoColor=white" alt="built-with-material-for-mkdocs"></a>

</div>
</div>

## <div align="center">概览</div>

SAHI 通过启用**切片推理**来检测大图像中的小物体，从而帮助开发人员克服了对象检测中的实际挑战。它支持各种流行的检测模型并提供易于使用的 API。

<div align="center">

🌐 [English](../../README.md) | 🇨🇳 [简体中文](README.md)

</div>

| 命令  | 描述  |
|---|---|
| [predict](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage)  | 使用任意 [ultralytics](https://github.com/ultralytics/ultralytics)/[mmdet](https://github.com/open-mmlab/mmdetection)/[huggingface](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads)/[torchvision](https://pytorch.org/vision/stable/models.html#object-detection) 模型进行切片或标准视频 / 图像预测 - 参见 [命令行指南](cli.md#predict-command-usage) |
| [predict-fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-fiftyone-command-usage)  | 使用任意支持的模型进行切片或标准预测，并在 [fiftyone应用](https://github.com/voxel51/fiftyone) 中探索结果 - [了解更多](fiftyone.md) |
| [coco slice](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-slice-command-usage)  | 自动切片 COCO 标注和图像文件 - 参见 [切片工具](slicing.md) |
| [coco fiftyone](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-fiftyone-command-usage)  | 在 [fiftyone ui](https://github.com/voxel51/fiftyone) 中探索 COCO 数据集的多个预测结果，按错误检测数量排序 |
| [coco evaluate](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-evaluate-command-usage)  | 针对给定的预测和真实数据评估 COCO 的类级别 AP 和 AR - 查看 [COCO 工具](coco.md) |
| [coco analyse](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-analyse-command-usage)  | 计算并导出多种错误分析图表 - 参见 [完整指南](../README.md) |
| [coco yolo](https://github.com/obss/sahi/blob/main/docs/cli.md#coco-yolo-command-usage)  | 将任意 COCO 数据集自动转换为 [ultralytics](https://github.com/ultralytics/ultralytics) 格式 |

### 社区认可

[📜 引用 SAHI 的出版物列表（当前超过 600 篇）](https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites=14065474760484865747&scipsc=&q=&scisbd=1)

[🏆 使用 SAHI 的竞赛获奖者列表](https://github.com/obss/sahi/discussions/688)

### AI 工具认可
SAHI 的文档已在 [Context7 MCP](https://context7.com/obss/sahi) 中建立索引，为 AI 编码助手提供最新的，版本特定的代码示例和 API 参考。我们还提供了一个遵循 AI 可读文档新兴标准的 [llms.txt](https://context7.com/obss/sahi/llms.txt) 文件。要将 SAHI 文档集成到您的 AI 开发工作流程中，请查看 [Context7 MCP 安装指南](https://github.com/upstash/context7#%EF%B8%8F-installation).

## <div align="center">安装</div>

### 基本安装
```bash
pip install sahi
```

<details closed>
<summary>
<big><b>详细安装说明（点击展开）</b></big>
</summary>

- 安装您所需的 PyTorch 和 torchvision 版本：

```console
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```
（为了获得mmdet框架的支持，您需要安装torch 2.1.2版本）：

```console
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

- 安装您所需的检测框架 (ultralytics):

```console
pip install ultralytics>=8.3.161
```

- 安装您所需的检测框架 (huggingface):

```console
pip install transformers>=4.49.0 timm
```

- 安装您所需的检测框架 (yolov5):

```console
pip install yolov5==7.0.14 sahi==0.11.21
```

- 安装您所需的检测框架 (mmdet):

```console
pip install mim
mim install mmdet==3.3.0
```

- 安装您所需的检测框架 (roboflow):

```console
pip install inference>=0.50.3 rfdetr>=1.1.0
```

</details>

## <div align="center">快速开始</div>

### 教程

- [SAHI 简介](https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80) - 请查阅 [完整的文档](../README.md) 以了解高级用法。

- [官方论文](https://ieeexplore.ieee.org/document/9897990) (ICIP 2022 oral)

- [预训练权重 和 ICIP 2022 论文文件](https://github.com/fcakyon/small-object-detection-benchmark)

- [视频教程（2025年）](https://www.youtube.com/watch?v=ILqMBah5ZvI) (推荐)

- [使用 FiftyOne 可视化并评估 SAHI 的预测结果](https://voxel51.com/blog/how-to-detect-small-objects/)

- [《探索 SAHI》——来自 learnopencv.com 的研究文章](https://learnopencv.com/slicing-aided-hyper-inference/)

- [Encord 对 Slicing Aided Hyper Inference（SAHI）的解读](https://encord.com/blog/slicing-aided-hyper-inference-explained/)

- [视频教程：SAHI 在小目标检测中的应用](https://www.youtube.com/watch?v=UuOJKxn-M8&t=270s)

- [视频推理支持现已上线](https://github.com/obss/sahi/discussions/626)

- [Kaggle notebook](https://www.kaggle.com/remekkinas/sahi-slicing-aided-hyper-inference-yv5-and-yx)

- [卫星图像目标检测](https://blog.ml6.eu/how-to-detect-small-objects-in-very-large-images-70234bab0f98)

- [误差分析绘图 & 评估](https://github.com/obss/sahi/discussions/622) (推荐)

- [交互式结果可视化与检查](https://github.com/obss/sahi/discussions/624) (推荐)

- [COCO 数据集转换](https://medium.com/codable/convert-any-dataset-to-coco-object-detection-format-with-sahi-95349e1fe2b7)

- [切片操作 notebook 示例](../../demo/slicing.ipynb)

- `YOLOX` + `SAHI` 示例: <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="sahi-yolox"></a>

- `YOLO12` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo12"></a>

- `YOLO11-OBB` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo11-obb"></a> (NEW)

- `YOLO11` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolo11"></a>

- `Roboflow/RF-DETR` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="roboflow"></a> (NEW)

- `RT-DETR v2` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-rtdetrv2"></a> (NEW)

- `RT-DETR` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-rtdetr"></a>

- `HuggingFace` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-huggingface"></a>

- `YOLOv5` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-yolov5"></a>

- `MMDetection` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-mmdetection"></a>

- `TorchVision` + `SAHI` 实战教程: <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="sahi-torchvision"></a>

<a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img width="600" src="https://user-images.githubusercontent.com/34196005/144092739-c1d9bade-a128-4346-947f-424ce00e5c4f.gif" alt="sahi-yolox"></a>

### 与框架无关的切片/标准预测

<img width="700" alt="sahi-predict" src="https://user-images.githubusercontent.com/34196005/149310540-e32f504c-6c9e-4691-8afd-59f3a1a457f0.gif">

请在 [CLI 文档](cli.md#predict-command-usage) 中查找关于使用 `sahi predict` 命令的详细信息，并查阅 [预测 API](predict.md) 以了解高级用法。

请在 [视频推理教程](https://github.com/obss/sahi/discussions/626) 中查找关于视频推理的详细信息。

### 误差分析绘图 & 评估

<img width="700" alt="sahi-analyse" src="https://user-images.githubusercontent.com/34196005/149537858-22b2e274-04e8-4e10-8139-6bdcea32feab.gif">

请在 [误差分析绘图 & 评估](https://github.com/obss/sahi/discussions/622) 中查找相关的详细信息。

### 交互式结果可视化与检查

<img width="700" alt="sahi-fiftyone" src="https://user-images.githubusercontent.com/34196005/149321540-e6dd5f3-36dc-4267-8574-a985dd0c6578.gif">

探索 [FiftyOne 集成](fiftyone.md) 以实现交互式可视化与检查。

### 其他实用工具

请查阅 [完整的 COCO 工具指南](coco.md) 了解 YOLO 格式转换、数据集切片、子采样、筛选、合并与分割等操作。了解更多关于 [切片工具](slicing.md) ，以实现对图像和数据集切片参数的精细控制。

## <div align="center">引用</div>
如果您在您的工作中使用了这个包，请如下文引用：

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

## <div align="center">贡献者</div>

欢迎贡献！请参阅我们的 [贡献指南](../../CONTRIBUTING.md) 来开始使用. 感谢所有贡献者🙏！

<p align="center">
    <a href="https://github.com/obss/sahi/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=obss/sahi" />
    </a>
</p>
