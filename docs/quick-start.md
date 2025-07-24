---
hide:
  - navigation
---

# Quick Start

Welcome to SAHI! This guide will get you up and running with the core features of the library, including installation, performing predictions, and using the command-line interface.

## 1. Installation

Install SAHI using pip. For object detection, it's recommended to also install `ultralytics`.

!!! example "Install"

    <p align="left" style="margin-bottom: -20px;">![PyPI - Python Version](https://img.shields.io/pypi/pyversions/sahi?logo=python&logoColor=gold)<p>

    === "Pip install (recommended)"

        Install or update the `sahi` package using pip by running `pip install -U sahi`. For more details on the `sahi` package, visit the [Python Package Index (PyPI)](https://pypi.org/project/sahi/).

        [![PyPI - Version](https://img.shields.io/pypi/v/sahi?logo=pypi&logoColor=white)](https://pypi.org/project/sahi/)
        [![Downloads](https://static.pepy.tech/badge/sahi)](https://www.pepy.tech/projects/sahi)

        ```bash
        # Install the sahi package from PyPI
        pip install sahi
        ```

        You can also install `sahi` directly from the [Sahi GitHub repository](https://github.com/obss/sahi). This can be useful if you want the latest development version. Ensure you have the Git command-line tool installed, and then run:

        ```bash
        # Install the sahi package from GitHub
        pip install git+https://github.com/obss/sahi.git@main
        ```

    === "Conda install"

        Conda can be used as an alternative package manager to pip. For more details, visit [Anaconda](https://anaconda.org/conda-forge/sahi). The Sahi feedstock repository for updating the conda package is available at [GitHub](https://github.com/conda-forge/sahi-feedstock/).

        [![Conda Version](https://img.shields.io/conda/vn/conda-forge/sahi?logo=condaforge)](https://anaconda.org/conda-forge/sahi)
        [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/sahi.svg)](https://anaconda.org/conda-forge/sahi)
        [![Conda Recipe](https://img.shields.io/badge/recipe-sahi-green.svg)](https://anaconda.org/conda-forge/sahi)
        [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/sahi.svg)](https://anaconda.org/conda-forge/sahi)

        ```bash
        # Install the sahi package using conda
        conda install -c conda-forge sahi
        ```

        !!! note

            If you are installing in a CUDA environment, it is best practice to install `ultralytics`, `pytorch`, and `pytorch-cuda` in the same command. This allows the conda package manager to resolve any conflicts. Alternatively, install `pytorch-cuda` last to override the CPU-specific `pytorch` package if necessary.
            ```bash
            # Install all packages together using conda
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

    === "Git clone"

        Clone the [Sahi GitHub repository](https://github.com/obss/sahi) if you are interested in contributing to development or wish to experiment with the latest source code. After cloning, navigate into the directory and install the package in editable mode `-e` using pip.

        [![GitHub last commit](https://img.shields.io/github/last-commit/obss/sahi?logo=github)](https://github.com/obss/sahi)
        [![GitHub commit activity](https://img.shields.io/github/commit-activity/t/obss/sahi)](https://github.com/obss/sahi/commits/main)
        [![GitHub contributors](https://img.shields.io/github/contributors/obss/sahi)](https://github.com/obss/sahi/graphs/contributors)

        ```bash
        # Clone the sahi repository
        git clone https://github.com/obss/sahi

        # Navigate to the cloned directory
        cd sahi

        # Install the package in editable mode for development
        pip install -e .
        ```


See the `sahi` [pyproject.toml](https://github.com/obss/sahi/blob/main/pyproject.toml) file for a list of dependencies.

## 2. Sliced Prediction with Python

Sliced inference is the core feature of SAHI, allowing you to detect small objects in large images. Here's a simple example using the Python API:

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Initialize a YOLOv8 model
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',
    model_path='yolov8n.pt', # or any other YOLOv8 model
    confidence_threshold=0.25,
    device="cuda:0", # or "cpu"
)

# Run sliced prediction
result = get_sliced_prediction(
    "path/to/your/image.jpg",
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

# Export visualizations
result.export_visuals(export_dir="demo_data/")

# Get predictions as a list of objects
predictions = result.object_prediction_list
```

## 3. Prediction with the CLI

SAHI also provides a powerful command-line interface for quick predictions without writing any Python code.

```bash
sahi predict --model_path yolov8n.pt --model_type yolov8 --source /path/to/images/ --slice_height 512 --slice_width 512
```

This command will run sliced inference on all images in the specified directory and save the results.

## Next Steps

You've now seen the basics of SAHI! To dive deeper, check out these resources:

* **Prediction In-Depth**: For advanced prediction options, see the [Prediction Utilities guide](predict.md).
* **Demos**: Explore our interactive notebooks in the [demo directory](../demo/) for hands-on examples with different models.
* **COCO Tools**: Learn how to create, manipulate, and convert datasets in the [COCO Utilities guide](coco.md).
* **All CLI Commands**: See the full list of commands in the [CLI documentation](cli.md).