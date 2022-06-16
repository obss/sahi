import contextlib
import importlib.util
import logging
import os

# adapted from https://github.com/huggingface/transformers/src/transformers/utils/import_utils.py

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
)


def get_package_info(package_name: str):
    """
    Returns the package version as a string and the package name as a string.
    """
    try:
        package = importlib.import_module(package_name)
        version = package.__version__
        name = package.__name__
        logger.info(f"{name} version {version} available.")
    except ImportError:
        version = None
        name = None
    return version, name


_torch_available, _torch_version = get_package_info("torch")
_torchvision_available, _torchvision_version = get_package_info("torchvision")
_yolov5_available, _yolov5_version = get_package_info("yolov5")
_mmdet_available, _mmdet_version = get_package_info("mmdet")
_mmcv_available, _mmcv_version = get_package_info("mmcv")
_detectron2_available, _detectron2_version = get_package_info("detectron2")
_fiftyone_available, _fiftyone_version = get_package_info("fiftyone")
_layer_available, _layer_version = get_package_info("layer")


def is_available(module_name: str):
    return importlib.util.find_spec(module_name) is not None


@contextlib.contextmanager
def check_requirements(package_names):
    """
    Raise error if module is not installed.
    """
    missing_packages = []
    for package_name in package_names:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)
    if missing_packages:
        raise ImportError(f"The following packages are required to use this module: {missing_packages}")
    yield
