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
    _is_available = is_available(package_name)

    if _is_available:
        try:
            import importlib.metadata as _importlib_metadata

            _version = _importlib_metadata.version(package_name)
        except (importlib.metadata.PackageNotFoundError, ModuleNotFoundError):
            try:
                _version = importlib.import_module(package_name).__version__
            except AttributeError:
                _version = "unknown"
        logger.info(f"{package_name} version {_version} is available.")
    else:
        _version = "N/A"

    return _is_available, _version


def print_enviroment_info():
    _torch_available, _torch_version = get_package_info("torch")
    _torchvision_available, _torchvision_version = get_package_info("torchvision")
    _tensorflow_available, _tensorflow_version = get_package_info("tensorflow")
    _tensorflow_hub_available, _tensorflow_hub_version = get_package_info("tensorflow-hub")
    _yolov5_available, _yolov5_version = get_package_info("yolov5")
    _mmdet_available, _mmdet_version = get_package_info("mmdet")
    _mmcv_available, _mmcv_version = get_package_info("mmcv")
    _detectron2_available, _detectron2_version = get_package_info("detectron2")
    _transformers_available, _transformers_version = get_package_info("transformers")
    _timm_available, _timm_version = get_package_info("timm")
    _layer_available, _layer_version = get_package_info("layer")
    _fiftyone_available, _fiftyone_version = get_package_info("fiftyone")
    _norfair_available, _norfair_version = get_package_info("norfair")


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
