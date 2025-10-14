from __future__ import annotations

import importlib.util
import platform
from collections.abc import Generator
from typing import Any

from sahi.logger import logger

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py


# Mapping from package names to their import names
PACKAGE_TO_MODULE_MAP = {
    "opencv-python": "cv2",
    "opencv-python-headless": "cv2",
    "pillow": "PIL",
}


def get_package_info(package_name: str) -> tuple[bool, str]:
    """Returns the package version as a string and the package name as a string."""

    if package_name not in PACKAGE_TO_MODULE_MAP:
        module_name = package_name
    else:
        module_name = PACKAGE_TO_MODULE_MAP.get(package_name, package_name)

    _is_available = is_available(module_name)

    if _is_available:
        try:
            _version = importlib.import_module(module_name).__version__
        except (ModuleNotFoundError, AttributeError):
            try:
                _version = importlib.import_module(package_name).__version__
            except AttributeError:
                _version = "unknown"
    else:
        _version = "N/A"

    logger.pkg_info(f"{package_name} version {_version} is installed.")
    return _is_available, _version


def sys_info():
    logger.pkg_info("System Information:")
    logger.pkg_info(f"Python version: {platform.python_version()}")
    logger.pkg_info(f"Platform: {platform.platform().capitalize()}")
    logger.pkg_info(f"Processor: {platform.processor().capitalize()}")
    logger.pkg_info(f"Machine: {platform.machine().capitalize()}")
    logger.pkg_info(f"System: {platform.system().capitalize()}")
    logger.pkg_info(f"Release: {platform.release().capitalize()}")
    logger.pkg_info(f"Version: {platform.version().capitalize()}")
    logger.pkg_info(f"Architecture: {platform.architecture()[0].capitalize()}")


def print_environment_info() -> None:
    sys_info()
    logger.pkg_info("=== Package Information ===")
    get_package_info("torch")
    get_package_info("torchvision")
    get_package_info("tensorflow")
    get_package_info("tensorflow-hub")
    get_package_info("ultralytics")
    get_package_info("yolov5")
    get_package_info("mmdet")
    get_package_info("mmcv")
    get_package_info("detectron2")
    get_package_info("transformers")
    get_package_info("timm")
    get_package_info("fiftyone")
    get_package_info("pillow")
    get_package_info("opencv-python")


def is_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def check_requirements(package_names: list[str]) -> Generator[None, Any, Any]:
    """Raise error if module is not installed."""
    missing_packages = []
    for package_name in package_names:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)
    if missing_packages:
        raise ImportError(f"The following packages are required to use this module: {missing_packages}")
    yield


def _parse_version(version_str: str) -> tuple[int, ...]:
    """Simple version parser that converts '1.2.3' to (1, 2, 3) for comparison."""
    try:
        return tuple(int(x) for x in version_str.split("."))
    except (ValueError, AttributeError):
        return (0,)  # Default to 0.0 for unparseable versions


def check_package_minimum_version(package_name: str, minimum_version: str, raise_error: bool = False) -> bool:
    """Check if module version meets minimum requirement.

    Args:
        package_name: Name of the package to check
        minimum_version: Minimum required version (e.g., '1.2.3')
        raise_error: If True, raises ImportError when version is too low

    Returns:
        True if version is compatible, False otherwise

    Raises:
        ImportError: If raise_error=True and version is incompatible
    """
    _is_available, _version = get_package_info(package_name)

    if not _is_available:
        if raise_error:
            raise ImportError(f"Package {package_name} is not installed.")
        return False

    if _version == "unknown":
        logger.warning(
            f"Could not determine version of {package_name}. Assuming version {minimum_version} is compatible."
        )
        return True

    if _version == "N/A":
        if raise_error:
            raise ImportError(f"Package {package_name} is not available.")
        return False

    # Compare versions using simple tuple comparison
    current_version = _parse_version(_version)
    required_version = _parse_version(minimum_version)

    is_compatible = current_version >= required_version

    if not is_compatible and raise_error:
        raise ImportError(
            f"Please upgrade {package_name} to version {minimum_version} or higher. Current version: {_version}"
        )

    return is_compatible
