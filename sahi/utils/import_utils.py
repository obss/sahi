"""Import utilities for checking package availability."""

import importlib.util
from collections.abc import Iterable

from sahi.logger import logger

# adapted from https://github.com/huggingface/transformers/blob/main/src/transformers/utils/import_utils.py


def get_package_info(package_name: str, verbose: bool = True) -> tuple[bool, str]:
    """Check whether a package is installed and retrieve its version.

    Args:
        package_name: The distribution name (e.g. "pillow") or module name (e.g. "torch") of the package.
        verbose: If True, log the package version when available.

    Returns:
        is_available (bool): Whether the package is installed.
        version_string (str): Version string, or "N/A" if not installed.
    """
    import importlib.metadata as _importlib_metadata

    try:
        _version = _importlib_metadata.version(package_name)
        _is_available = True
    except _importlib_metadata.PackageNotFoundError:
        _is_available = is_available(package_name)
        _version = "N/A"
        if _is_available:
            _version = getattr(importlib.import_module(package_name), "__version__", "unknown")

    if _is_available and verbose:
        logger.pkg_info(f"{package_name} version {_version} is available.")

    return _is_available, _version


def print_environment_info() -> None:
    """Log version info for all commonly used SAHI dependency packages."""
    for package_name in (
        "torch",
        "torchvision",
        "ultralytics",
        "yolov5",
        "mmdet",
        "mmcv",
        "detectron2",
        "transformers",
        "timm",
        "fiftyone",
        "pillow",
    ):
        get_package_info(package_name)
    for distribution, version in get_opencv_distribution_versions().items():
        logger.pkg_info(f"{distribution} version {version} is available.")


OPENCV_DISTRIBUTIONS = (
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python",
    "opencv-contrib-python-headless",
)


def get_opencv_distribution_versions() -> dict[str, str]:
    """Collect the installed versions of every OpenCV distribution.

    Returns:
        Mapping of distribution name to version, for those that are installed.
    """
    import importlib.metadata as _importlib_metadata

    versions = {}
    for distribution in OPENCV_DISTRIBUTIONS:
        try:
            versions[distribution] = _importlib_metadata.version(distribution)
        except _importlib_metadata.PackageNotFoundError:
            continue
    return versions


def get_opencv_conflict_message() -> str | None:
    """Describe an OpenCV installation that mixes distribution versions.

    All OpenCV distributions install into the same ``cv2`` directory, so
    installing more than one of them at different versions leaves a mixture of
    Python and native files behind, and ``import cv2`` fails with a confusing
    error such as ``partially initialized module 'cv2' has no attribute
    'gapi_wip_gst_GStreamerPipeline'``.

    Returns:
        A message explaining how to fix the installation, or None when the
            installed OpenCV distributions agree on a single version.
    """
    versions = get_opencv_distribution_versions()
    if len(set(versions.values())) < 2:
        return None

    from packaging import version as version_parser

    installed = ", ".join(f"{name}=={version}" for name, version in versions.items())
    newest = max(versions.values(), key=version_parser.parse)
    suggestion = " ".join(f"{name}=={newest}" for name in versions)
    return (
        f"Conflicting OpenCV installations detected: {installed}. They all install into the same 'cv2' "
        "directory, so mixing versions leaves a broken 'cv2' package behind. Keep a single OpenCV "
        f"distribution, or reinstall all of them at the same version, e.g. "
        f"`pip install --force-reinstall {suggestion}`."
    )


def is_available(module_name: str) -> bool:
    """Check whether a Python module is importable.

    Args:
        module_name: Dotted module name (e.g. "torch", "torchvision").

    Returns:
        True if the module can be found by the import system.
    """
    return importlib.util.find_spec(module_name) is not None


def check_requirements(package_names: Iterable[str]) -> None:
    """Verify that all required packages are importable.

    Args:
        package_names: Iterable of package names to check.

    Raises:
        ImportError: If any of the listed packages cannot be found.
    """
    missing_packages = []
    for package_name in package_names:
        if importlib.util.find_spec(package_name) is None:
            missing_packages.append(package_name)
    if missing_packages:
        raise ImportError(f"The following packages are required to use this module: {missing_packages}")


def check_package_minimum_version(package_name: str, minimum_version: str, verbose: bool = False) -> bool:
    """Check whether an installed package meets a minimum version requirement.

    Args:
        package_name: The name of the package to check.
        minimum_version: The minimum acceptable version string (e.g. "1.0.0").
        verbose: If True, log the detected package version.

    Returns:
        True if the package is missing (assumed compatible), its version
            is unknown, or its version meets the minimum. False if the
            installed version is below the minimum.
    """
    from packaging import version

    _is_available, _version = get_package_info(package_name, verbose=verbose)
    if _is_available:
        if _version == "unknown":
            logger.warning(
                f"Could not determine version of {package_name}. Assuming version {minimum_version} is compatible."
            )
        else:
            if version.parse(_version) < version.parse(minimum_version):
                return False
    return True


def ensure_package_minimum_version(package_name: str, minimum_version: str, verbose: bool = False) -> None:
    """Ensure a package meets a minimum version, raising on failure.

    Args:
        package_name: The name of the package to check.
        minimum_version: The minimum acceptable version string (e.g. "1.0.0").
        verbose: If True, log the detected package version.

    Raises:
        ImportError: If the installed version is below minimum_version.
    """
    if not check_package_minimum_version(package_name, minimum_version, verbose=verbose):
        raise ImportError(f"Please upgrade {package_name} to version {minimum_version} or higher to use this module.")
