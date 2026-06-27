from __future__ import annotations

import inspect
from collections.abc import Callable

import pytest

from sahi.utils import import_utils
from sahi.utils.import_utils import (
    check_package_minimum_version,
    check_requirements,
    ensure_package_minimum_version,
    get_package_info,
    is_available,
)

MISSING_PACKAGE = "this_package_definitely_does_not_exist_12345"


@pytest.fixture
def fake_package(monkeypatch: pytest.MonkeyPatch) -> Callable[[bool, str], None]:
    """Stub get_package_info so version checks don't depend on real installs."""

    def _set(available: bool, version: str) -> None:
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (available, version))

    return _set


@pytest.mark.parametrize("func", [check_requirements, ensure_package_minimum_version])
def test_helpers_are_not_generators(func: Callable) -> None:
    """A trailing yield would silently disable these checks; guard against it."""
    assert not inspect.isgeneratorfunction(func)


class TestPackageInfoHelpers:
    def test_is_available(self) -> None:
        assert is_available("numpy") is True
        assert is_available(MISSING_PACKAGE) is False

    def test_get_package_info_installed(self) -> None:
        available, version = get_package_info("numpy", verbose=False)
        assert available is True
        assert version not in ("N/A", "")

    def test_get_package_info_missing(self) -> None:
        assert get_package_info(MISSING_PACKAGE, verbose=False) == (False, "N/A")


class TestCheckRequirements:
    def test_passes_when_all_present(self) -> None:
        assert check_requirements(["numpy"]) is None

    def test_passes_for_empty_iterable(self) -> None:
        assert check_requirements([]) is None

    def test_raises_only_for_missing(self) -> None:
        """A present package mixed with a missing one raises naming only the missing one."""
        with pytest.raises(ImportError) as exc_info:
            check_requirements(["numpy", MISSING_PACKAGE])
        message = str(exc_info.value)
        assert MISSING_PACKAGE in message
        assert "numpy" not in message


# (available, version, min_version, satisfied) — satisfied=False only when known and too low.
VERSION_CASES = [
    pytest.param(True, "2.0.0", "1.0.0", True, id="satisfied"),
    pytest.param(True, "1.0.0", "2.0.0", False, id="too-low"),
    pytest.param(False, "N/A", "2.0.0", True, id="absent"),
    pytest.param(True, "unknown", "2.0.0", True, id="unknown"),
]


@pytest.mark.parametrize("available, version, min_version, satisfied", VERSION_CASES)
def test_check_package_minimum_version(
    fake_package: Callable[[bool, str], None],
    available: bool,
    version: str,
    min_version: str,
    satisfied: bool,
) -> None:
    fake_package(available, version)
    assert check_package_minimum_version("somepkg", min_version) is satisfied


@pytest.mark.parametrize("available, version, min_version, satisfied", VERSION_CASES)
def test_ensure_package_minimum_version(
    fake_package: Callable[[bool, str], None],
    available: bool,
    version: str,
    min_version: str,
    satisfied: bool,
) -> None:
    fake_package(available, version)
    if satisfied:
        assert ensure_package_minimum_version("somepkg", min_version) is None
    else:
        with pytest.raises(ImportError, match="somepkg"):
            ensure_package_minimum_version("somepkg", min_version)
