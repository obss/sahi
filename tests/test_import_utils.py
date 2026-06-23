"""Tests for sahi.utils.import_utils.

These tests guard against a regression where ``check_requirements`` and
``ensure_package_minimum_version`` were accidentally written as generator
functions (a trailing ``yield``). Because the callers never iterate the
returned object, the function bodies never executed and the intended
``ImportError`` was silently never raised. The ``TestNotGenerators`` cases
lock in the correct (non-generator) behavior.
"""

from __future__ import annotations

import inspect

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


class TestNotGenerators:
    """Regression guards: these helpers must not be generator functions."""

    def test_check_requirements_is_not_a_generator(self) -> None:
        """check_requirements should be a regular function, not a generator."""
        assert not inspect.isgeneratorfunction(check_requirements)

    def test_ensure_package_minimum_version_is_not_a_generator(self) -> None:
        """ensure_package_minimum_version should be a regular function, not a generator."""
        assert not inspect.isgeneratorfunction(ensure_package_minimum_version)


class TestPackageInfoHelpers:
    """Tests for is_available and get_package_info."""

    def test_is_available_true_for_installed_package(self) -> None:
        """An installed package is reported as available."""
        assert is_available("numpy") is True

    def test_is_available_false_for_missing_package(self) -> None:
        """A missing package is reported as unavailable."""
        assert is_available(MISSING_PACKAGE) is False

    def test_get_package_info_for_installed_package(self) -> None:
        """get_package_info returns availability and a real version string."""
        available, version = get_package_info("numpy", verbose=False)
        assert available is True
        assert version not in ("N/A", "")

    def test_get_package_info_for_missing_package(self) -> None:
        """get_package_info reports a missing package with an N/A version."""
        available, version = get_package_info(MISSING_PACKAGE, verbose=False)
        assert available is False
        assert version == "N/A"


class TestCheckRequirements:
    """Tests for check_requirements."""

    def test_passes_when_all_present(self) -> None:
        """No error is raised when every package is importable."""
        assert check_requirements(["numpy"]) is None

    def test_raises_for_missing_package(self) -> None:
        """A missing package raises ImportError naming the package."""
        with pytest.raises(ImportError) as exc_info:
            check_requirements([MISSING_PACKAGE])
        assert MISSING_PACKAGE in str(exc_info.value)

    def test_raises_when_any_missing(self) -> None:
        """A present package mixed with a missing one still raises for the missing one."""
        with pytest.raises(ImportError) as exc_info:
            check_requirements(["numpy", MISSING_PACKAGE])
        message = str(exc_info.value)
        assert MISSING_PACKAGE in message
        assert "numpy" not in message

    def test_passes_for_empty_iterable(self) -> None:
        """An empty requirement list raises nothing."""
        assert check_requirements([]) is None


class TestEnsurePackageMinimumVersion:
    """Tests for ensure_package_minimum_version (raises on too-low version)."""

    def test_passes_when_version_satisfied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An installed package at or above the minimum version raises nothing."""
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (True, "2.0.0"))
        assert ensure_package_minimum_version("somepkg", "1.0.0") is None

    def test_raises_when_version_too_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An installed package below the minimum version raises ImportError."""
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (True, "1.0.0"))
        with pytest.raises(ImportError) as exc_info:
            ensure_package_minimum_version("somepkg", "2.0.0")
        assert "somepkg" in str(exc_info.value)

    def test_passes_when_package_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A version requirement is not enforced on an absent package."""
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (False, "N/A"))
        assert ensure_package_minimum_version("somepkg", "2.0.0") is None

    def test_passes_when_version_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An unknown installed version is assumed compatible (warns, does not raise)."""
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (True, "unknown"))
        assert ensure_package_minimum_version("somepkg", "2.0.0") is None


class TestCheckPackageMinimumVersion:
    """Tests for check_package_minimum_version (returns bool, never raises)."""

    def test_true_when_satisfied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True when the installed version meets the minimum."""
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (True, "2.0.0"))
        assert check_package_minimum_version("somepkg", "1.0.0") is True

    def test_false_when_too_low(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns False when the installed version is below the minimum."""
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (True, "1.0.0"))
        assert check_package_minimum_version("somepkg", "2.0.0") is False

    def test_true_when_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True (assumed compatible) when the package is absent."""
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (False, "N/A"))
        assert check_package_minimum_version("somepkg", "2.0.0") is True

    def test_true_when_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns True (assumed compatible) when the version is unknown."""
        monkeypatch.setattr(import_utils, "get_package_info", lambda name, verbose=False: (True, "unknown"))
        assert check_package_minimum_version("somepkg", "2.0.0") is True
