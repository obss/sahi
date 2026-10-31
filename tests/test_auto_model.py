"""Tests for the AutoDetectionModel registry."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import sahi.models
from sahi.auto_model import MODEL_TYPE_TO_MODEL_CLASS_NAME

MODELS_DIR = Path(sahi.models.__file__).parent


@pytest.mark.parametrize(("model_type", "class_name"), sorted(MODEL_TYPE_TO_MODEL_CLASS_NAME.items()))
def test_model_type_points_at_a_defined_class(model_type: str, class_name: str) -> None:
    """Every registered model type has a module that defines its class, without importing the framework."""
    module_path = MODELS_DIR / f"{model_type}.py"
    assert module_path.is_file(), f"{model_type} has no module at {module_path}"

    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    assert class_name in defined, f"{module_path.name} does not define {class_name}"
