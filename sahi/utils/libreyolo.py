"""LibreYOLO model utilities and constants."""

from __future__ import annotations

import urllib.request
from os import path
from pathlib import Path


class LibreYoloTestConstants:
    """LibreYOLO test model configurations."""

    LIBREYOLO9T_MODEL_URL = (
        "https://huggingface.co/LibreYOLO/LibreYOLO9t/resolve/main/LibreYOLO9t.pt"
    )
    LIBREYOLO9T_MODEL_PATH = "tests/data/models/libreyolo/LibreYOLO9t.pt"


def download_libreyolo9t_model(destination_path: str | None = None) -> None:
    """Download the LibreYOLO9t model for testing."""
    if destination_path is None:
        destination_path = LibreYoloTestConstants.LIBREYOLO9T_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            LibreYoloTestConstants.LIBREYOLO9T_MODEL_URL,
            destination_path,
        )
