"""Tests for prediction module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sahi.prediction
import sahi.utils.cv
from sahi.prediction import PredictionResult, PredictionScore


class TestPrediction:
    """Test cases for prediction functionality."""

    def test_prediction_score(self) -> None:
        """Test PredictionScore value and comparison operations."""
        prediction_score = PredictionScore(np.array(0.6))
        assert isinstance(prediction_score.value, float)
        assert prediction_score.is_greater_than_threshold(0.5)
        assert not prediction_score.is_greater_than_threshold(0.7)


class TestPredictionResultImage:
    """Test the source image is decoded only when its pixels are asked for.

    Sliced prediction streams a gigapixel scan a band at a time so the whole image is
    never resident, and decoding it here to read `.size` would undo that.
    """

    @pytest.fixture
    def image_path(self, tmp_path: Path) -> str:
        path = str(tmp_path / "scan.png")
        Image.fromarray(np.full((30, 40, 3), 77, dtype=np.uint8)).save(path)
        return path

    def test_dimensions_do_not_decode_the_image(self, image_path: str, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test width and height come from the header, with no decode."""
        calls: list[object] = []
        real = sahi.prediction.read_image_as_pil
        monkeypatch.setattr(
            sahi.prediction,
            "read_image_as_pil",
            lambda image, *args, **kwargs: (calls.append(image), real(image, *args, **kwargs))[1],
        )

        result = PredictionResult(object_prediction_list=[], image=image_path)

        assert (result.image_width, result.image_height) == (40, 30)
        assert calls == [], "constructing a PredictionResult must not decode the image"

    def test_image_still_returns_the_source_pixels(self, image_path: str) -> None:
        """Test `.image` returns what an eager decode would have returned."""
        result = PredictionResult(object_prediction_list=[], image=image_path)

        np.testing.assert_array_equal(np.asarray(result.image), np.full((30, 40, 3), 77, dtype=np.uint8))

    def test_image_is_decoded_only_once(self, image_path: str) -> None:
        """Test the decoded image is cached across repeated access."""
        result = PredictionResult(object_prediction_list=[], image=image_path)

        assert result.image is result.image

    def test_missing_file_still_fails_at_construction(self, tmp_path: Path) -> None:
        """Test a bad path still raises at construction rather than on first access."""
        with pytest.raises(FileNotFoundError):
            PredictionResult(object_prediction_list=[], image=str(tmp_path / "nope.png"))

    def test_url_source_is_fetched_exactly_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test a remote image is downloaded once.

        There is no header to peek at, so sizing it costs a full download and that
        download is kept rather than repeated on the first `.image` access.
        """
        calls: list[object] = []
        fetched = Image.new("RGB", (40, 30))

        def fake_fetch(image: object, *args: object, **kwargs: object) -> Image.Image:
            calls.append(image)
            return fetched

        monkeypatch.setattr(sahi.prediction, "read_image_as_pil", fake_fetch)
        monkeypatch.setattr(sahi.utils.cv, "read_image_as_pil", fake_fetch)

        result = PredictionResult(object_prediction_list=[], image="https://example.com/scan.jpg")

        assert (result.image_width, result.image_height) == (40, 30)
        assert result.image is fetched
        assert len(calls) == 1, f"expected one fetch, saw {len(calls)}"

    @pytest.mark.parametrize("as_array", [True, False])
    def test_accepts_in_memory_images(self, as_array: bool) -> None:
        """Test arrays and Pillow images, which are already decoded, keep working."""
        pixels = np.full((30, 40, 3), 12, dtype=np.uint8)
        image = pixels if as_array else Image.fromarray(pixels)

        result = PredictionResult(object_prediction_list=[], image=image)

        assert (result.image_width, result.image_height) == (40, 30)
        np.testing.assert_array_equal(np.asarray(result.image), pixels)

    def test_image_can_be_replaced(self, image_path: str) -> None:
        """Test `.image` is still assignable, as it was before it became lazy."""
        result = PredictionResult(object_prediction_list=[], image=image_path)
        replacement = Image.fromarray(np.full((10, 20, 3), 5, dtype=np.uint8))

        result.image = replacement

        assert result.image is replacement
