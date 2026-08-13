"""Tests for computer vision utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from sahi.utils.cv import (
    IMAGE_EXTENSIONS_LOSSY,
    Colors,
    apply_color_mask,
    get_bbox_from_bool_mask,
    get_coco_segmentation_from_bool_mask,
    read_image,
    read_image_as_pil,
    read_image_size,
)

LOCAL_IMAGES = [
    "tests/data/coco_utils/terrain1.jpg",
    "tests/data/coco_utils/terrain2.png",
    "tests/data/coco_utils/terrain2_gray.png",
    "tests/data/small-vehicles1.jpeg",
]


class TestCvUtils:
    """Test cases for CV utility functions."""

    def test_hex_to_rgb(self) -> None:
        """Test hex to RGB color conversion."""
        colors = Colors()
        assert colors.hex_to_rgb("#FF3838") == (255, 56, 56)

    def test_hex_to_rgb_retrieve(self) -> None:
        """Test Colors class color retrieval."""
        colors = Colors()
        assert colors(0) == (255, 56, 56)

    @patch("sahi.utils.cv.cv2.cvtColor")
    @patch("sahi.utils.cv.cv2.imread")
    def test_read_image(self, mock_imread: MagicMock, mock_cvtColor: MagicMock) -> None:
        """Test image reading with mocked cv2."""
        fake_image = "test.jpg"
        fake_image_val = np.array([[[10, 20, 30]]], dtype=np.uint8)
        fake_image_rbg_val = np.array([[[10, 20, 30]]], dtype=np.uint8)
        mock_imread.return_value = fake_image_val
        mock_cvtColor.return_value = fake_image_rbg_val

        result = read_image(fake_image)

        # mock_cv2.assert_called_once_with(fake_image)
        mock_imread.assert_called_once_with(fake_image)
        np.testing.assert_array_equal(result, fake_image_rbg_val)

    def test_apply_color_mask(self) -> None:
        """Test applying color mask to image."""
        image = np.array([[0, 1]], dtype=np.uint8)
        color = (255, 0, 0)

        expected_output = np.array([[[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)

        result = apply_color_mask(image, color)

        np.testing.assert_array_equal(result, expected_output)

    def test_get_coco_segmentation_from_bool_mask_simple(self) -> None:
        """Test COCO segmentation from empty boolean mask."""
        mask = np.zeros((10, 10), dtype=bool)
        result = get_coco_segmentation_from_bool_mask(mask)
        assert result == []

    def test_get_coco_segmentation_from_bool_mask_polygon(self) -> None:
        """Test COCO segmentation from boolean mask with polygons."""
        mask = np.zeros((10, 20), dtype=bool)
        mask[1:4, 1:4] = True
        mask[5:8, 5:8] = True
        result = get_coco_segmentation_from_bool_mask(mask)
        assert len(result) == 2

    def test_get_bbox_from_bool_mask(self) -> None:
        """Test bounding box extraction from boolean mask."""
        mask = np.array(
            [
                [False, False, False],
                [False, True, True],
                [False, True, True],
                [False, False, False],
            ]
        )
        expected_result = [1, 1, 2, 2]
        result = get_bbox_from_bool_mask(mask)
        assert result == expected_result

    @pytest.mark.parametrize("image_path", LOCAL_IMAGES)
    def test_read_image_size_matches_full_decode(self, image_path: str) -> None:
        """Reading the size from the header agrees with decoding the whole image."""
        assert read_image_size(image_path) == read_image_as_pil(image_path).size

    @pytest.mark.parametrize("image_format, suffix", [("JPEG", ".jpg"), ("TIFF", ".tif")])
    @pytest.mark.parametrize("exif_fix", [True, False])
    def test_read_image_size_honors_exif_orientation(
        self, tmp_path: Path, image_format: str, suffix: str, exif_fix: bool
    ) -> None:
        """An orientation tag that turns the image must swap the reported width and height."""
        image_path = tmp_path / f"rotated{suffix}"
        image = Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8))  # 40 wide, 20 tall
        exif = image.getexif()
        exif[0x0112] = 6  # rotate a quarter turn
        image.save(image_path, format=image_format, exif=exif)

        assert read_image_size(image_path, exif_fix=exif_fix) == read_image_as_pil(image_path, exif_fix=exif_fix).size

    @pytest.mark.parametrize("image_path", LOCAL_IMAGES)
    def test_read_image_as_pil_return_arr_matches_pil_decode(self, image_path: str) -> None:
        """Decoding straight to an array gives the same pixels as converting from PIL."""
        from_pil = np.asarray(read_image_as_pil(image_path))
        as_arr = read_image_as_pil(image_path, return_arr=True)

        assert as_arr.dtype == np.uint8
        assert as_arr.shape == from_pil.shape
        # decoders may round a lossy sample differently; a channel swap would be off by far more
        tolerance = 1 if Path(image_path).suffix in IMAGE_EXTENSIONS_LOSSY else 0
        assert np.abs(as_arr.astype(int) - from_pil.astype(int)).max() <= tolerance
