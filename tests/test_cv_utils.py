"""Tests for computer vision utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from sahi.utils.cv import (
    Colors,
    apply_color_mask,
    get_bbox_from_bool_mask,
    get_coco_segmentation_from_bool_mask,
    read_image,
    read_image_as_pil,
    read_image_size,
)


class TestReadImageSize:
    """Test read_image_size, which reads sizes from the header rather than decoding."""

    def _save(self, path: str, height: int, width: int, orientation: int | None = None) -> None:
        image = Image.fromarray(np.full((height, width, 3), 90, dtype=np.uint8))
        if orientation is None:
            image.save(path)
            return
        exif = Image.Exif()
        exif[0x0112] = orientation
        image.save(path, exif=exif)

    def test_matches_a_full_decode_for_a_plain_image(self, tmp_path: Path) -> None:
        """Test the header read agrees with a full decode."""
        path = str(tmp_path / "plain.png")
        self._save(path, height=30, width=40)

        assert read_image_size(path) == read_image_as_pil(path).size == (40, 30)

    @pytest.mark.parametrize("suffix", [".jpg", ".png", ".tif"])
    @pytest.mark.parametrize("orientation", [5, 6, 7, 8])
    def test_quarter_turn_exif_swaps_the_axes(self, tmp_path: Path, orientation: int, suffix: str) -> None:
        """Test EXIF orientations 5-8 swap the reported axes.

        These are quarter turns, which `read_image_as_pil` applies, so a header read
        alone reports them backwards. Containers differ in who applies the tag: PIL's
        TIFF plugin does it on load, so its header size already accounts for it.
        """
        path = str(tmp_path / f"rotated{suffix}")
        self._save(path, height=30, width=40, orientation=orientation)

        assert read_image_size(path) == read_image_as_pil(path).size == (30, 40)

    @pytest.mark.parametrize("suffix", [".jpg", ".png", ".tif"])
    @pytest.mark.parametrize("orientation", [1, 2, 3, 4])
    def test_non_rotating_exif_keeps_the_axes(self, tmp_path: Path, orientation: int, suffix: str) -> None:
        """Test EXIF orientations 2-4 keep the reported axes, being flips and 180s."""
        path = str(tmp_path / f"flipped{suffix}")
        self._save(path, height=30, width=40, orientation=orientation)

        assert read_image_size(path) == read_image_as_pil(path).size == (40, 30)

    def test_exif_is_ignored_when_the_caller_disables_the_fix(self, tmp_path: Path) -> None:
        """Test exif_fix=False reports the header size, no rotation applied."""
        path = str(tmp_path / "rotated.jpg")
        self._save(path, height=30, width=40, orientation=6)

        assert read_image_size(path, exif_fix=False) == (40, 30)

    @pytest.mark.parametrize("as_array", [True, False])
    def test_in_memory_images_report_their_own_size(self, as_array: bool) -> None:
        """Test already-decoded inputs, which have nothing to defer and no EXIF left."""
        pixels = np.full((30, 40, 3), 5, dtype=np.uint8)

        assert read_image_size(pixels if as_array else Image.fromarray(pixels)) == (40, 30)

    def test_chw_array_reports_the_transposed_size(self) -> None:
        """Test CHW arrays report the size read_image_as_pil's transpose produces."""
        chw = np.zeros((3, 30, 40), dtype=np.uint8)

        assert read_image_size(chw) == (40, 30)

    def test_falls_back_to_full_decode_when_the_header_read_fails(self, tmp_path: Path) -> None:
        """Test a file PIL cannot open falls back to read_image_as_pil's skimage path.

        The fallback has to be stubbed: `sahi.utils.cv.Image` is the PIL module itself, so
        patching `Image.open` breaks skimage's own reader too. The stub reports a size the
        file does not have, so a header read that quietly succeeded would fail the assert.
        """
        path = str(tmp_path / "odd.png")
        Image.fromarray(np.full((30, 40, 3), 9, dtype=np.uint8)).save(path)

        with patch("sahi.utils.cv.Image.open", side_effect=OSError("unsupported")):
            with patch("sahi.utils.cv.read_image_as_pil", return_value=Image.new("RGB", (7, 5))) as fallback:
                assert read_image_size(path) == (7, 5)

        fallback.assert_called_once()


class TestSixteenBitDecodeParity:
    """Test the cv2 fast path returns what the PIL path returns for the same file."""

    def test_return_arr_matches_pil_path_for_16bit_png(self, tmp_path: Path) -> None:
        """Test 16-bit files take the PIL path.

        cv2's IMREAD_COLOR rescales 16-bit samples where PIL clips I;16 at 255, and the
        two entry points must not disagree on the same file.
        """
        path = str(tmp_path / "deep.png")
        pixels = np.array([[55745, 41743, 33497, 120]], dtype=np.uint16)
        cv2.imwrite(path, pixels)

        via_arr = read_image_as_pil(path, return_arr=True)
        via_pil = np.asarray(read_image_as_pil(path))

        np.testing.assert_array_equal(via_arr, via_pil)


def pil_reference_decode(path: str, exif_fix: bool = True) -> np.ndarray:
    """Decode exactly as sahi did before the OpenCV fast path existed.

    Every other pixel test compares one current code path against another, and both now
    share the cv2 decode, so a regression in it would be invisible to all of them. This
    is the fixed reference they cannot drift away from together.
    """
    image = Image.open(path).convert("RGB")
    if exif_fix:
        ImageOps.exif_transpose(image, in_place=True)
    return np.asarray(image)


class TestDecoderParity:
    """Test the current decode against the pre-change PIL decode."""

    @pytest.mark.parametrize(
        ("mode", "suffix"),
        [
            (mode, suffix)
            for mode in ("RGB", "L", "1", "P", "LA", "RGBA")
            for suffix in (".png", ".tif", ".jpg")
            if not (suffix == ".jpg" and mode in ("P", "LA", "RGBA"))
        ],
    )
    def test_decode_matches_the_pil_reference(self, tmp_path: Path, mode: str, suffix: str) -> None:
        """Test each colour mode decodes to what the PIL path produced."""
        path = str(tmp_path / f"{mode}{suffix}")
        source = np.random.default_rng(21).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        Image.fromarray(source).convert(mode).save(path)

        result = read_image_as_pil(path, return_arr=True)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, pil_reference_decode(path))

    @pytest.mark.parametrize("suffix", [".jpg", ".tif"])
    def test_cmyk_decode_matches_the_pil_reference_within_rounding(self, tmp_path: Path, suffix: str) -> None:
        """Test CMYK matches the PIL reference to within a level.

        cv2 and PIL round the CMYK inversion differently, so this one is not exact. The
        bound still matters: libvips converts CMYK without the Adobe inversion at all,
        which is 127 levels out, not one.
        """
        path = str(tmp_path / f"cmyk{suffix}")
        source = np.random.default_rng(22).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        Image.fromarray(source).convert("CMYK").save(path)

        result = read_image_as_pil(path, return_arr=True)

        np.testing.assert_allclose(np.asarray(result, dtype=int), pil_reference_decode(path).astype(int), atol=1)

    @pytest.mark.parametrize("suffix", [".jpg", ".png", ".tif"])
    @pytest.mark.parametrize("orientation", [1, 2, 3, 4, 5, 6, 7, 8])
    @pytest.mark.parametrize("exif_fix", [True, False])
    def test_oriented_decode_matches_the_pil_reference(
        self, tmp_path: Path, orientation: int, suffix: str, exif_fix: bool
    ) -> None:
        """Test the EXIF orientation is applied exactly as the PIL path applied it."""
        path = str(tmp_path / f"rotated{suffix}")
        exif = Image.Exif()
        exif[0x0112] = orientation
        source = np.random.default_rng(23).integers(0, 255, (24, 32, 3), dtype=np.uint8)
        Image.fromarray(source).save(path, exif=exif)

        result = read_image_as_pil(path, return_arr=True, exif_fix=exif_fix)

        np.testing.assert_array_equal(result, pil_reference_decode(path, exif_fix=exif_fix))


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

    def test_read_image_as_pil_return_arr_matches_pil_path(self, tmp_path: Path) -> None:
        """Test the array fast path is pixel-identical to the PIL path, RGB order included.

        The fast path decodes via OpenCV, which yields BGR, and a missed channel swap
        degrades predictions silently rather than raising.
        """
        # asymmetric channels: a BGR/RGB mix-up cannot pass by coincidence
        expected = np.zeros((4, 6, 3), dtype=np.uint8)
        expected[..., 0], expected[..., 1], expected[..., 2] = 200, 120, 40
        path = str(tmp_path / "swatch.png")
        Image.fromarray(expected).save(path)

        result = read_image_as_pil(path, return_arr=True)

        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, expected)
        np.testing.assert_array_equal(result, np.asarray(Image.open(path).convert("RGB")))

    def test_read_image_as_pil_return_arr_falls_back_when_opencv_fails(self, tmp_path: Path) -> None:
        """Test a file OpenCV cannot decode falls back to the PIL path."""
        expected = np.full((3, 5, 3), 77, dtype=np.uint8)
        path = str(tmp_path / "fallback.png")
        Image.fromarray(expected).save(path)

        with patch("sahi.utils.cv.cv2.imread", return_value=None):
            result = read_image_as_pil(path, return_arr=True)

        np.testing.assert_array_equal(np.asarray(result), expected)

    def test_read_image_as_pil_return_arr_survives_opencv_pixel_limit(self, tmp_path: Path) -> None:
        """Test a cv2.imread that raises falls back rather than propagating.

        Images above CV_IO_MAX_IMAGE_PIXELS raise instead of returning None, and real
        gigapixel scans hit it (39266x29140 is 1.14e9 px against OpenCV's 2**30 cap).
        """
        expected = np.full((3, 5, 3), 42, dtype=np.uint8)
        path = str(tmp_path / "huge.png")
        Image.fromarray(expected).save(path)
        boom = cv2.error("OpenCV(5.0.0) ... (-215:Assertion failed) pixels <= CV_IO_MAX_IMAGE_PIXELS")

        with patch("sahi.utils.cv.cv2.imread", side_effect=boom):
            result = read_image_as_pil(path, return_arr=True)

        np.testing.assert_array_equal(np.asarray(result), expected)

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
