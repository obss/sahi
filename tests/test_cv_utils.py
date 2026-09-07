"""Tests for computer vision utility functions."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from sahi.utils.cv import (
    IMAGE_EXTENSIONS_LOSSY,
    Colors,
    apply_color_mask,
    get_bbox_from_bool_mask,
    get_coco_segmentation_from_bool_mask,
    get_video_reader,
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

_PIXELS = np.random.default_rng(3).integers(0, 256, (37, 53, 3), dtype=np.uint8)

# One entry per mode and container the array decode has to agree with PIL on. The
# deep ones are the reason the header is read first: their dtype only shows up
# after a full decode, which is the cost this is all trying to avoid.
DECODER_FIXTURES = {
    "rgb.jpg": lambda: Image.fromarray(_PIXELS),
    "gray.jpg": lambda: Image.fromarray(_PIXELS[..., 0]),
    "cmyk.jpg": lambda: Image.fromarray(_PIXELS).convert("CMYK"),
    "rgb.png": lambda: Image.fromarray(_PIXELS),
    "rgba.png": lambda: Image.fromarray(np.dstack([_PIXELS, _PIXELS[..., :1]])),
    "palette.png": lambda: Image.fromarray(_PIXELS).convert("P", palette=Image.Palette.ADAPTIVE),
    "luma_alpha.png": lambda: Image.fromarray(_PIXELS[..., 0]).convert("LA"),
    "bilevel.png": lambda: Image.fromarray(_PIXELS[..., 0] > 127),
    "deep.png": lambda: Image.fromarray(_PIXELS[..., 0].astype(np.uint16) * 257),
    "rgb.bmp": lambda: Image.fromarray(_PIXELS),
    "rgb.webp": lambda: Image.fromarray(_PIXELS),
    "rgb.tif": lambda: Image.fromarray(_PIXELS),
    "deep.tif": lambda: Image.fromarray(_PIXELS[..., 0].astype(np.uint16) * 257),
}


def assert_decode_parity(image_path: str | Path, exif_fix: bool = True) -> None:
    """The array decode must give the pixels the PIL decode would have given."""
    from_pil = np.asarray(read_image_as_pil(str(image_path), exif_fix=exif_fix))
    as_arr = read_image_as_pil(str(image_path), exif_fix=exif_fix, return_arr=True)

    assert as_arr.dtype == from_pil.dtype
    assert as_arr.shape == from_pil.shape
    # decoders may round a lossy sample differently; a channel swap would be off by far more
    tolerance = 1 if Path(str(image_path)).suffix in IMAGE_EXTENSIONS_LOSSY else 0
    assert np.abs(as_arr.astype(int) - from_pil.astype(int)).max() <= tolerance


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
        assert_decode_parity(image_path)

    @pytest.mark.parametrize("name", list(DECODER_FIXTURES))
    @pytest.mark.parametrize("exif_fix", [True, False])
    def test_array_decode_matches_pil_decode_per_mode(self, tmp_path: Path, name: str, exif_fix: bool) -> None:
        """Every mode and container either decodes identically or falls back to PIL."""
        image_path = tmp_path / name
        DECODER_FIXTURES[name]().save(image_path)
        assert_decode_parity(image_path, exif_fix=exif_fix)

    @pytest.mark.parametrize("suffix", [".jpg", ".tif"])
    @pytest.mark.parametrize("orientation", range(1, 9))
    @pytest.mark.parametrize("exif_fix", [True, False])
    def test_array_decode_matches_pil_decode_per_orientation(
        self, tmp_path: Path, suffix: str, orientation: int, exif_fix: bool
    ) -> None:
        """Both decoders must land on the same pixels for every EXIF orientation."""
        image_path = tmp_path / f"rotated{suffix}"
        image = Image.fromarray(_PIXELS)
        exif = image.getexif()
        exif[0x0112] = orientation
        image.save(image_path, exif=exif)

        assert_decode_parity(image_path, exif_fix=exif_fix)


def write_test_video(path: Path, num_frames: int, fps: float, size: tuple[int, int] = (64, 64)) -> None:
    """Write a solid-colour mp4 with an exactly known frame count and fps."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for frame_ind in range(num_frames):
        writer.write(np.full((size[1], size[0], 3), frame_ind % 256, dtype=np.uint8))
    writer.release()


class TestGetVideoReader:
    """`get_video_reader` has to account for the frames `frame_skip_interval` drops."""

    SOURCE_FPS = 30.0

    @pytest.mark.parametrize("frame_skip_interval", [0, 1, 3])
    @pytest.mark.parametrize("source_frames", [60, 61])
    def test_reported_frame_count_matches_frames_yielded(
        self, tmp_path: Path, source_frames: int, frame_skip_interval: int
    ) -> None:
        """The count handed to tqdm must be what the generator goes on to yield.

        Each iteration seeks ahead by frame_skip_interval and then reads one more, so
        the caller sees one frame per frame_skip_interval + 1 of the source. The count
        used to be corrected only while a visual was being rendered, which left the
        progress bar short by that factor on every headless run.
        """
        source = tmp_path / "source.mp4"
        write_test_video(source, source_frames, self.SOURCE_FPS)
        save_dir = tmp_path / "export"
        save_dir.mkdir()

        frames, _, _, num_frames = get_video_reader(
            str(source), str(save_dir), frame_skip_interval, export_visual=False, view_visual=False
        )

        assert num_frames == source_frames // (frame_skip_interval + 1)
        assert sum(1 for _ in frames) == num_frames

    @pytest.mark.parametrize("frame_skip_interval", [0, 1, 3])
    def test_exported_video_keeps_the_source_duration(self, tmp_path: Path, frame_skip_interval: int) -> None:
        """Dropping frames has to lower the export fps by the same factor.

        Otherwise the kept frames play back over a shorter span than they were
        captured in, and the export runs fast by frame_skip_interval + 1.
        """
        source_frames = 60
        source = tmp_path / "source.mp4"
        write_test_video(source, source_frames, self.SOURCE_FPS)
        save_dir = tmp_path / "export"
        save_dir.mkdir()

        frames, video_writer, _, _ = get_video_reader(
            str(source), str(save_dir), frame_skip_interval, export_visual=True, view_visual=False
        )
        assert video_writer is not None
        for frame in frames:
            video_writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        video_writer.release()

        exported = cv2.VideoCapture(str(save_dir / "source.mp4"))
        exported_fps = exported.get(cv2.CAP_PROP_FPS)
        exported_frames = int(exported.get(cv2.CAP_PROP_FRAME_COUNT))
        exported.release()

        assert exported_fps == pytest.approx(self.SOURCE_FPS / (frame_skip_interval + 1))
        assert exported_frames / exported_fps == pytest.approx(source_frames / self.SOURCE_FPS, rel=0.05)
