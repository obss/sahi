"""Tests for image slicing functionality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sahi.slicing import _slice_file_suffix, shift_bboxes, shift_masks, slice_coco, slice_image
from sahi.utils.coco import Coco
from sahi.utils.cv import read_image


class TestSlicing:
    """Test image slicing functionality."""

    def test_slice_image(self) -> None:
        """Test slicing an image with multiple input formats."""
        # read coco file
        coco_path = "tests/data/coco_utils/terrain1_coco.json"
        coco = Coco.from_coco_dict_or_path(coco_path)

        output_file_name = None
        output_dir = None
        image_path = "tests/data/coco_utils/" + coco.images[0].file_name
        slice_image_result = slice_image(
            image=image_path,
            coco_annotation_list=coco.images[0].annotations,
            output_file_name=output_file_name,
            output_dir=output_dir,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        assert len(slice_image_result) == 18
        assert len(slice_image_result.images) == 18
        assert len(slice_image_result.coco_images) == 18
        assert slice_image_result.coco_images[0].annotations == []
        assert slice_image_result.coco_images[15].annotations[1].area == 7296
        assert slice_image_result.coco_images[15].annotations[1].bbox == [17, 186, 48, 152]
        assert isinstance(slice_image_result[0], dict)
        assert slice_image_result[0]["image"].shape == (512, 512, 3)
        assert slice_image_result[3]["starting_pixel"] == [924, 0]  # type: ignore[call-overload]
        assert isinstance(slice_image_result[0:4], list)
        assert len(slice_image_result[0:4]) == 4

        image_cv = read_image(image_path)
        slice_image_result = slice_image(
            image=image_cv,
            coco_annotation_list=coco.images[0].annotations,
            output_file_name=output_file_name,
            output_dir=output_dir,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        assert len(slice_image_result.images) == 18
        assert len(slice_image_result.coco_images) == 18
        assert slice_image_result.coco_images[0].annotations == []
        assert slice_image_result.coco_images[15].annotations[1].area == 7296
        assert slice_image_result.coco_images[15].annotations[1].bbox == [17, 186, 48, 152]

        image_pil = Image.open(image_path)
        slice_image_result = slice_image(
            image=image_pil,
            coco_annotation_list=coco.images[0].annotations,
            output_file_name=output_file_name,
            output_dir=output_dir,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        assert len(slice_image_result.images) == 18
        assert len(slice_image_result.coco_images) == 18
        assert slice_image_result.coco_images[0].annotations == []
        assert slice_image_result.coco_images[15].annotations[1].area == 7296
        assert slice_image_result.coco_images[15].annotations[1].bbox == [17, 186, 48, 152]

    def test_slice_coco(self) -> None:
        """Test slicing COCO annotations and images."""
        import shutil

        coco_annotation_file_path = "tests/data/coco_utils/terrain1_coco.json"
        image_dir = "tests/data/coco_utils/"
        output_coco_annotation_file_name = "test_out"
        output_dir = "tests/data/coco_utils/test_out/"
        ignore_negative_samples = True
        coco_dict, _ = slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir=image_dir,
            output_coco_annotation_file_name=output_coco_annotation_file_name,
            output_dir=output_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        assert len(coco_dict["images"]) == 5
        assert coco_dict["images"][1]["height"] == 512
        assert coco_dict["images"][1]["width"] == 512
        assert len(coco_dict["annotations"]) == 14
        assert coco_dict["annotations"][2]["id"] == 3
        assert coco_dict["annotations"][2]["image_id"] == 2
        assert coco_dict["annotations"][2]["category_id"] == 1
        assert coco_dict["annotations"][2]["area"] == 12483
        assert coco_dict["annotations"][2]["bbox"] == [340, 204, 73, 171]

        shutil.rmtree(output_dir, ignore_errors=True)

        coco_annotation_file_path = "tests/data/coco_utils/terrain1_coco.json"
        image_dir = "tests/data/coco_utils/"
        output_coco_annotation_file_name = "test_out"
        output_dir = "tests/data/coco_utils/test_out/"
        ignore_negative_samples = False
        coco_dict, _ = slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir=image_dir,
            output_coco_annotation_file_name=output_coco_annotation_file_name,
            output_dir=output_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.1,
            overlap_width_ratio=0.4,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        assert len(coco_dict["images"]) == 18
        assert coco_dict["images"][1]["height"] == 512
        assert coco_dict["images"][1]["width"] == 512
        assert len(coco_dict["annotations"]) == 14
        assert coco_dict["annotations"][2]["id"] == 3
        assert coco_dict["annotations"][2]["image_id"] == 14
        assert coco_dict["annotations"][2]["category_id"] == 1
        assert coco_dict["annotations"][2]["area"] == 12483
        assert coco_dict["annotations"][2]["bbox"] == [340, 204, 73, 171]

        shutil.rmtree(output_dir, ignore_errors=True)

    def test_shift_bboxes(self) -> None:
        """Test shifting bounding boxes with different input types."""
        import torch

        bboxes = [[1, 2, 3, 4]]
        shift_x = 10
        shift_y = 20
        shifted_bboxes = shift_bboxes(bboxes=bboxes, offset=[shift_x, shift_y])
        assert shifted_bboxes == [[11, 22, 13, 24]]
        assert isinstance(shifted_bboxes, list)

        np_bboxes = np.array([[1, 2, 3, 4]])
        shifted_np_bboxes = shift_bboxes(bboxes=np_bboxes, offset=[shift_x, shift_y])
        assert shifted_np_bboxes.tolist() == [[11, 22, 13, 24]]
        assert isinstance(shifted_np_bboxes, np.ndarray)

        torch_bboxes = torch.tensor([[1, 2, 3, 4]])
        shifted_torch_bboxes = shift_bboxes(bboxes=torch_bboxes, offset=[shift_x, shift_y])
        assert shifted_torch_bboxes.tolist() == [[11, 22, 13, 24]]
        assert isinstance(shifted_torch_bboxes, torch.Tensor)

    @pytest.mark.parametrize(
        "image, out_ext, expected",
        [
            # a path is not a PIL image, so it carries no filename to inherit and exports png
            ("source.jpg", None, ".png"),
            ("source.bmp", None, ".png"),
            (np.zeros((4, 4, 3), dtype=np.uint8), None, ".png"),
            ("source.bmp", ".tif", ".tif"),
        ],
    )
    def test_slice_file_suffix(self, image: str | np.ndarray, out_ext: str | None, expected: str) -> None:
        """Exported slices keep the extension chosen before slicing switched to an array."""
        assert _slice_file_suffix(image, out_ext) == expected

    def test_slice_file_suffix_inherits_a_lossless_source_extension(self, tmp_path: Path) -> None:
        """An open PIL image is the only input carrying a filename, and a lossless one is kept."""
        bmp_path = tmp_path / "source.bmp"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(bmp_path)

        with Image.open(bmp_path) as image:
            assert _slice_file_suffix(image) == ".bmp"

    def test_slice_image_from_path_matches_array_input(self) -> None:
        """Slicing a path gives the same slices, and the same png export suffix, as slicing its array."""
        image_path = "tests/data/coco_utils/terrain1.jpg"
        kwargs = dict(
            slice_height=512, slice_width=512, overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=False
        )

        from_path = slice_image(image=image_path, **kwargs)  # type: ignore[arg-type]
        from_array = slice_image(image=read_image(image_path), **kwargs)  # type: ignore[arg-type]

        assert len(from_path) == len(from_array)
        for path_slice, array_slice in zip(from_path.images, from_array.images):
            np.testing.assert_array_equal(path_slice, array_slice)
        # a lossy source exports as png, so repeated slicing does not compound the compression
        assert all(name.endswith(".png") for name in from_path.filenames)

    def test_shift_masks(self) -> None:
        """Test shifting mask arrays."""
        masks = np.zeros((3, 30, 30), dtype=bool)
        shift_x = 10
        shift_y = 20
        full_shape = [720, 1280]
        shifted_masks = shift_masks(masks=masks, offset=[shift_x, shift_y], full_shape=full_shape)
        assert shifted_masks.shape == (3, 720, 1280)
        assert isinstance(shifted_masks, np.ndarray)
