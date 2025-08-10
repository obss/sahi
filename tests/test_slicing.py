import numpy as np
from PIL import Image

from sahi.slicing import shift_bboxes, shift_masks, slice_coco, slice_image
from sahi.utils.coco import Coco
from sahi.utils.cv import read_image


class TestSlicing:
    def test_slice_image(self):
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
        assert slice_image_result[3]["starting_pixel"] == [924, 0]
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

    def test_slice_coco(self):
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

    def test_shift_bboxes(self):
        import torch

        bboxes = [[1, 2, 3, 4]]
        shift_x = 10
        shift_y = 20
        shifted_bboxes = shift_bboxes(bboxes=bboxes, offset=[shift_x, shift_y])
        assert shifted_bboxes == [[11, 22, 13, 24]]
        assert isinstance(shifted_bboxes, list)

        bboxes = np.array([[1, 2, 3, 4]])
        shifted_bboxes = shift_bboxes(bboxes=bboxes, offset=[shift_x, shift_y])
        assert shifted_bboxes.tolist() == [[11, 22, 13, 24]]
        assert isinstance(shifted_bboxes, np.ndarray)

        bboxes = torch.tensor([[1, 2, 3, 4]])
        shifted_bboxes = shift_bboxes(bboxes=bboxes, offset=[shift_x, shift_y])
        assert shifted_bboxes.tolist() == [[11, 22, 13, 24]]
        assert isinstance(shifted_bboxes, torch.Tensor)

    def test_shift_masks(self):
        masks = np.zeros((3, 30, 30), dtype=bool)
        shift_x = 10
        shift_y = 20
        full_shape = [720, 1280]
        shifted_masks = shift_masks(masks=masks, offset=[shift_x, shift_y], full_shape=full_shape)
        assert shifted_masks.shape == (3, 720, 1280)
        assert isinstance(shifted_masks, np.ndarray)
