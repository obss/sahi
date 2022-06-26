# OBSS SAHI Tool
# Code written by Murat C Varer, 2022.


import unittest

import numpy as np
from PIL import Image

from sahi.slicing import slice_coco, slice_image
from sahi.utils.coco import Coco
from sahi.utils.cv import read_image


class TestAutoSlicing(unittest.TestCase):
    def test_auto_slice_image(self):
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
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(slice_image_result.images), 18)
        self.assertEqual(len(slice_image_result.coco_images), 18)
        self.assertEqual(slice_image_result.coco_images[0].annotations, [])
        self.assertEqual(slice_image_result.coco_images[15].annotations[1].area, 7296)
        self.assertEqual(
            slice_image_result.coco_images[15].annotations[1].bbox,
            [17, 356, 48, 152],
        )

        image_cv = read_image(image_path)
        slice_image_result = slice_image(
            image=image_cv,
            coco_annotation_list=coco.images[0].annotations,
            output_file_name=output_file_name,
            output_dir=output_dir,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(slice_image_result.images), 18)
        self.assertEqual(len(slice_image_result.coco_images), 18)
        self.assertEqual(slice_image_result.coco_images[0].annotations, [])
        self.assertEqual(slice_image_result.coco_images[15].annotations[1].area, 7296)
        self.assertEqual(
            slice_image_result.coco_images[15].annotations[1].bbox,
            [17, 356, 48, 152],
        )

        image_pil = Image.open(image_path)
        slice_image_result = slice_image(
            image=image_pil,
            coco_annotation_list=coco.images[0].annotations,
            output_file_name=output_file_name,
            output_dir=output_dir,
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(slice_image_result.images), 18)
        self.assertEqual(len(slice_image_result.coco_images), 18)
        self.assertEqual(slice_image_result.coco_images[0].annotations, [])
        self.assertEqual(slice_image_result.coco_images[15].annotations[1].area, 7296)
        self.assertEqual(
            slice_image_result.coco_images[15].annotations[1].bbox,
            [17, 356, 48, 152],
        )

    def test_auto_slice_coco(self):
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
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(coco_dict["images"]), 8)
        self.assertEqual(coco_dict["images"][1]["height"], 512)
        self.assertEqual(coco_dict["images"][1]["width"], 512)
        self.assertEqual(len(coco_dict["annotations"]), 22)
        self.assertEqual(coco_dict["annotations"][2]["id"], 3)
        self.assertEqual(coco_dict["annotations"][2]["image_id"], 2)
        self.assertEqual(coco_dict["annotations"][2]["category_id"], 1)
        self.assertEqual(coco_dict["annotations"][2]["area"], 12483)
        self.assertEqual(
            coco_dict["annotations"][2]["bbox"],
            [238, 237, 73, 171],
        )

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
            min_area_ratio=0.1,
            out_ext=".png",
            verbose=False,
        )

        self.assertEqual(len(coco_dict["images"]), 20)
        self.assertEqual(coco_dict["images"][1]["height"], 512)
        self.assertEqual(coco_dict["images"][1]["width"], 512)
        self.assertEqual(len(coco_dict["annotations"]), 22)
        self.assertEqual(coco_dict["annotations"][2]["id"], 3)
        self.assertEqual(coco_dict["annotations"][2]["image_id"], 12)
        self.assertEqual(coco_dict["annotations"][2]["category_id"], 1)
        self.assertEqual(coco_dict["annotations"][2]["area"], 12483)
        self.assertEqual(
            coco_dict["annotations"][2]["bbox"],
            [238, 237, 73, 171],
        )

        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
