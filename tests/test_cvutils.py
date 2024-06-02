import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from sahi.utils.cv import (
    Colors,
    apply_color_mask,
    exif_transpose,
    get_bbox_from_bool_mask,
    get_coco_segmentation_from_bool_mask,
    read_image,
)


class TestCvUtils(unittest.TestCase):
    def test_hex_to_rgb(self):
        colors = Colors()
        self.assertEqual(colors.hex_to_rgb("#FF3838"), (255, 56, 56))

    def test_hex_to_rgb_retrieve(self):
        colors = Colors()
        self.assertEqual(colors(0), (255, 56, 56))

    @patch("sahi.utils.cv.cv2.cvtColor")
    @patch("sahi.utils.cv.cv2.imread")
    def test_read_image(self, mock_imread, mock_cvtColor):
        fake_image = "test.jpg"
        fake_image_val = np.array([[[10, 20, 30]]], dtype=np.uint8)
        fake_image_rbg_val = np.array([[[10, 20, 30]]], dtype=np.uint8)
        mock_imread.return_value = fake_image_val
        mock_cvtColor.return_value = fake_image_rbg_val

        result = read_image(fake_image)

        # mock_cv2.assert_called_once_with(fake_image)
        mock_imread.assert_called_once_with(fake_image)
        np.testing.assert_array_equal(result, fake_image_rbg_val)

    def test_apply_color_mask(self):
        image = np.array([[0, 1]], dtype=np.uint8)
        color = (255, 0, 0)

        expected_output = np.array([[[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)

        result = apply_color_mask(image, color)

        np.testing.assert_array_equal(result, expected_output)

    def test_get_coco_segmentation_from_bool_mask_simple(self):
        mask = np.zeros((10, 10), dtype=bool)
        result = get_coco_segmentation_from_bool_mask(mask)
        self.assertEqual(result, [])

    def test_get_coco_segmentation_from_bool_mask_polygon(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[1:4, 1:4] = True
        mask[5:8, 5:8] = True
        result = get_coco_segmentation_from_bool_mask(mask)
        self.assertEqual(len(result), 2)

    def test_get_bbox_from_bool_mask(self):
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
        self.assertEqual(result, expected_result)

    def test_exif_transpose_simple(self):
        test_image = Image.new("RGB", (100, 100), color="red")
        transposed = exif_transpose(test_image)
        self.assertEqual(transposed, test_image)

    def test_exif_transpose_non_standard(self):
        test_image = Image.new("RGB", (100, 100), color="red")
        exif = test_image.getexif()
        exif[0x0112] = 9
        test_image.info["exif"] = exif.tobytes()
        transposed = exif_transpose(test_image)
        self.assertEqual(transposed, test_image)


if __name__ == "__main__":
    unittest.main()
