# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import unittest


class TestAnnotation(unittest.TestCase):
    def test_bounding_box(self):
        from sahi.annotation import BoundingBox

        bbox_minmax = [30, 30, 100, 150]
        shift_amount = [50, 40]

        bbox = BoundingBox(bbox_minmax, shift_amount=[0, 0])
        expanded_bbox = bbox.get_expanded_box(ratio=0.1)

        bbox = BoundingBox(bbox_minmax, shift_amount=shift_amount)
        shifted_bbox = bbox.get_shifted_box()

        # compare
        self.assertEqual(expanded_bbox.to_coco_bbox(), [18, 23, 94, 134])
        self.assertEqual(expanded_bbox.to_voc_bbox(), [18, 23, 112, 157])
        self.assertEqual(shifted_bbox.to_voc_bbox(), [80, 70, 150, 190])

    def test_category(self):
        from sahi.annotation import Category

        category_id = 1
        category_name = "car"
        category = Category(id=category_id, name=category_name)
        self.assertEqual(category.id, category_id)
        self.assertEqual(category.name, category_name)

    def test_mask(self):
        from sahi.annotation import Mask

        coco_segmentation = [[1, 1, 325, 125, 250, 200, 5, 200]]
        full_shape_height, full_shape_width = 500, 600
        full_shape = [full_shape_height, full_shape_width]

        mask = Mask.from_coco_segmentation(segmentation=coco_segmentation, full_shape=full_shape)

        self.assertEqual(mask.full_shape_height, full_shape_height)
        self.assertEqual(mask.full_shape_width, full_shape_width)
        self.assertEqual(mask.bool_mask[11, 2], True)

    def test_object_annotation(self):
        from sahi.annotation import ObjectAnnotation


if __name__ == "__main__":
    unittest.main()
