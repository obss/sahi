# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2025.

import logging

import pytest

logger = logging.getLogger(__name__)


class TestAnnotation:
    def test_bounding_box(self):
        from sahi.annotation import BoundingBox

        bbox_minmax = [30.0, 30.0, 100.0, 150.0]
        shift_amount = [50, 40]

        bbox = BoundingBox(bbox_minmax)
        expanded_bbox = bbox.get_expanded_box(ratio=0.1)

        bbox = BoundingBox(bbox_minmax, shift_amount=shift_amount)
        shifted_bbox = bbox.get_shifted_box()

        # compare
        assert expanded_bbox.to_xywh() == [18, 23, 94, 134]
        assert expanded_bbox.to_xyxy() == [18, 23, 112, 157]
        assert shifted_bbox.to_xyxy() == [80, 70, 150, 190]

    def test_bounding_box_immutability(self):
        import dataclasses

        from sahi.annotation import BoundingBox

        bbox_tuple = (10.0, 20.0, 30.0, 40.0)
        bbox = BoundingBox(bbox_tuple)

        # Attempt to mutate the box tuple directly
        with pytest.raises(TypeError):
            bbox.box[0] = 99.0

        # Attempt to mutate the shift_amount tuple directly
        with pytest.raises(TypeError):
            bbox.shift_amount[0] = 99

        # Attempt to assign a new value to an attribute
        with pytest.raises(dataclasses.FrozenInstanceError):
            bbox.box = (1.0, 2.0, 3.0, 4.0)

        # Attempt to assign a new value to a property
        with pytest.raises(dataclasses.FrozenInstanceError):
            bbox.minx = 123.0

        # Confirm the values remain unchanged
        assert bbox.box == bbox_tuple
        assert bbox.shift_amount == (0, 0)

    def test_category(self):
        from sahi.annotation import Category

        category_id = 1
        category_name = "car"
        category = Category(id=category_id, name=category_name)
        assert category.id == category_id
        assert category.name == category_name

        # id must be int
        with pytest.raises(TypeError):
            Category(id="not-an-int", name="car")

        # name must be str
        with pytest.raises(TypeError):
            Category(id=1, name=123)

    def test_category_immutability(self):
        import dataclasses

        from sahi.annotation import Category

        category = Category(id=5, name="person")

        # Attempt to mutate the id directly
        with pytest.raises(dataclasses.FrozenInstanceError):
            category.id = 10

        # Attempt to mutate the name directly
        with pytest.raises(dataclasses.FrozenInstanceError):
            category.name = "cat"

        # Confirm the values remain unchanged
        assert category.id == 5
        assert category.name == "person"

    def test_mask(self):
        from sahi.annotation import Mask

        coco_segmentation = [[1.0, 1.0, 325.0, 125.0, 250.0, 200.0, 5.0, 200.0]]
        full_shape_height, full_shape_width = 500, 600
        full_shape = [full_shape_height, full_shape_width]

        mask = Mask(segmentation=coco_segmentation, full_shape=full_shape)

        assert mask.full_shape_height == full_shape_height
        assert mask.full_shape_width == full_shape_width
        logger.debug(f"{type(mask.bool_mask[11, 2])=} {mask.bool_mask[11, 2]=}")
        assert mask.bool_mask[11, 2]

    def test_object_annotation(self):
        from sahi.annotation import ObjectAnnotation

        bbox = [100, 200, 150, 230]
        coco_bbox = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
        category_id = 2
        category_name = "car"
        shift_amount = [0, 0]
        image_height = 1080
        image_width = 1920
        full_shape = [image_height, image_width]

        object_annotation1 = ObjectAnnotation(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

        object_annotation2 = ObjectAnnotation.from_coco_annotation_dict(
            annotation_dict={"bbox": coco_bbox, "category_id": category_id, "segmentation": []},
            category_name=category_name,
            full_shape=full_shape,
            shift_amount=shift_amount,
        )

        object_annotation3 = ObjectAnnotation.from_coco_bbox(
            bbox=coco_bbox,
            category_id=category_id,
            category_name=category_name,
            full_shape=full_shape,
            shift_amount=shift_amount,
        )

        assert object_annotation1.bbox.minx == bbox[0]
        assert object_annotation1.bbox.miny == bbox[1]
        assert object_annotation1.bbox.maxx == bbox[2]
        assert object_annotation1.bbox.maxy == bbox[3]
        assert object_annotation1.category.id == category_id
        assert object_annotation1.category.name == category_name

        assert object_annotation2.bbox.minx == bbox[0]
        assert object_annotation2.bbox.miny == bbox[1]
        assert object_annotation2.bbox.maxx == bbox[2]
        assert object_annotation2.bbox.maxy == bbox[3]
        assert object_annotation2.category.id == category_id
        assert object_annotation2.category.name == category_name

        assert object_annotation3.bbox.minx == bbox[0]
        assert object_annotation3.bbox.miny == bbox[1]
        assert object_annotation3.bbox.maxx == bbox[2]
        assert object_annotation3.bbox.maxy == bbox[3]
        assert object_annotation3.category.id == category_id
        assert object_annotation3.category.name == category_name
