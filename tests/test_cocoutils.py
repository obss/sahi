import unittest

from sahi.utils.coco import (
    CocoAnnotation,
    get_imageid2annotationlist_mapping,
    merge,
    update_categories,
)
from sahi.utils.file import load_json


class TestCocoUtils(unittest.TestCase):
    def test_coco_annotation(self):
        coco_segmentation = [[1, 1, 325, 125, 250, 200, 5, 200]]
        category_id = 3
        category_name = "car"
        coco_annotation = CocoAnnotation.from_coco_segmentation(
            segmentation=coco_segmentation,
            category_id=category_id,
            category_name=category_name,
        )

        self.assertAlmostEqual(coco_annotation.area, 41177.5, 1)
        self.assertEqual(coco_annotation.bbox, [1, 1, 324, 199])
        self.assertEqual(coco_annotation.category_id, category_id)
        self.assertEqual(coco_annotation.category_name, category_name)
        self.assertEqual(coco_annotation.segmentation, coco_segmentation)

        coco_bbox = [1, 1, 100, 100]
        category_id = 3
        coco_annotation = CocoAnnotation.from_coco_bbox(
            bbox=coco_bbox,
            category_id=category_id,
            category_name=category_name,
        )

        self.assertEqual(coco_annotation.area, 10000)
        self.assertEqual(coco_annotation.bbox, coco_bbox)
        self.assertEqual(coco_annotation.category_id, category_id)
        self.assertEqual(coco_annotation.category_name, category_name)
        self.assertEqual(coco_annotation.segmentation, [])

    def test_coco_image(self):
        from sahi.utils.coco import CocoImage

        # init coco image
        file_name = "tests/data/small-vehicles1.jpeg"
        height = 580
        width = 1068
        coco_image = CocoImage(file_name, height, width)

        # create and add first annotation
        coco_segmentation = [[1, 1, 325, 125, 250, 200, 5, 200]]
        category_id = 3
        category_name = "car"
        coco_annotation_1 = CocoAnnotation.from_coco_segmentation(
            segmentation=coco_segmentation,
            category_id=category_id,
            category_name=category_name,
        )
        coco_image.add_annotation(coco_annotation_1)

        # create and add second annotation
        coco_bbox = [1, 1, 100, 100]
        category_id = 2
        category_name = "bus"
        coco_annotation_2 = CocoAnnotation.from_coco_bbox(
            bbox=coco_bbox, category_id=category_id, category_name=category_name
        )
        coco_image.add_annotation(coco_annotation_2)

        # compare
        self.assertEqual(coco_image.file_name, file_name)
        self.assertEqual(coco_image.height, height)
        self.assertEqual(coco_image.width, width)
        self.assertEqual(len(coco_image.annotations), 2)
        self.assertEqual(coco_image.annotations[0].category_id, 3)
        self.assertEqual(coco_image.annotations[0].category_name, "car")
        self.assertEqual(coco_image.annotations[0].segmentation, coco_segmentation)
        self.assertEqual(coco_image.annotations[1].category_id, 2)
        self.assertEqual(coco_image.annotations[1].category_name, "bus")
        self.assertEqual(coco_image.annotations[1].bbox, coco_bbox)

    def test_coco(self):
        from sahi.utils.coco import Coco

        category_mapping = {"1": "human", "2": "car"}
        # init coco
        coco_path = "tests/data/coco_utils/terrain_all_coco.json"
        coco_dict = load_json(coco_path)
        coco1 = Coco(coco_dict)
        coco2 = Coco.from_coco_path(coco_path)

        # compare
        self.assertEqual(len(coco1.images), 3)
        self.assertEqual(len(coco2.images), 3)
        self.assertEqual(coco1.images[2].annotations[1].category_name, "human")
        self.assertEqual(coco2.images[2].annotations[1].category_name, "human")
        self.assertEqual(
            coco1.images[1].annotations[1].segmentation,
            [[501, 451, 622, 451, 622, 543, 501, 543]],
        )
        self.assertEqual(
            coco2.images[1].annotations[1].segmentation,
            [[501, 451, 622, 451, 622, 543, 501, 543]],
        )
        self.assertEqual(
            coco1.category_mapping,
            category_mapping,
        )
        self.assertEqual(
            coco2.category_mapping,
            category_mapping,
        )

    def test_update_categories(self):
        coco_path = "tests/data/coco_utils/terrain2_coco.json"
        source_coco_dict = load_json(coco_path)

        self.assertEqual(len(source_coco_dict["annotations"]), 5)
        self.assertEqual(len(source_coco_dict["images"]), 1)
        self.assertEqual(len(source_coco_dict["categories"]), 1)
        self.assertEqual(
            source_coco_dict["categories"],
            [{"id": 1, "name": "car", "supercategory": "car"}],
        )
        self.assertEqual(source_coco_dict["annotations"][1]["category_id"], 1)

        # update categories
        desired_name2id = {"human": 1, "car": 2, "big_vehicle": 3}
        target_coco_dict = update_categories(
            desired_name2id=desired_name2id, coco_dict=source_coco_dict
        )

        self.assertEqual(len(target_coco_dict["annotations"]), 5)
        self.assertEqual(len(target_coco_dict["images"]), 1)
        self.assertEqual(len(target_coco_dict["categories"]), 3)
        self.assertEqual(
            target_coco_dict["categories"],
            [
                {"id": 1, "name": "human", "supercategory": "human"},
                {"id": 2, "name": "car", "supercategory": "car"},
                {"id": 3, "name": "big_vehicle", "supercategory": "big_vehicle"},
            ],
        )
        self.assertEqual(target_coco_dict["annotations"][1]["category_id"], 2)

    def test_get_imageid2annotationlist_mapping(self):
        coco_path = "tests/data/coco_utils/combined_coco.json"
        coco_dict = load_json(coco_path)
        imageid2annotationlist_mapping = get_imageid2annotationlist_mapping(coco_dict)
        self.assertEqual(len(imageid2annotationlist_mapping), 2)

        def check_image_id(image_id):

            image_ids = [
                annotationlist["image_id"]
                for annotationlist in imageid2annotationlist_mapping[image_id]
            ]
            self.assertEqual(image_ids, [image_id] * len(image_ids))

        check_image_id(image_id=1)
        check_image_id(image_id=2)

    def test_merge(self):
        # load coco files to be combined
        coco_path1 = "tests/data/coco_utils/terrain1_coco.json"
        coco_path2 = "tests/data/coco_utils/terrain2_coco.json"
        coco_dict1 = load_json(coco_path1)
        self.assertEqual(len(coco_dict1["images"]), 1)
        self.assertEqual(len(coco_dict1["annotations"]), 7)
        self.assertEqual(len(coco_dict1["categories"]), 1)

        coco_dict2 = load_json(coco_path2)
        self.assertEqual(len(coco_dict2["images"]), 1)
        self.assertEqual(len(coco_dict2["annotations"]), 5)
        self.assertEqual(len(coco_dict2["categories"]), 1)

        # merge without desired_name2id
        merged_coco_dict = merge(coco_dict1, coco_dict2)
        self.assertEqual(len(merged_coco_dict["images"]), 2)
        self.assertEqual(len(merged_coco_dict["annotations"]), 7)
        self.assertEqual(len(merged_coco_dict["categories"]), 1)

        # merge with desired_name2id
        desired_name2id = {"human": 1, "car": 2, "big_vehicle": 3}
        merged_coco_dict = merge(coco_dict1, coco_dict2, desired_name2id)
        self.assertEqual(len(merged_coco_dict["images"]), 2)
        self.assertEqual(len(merged_coco_dict["annotations"]), 12)
        self.assertEqual(len(merged_coco_dict["categories"]), 3)
        self.assertEqual(merged_coco_dict["annotations"][6]["category_id"], 1)
        self.assertEqual(merged_coco_dict["annotations"][6]["image_id"], 1)
        self.assertEqual(merged_coco_dict["annotations"][6]["id"], 7)
        self.assertEqual(merged_coco_dict["annotations"][7]["category_id"], 2)
        self.assertEqual(merged_coco_dict["annotations"][7]["image_id"], 2)
        self.assertEqual(merged_coco_dict["annotations"][7]["id"], 8)


if __name__ == "__main__":
    unittest.main()
