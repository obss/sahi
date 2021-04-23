# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import shutil
import unittest

from sahi.utils.coco import get_imageid2annotationlist_mapping, merge, update_categories
from sahi.utils.file import load_json


class TestCocoUtils(unittest.TestCase):
    def test_coco_categories(self):
        from sahi.utils.coco import CocoCategory

        category_id = 0
        category_name = "human"
        supercategory = "human"
        coco_category1 = CocoCategory(
            id=category_id, name=category_name, supercategory=supercategory
        )
        coco_category2 = CocoCategory(id=category_id, name=category_name)
        coco_category3 = CocoCategory.from_coco_category(
            {
                "id": category_id,
                "name": category_name,
                "supercategory": supercategory,
            }
        )

        self.assertEqual(coco_category1.id, category_id)
        self.assertEqual(coco_category1.id, coco_category2.id)
        self.assertEqual(coco_category1.id, coco_category3.id)

        self.assertEqual(coco_category1.name, category_name)
        self.assertEqual(coco_category1.name, coco_category2.name)
        self.assertEqual(coco_category1.name, coco_category3.name)

        self.assertEqual(coco_category1.supercategory, supercategory)
        self.assertEqual(coco_category1.supercategory, coco_category2.supercategory)
        self.assertEqual(coco_category1.supercategory, coco_category3.supercategory)

        self.assertEqual(coco_category1.json["id"], category_id)
        self.assertEqual(coco_category1.json["name"], category_name)
        self.assertEqual(coco_category1.json["supercategory"], supercategory)

    def test_coco_annotation(self):
        from sahi.utils.coco import CocoAnnotation

        coco_segmentation = [[1, 1, 325, 125, 250, 200, 5, 200]]
        category_id = 3
        category_name = "car"
        coco_annotation = CocoAnnotation.from_coco_segmentation(
            segmentation=coco_segmentation,
            category_id=category_id,
            category_name=category_name,
        )

        self.assertAlmostEqual(coco_annotation.area, 41177, 1)
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

    def test_cocovid_annotation(self):
        from sahi.utils.coco import CocoVidAnnotation

        bbox = [1, 1, 324, 199]
        category_id = 3
        category_name = "car"
        image_id = 13
        instance_id = 22
        iscrowd = 0
        cocovid_annotation = CocoVidAnnotation(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            image_id=image_id,
            instance_id=instance_id,
            iscrowd=iscrowd,
        )

        self.assertEqual(cocovid_annotation.json["bbox"], bbox)
        self.assertEqual(cocovid_annotation.json["category_id"], category_id)
        self.assertEqual(cocovid_annotation.json["category_name"], category_name)
        self.assertEqual(cocovid_annotation.json["image_id"], image_id)
        self.assertEqual(cocovid_annotation.json["instance_id"], instance_id)
        self.assertEqual(cocovid_annotation.json["iscrowd"], iscrowd)

    def test_coco_image(self):
        from sahi.utils.coco import CocoAnnotation, CocoImage

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

    def test_cocovid_image(self):
        from sahi.utils.coco import CocoVidAnnotation, CocoVidImage

        # init coco image
        file_name = "tests/data/small-vehicles1.jpeg"
        height = 580
        width = 1068
        cocovid_image = CocoVidImage(file_name, height, width)

        # create and add first annotation
        bbox1 = [1, 1, 324, 199]
        category_id1 = 3
        category_name1 = "car"
        image_id1 = 13
        instance_id1 = 22
        iscrowd1 = 0
        cocovid_annotation_1 = CocoVidAnnotation(
            bbox=bbox1,
            category_id=category_id1,
            category_name=category_name1,
            image_id=image_id1,
            instance_id=instance_id1,
            iscrowd=iscrowd1,
        )
        cocovid_image.add_annotation(cocovid_annotation_1)

        # create and add second annotation
        bbox2 = [1, 1, 50, 150]
        category_id2 = 2
        category_name2 = "human"
        image_id2 = 14
        instance_id2 = 23
        iscrowd2 = 0
        cocovid_annotation_2 = CocoVidAnnotation(
            bbox=bbox2,
            category_id=category_id2,
            category_name=category_name2,
            image_id=image_id2,
            instance_id=instance_id2,
            iscrowd=iscrowd2,
        )
        cocovid_image.add_annotation(cocovid_annotation_2)

        # compare
        self.assertEqual(cocovid_image.file_name, file_name)
        self.assertEqual(cocovid_image.json["file_name"], file_name)
        self.assertEqual(cocovid_image.height, height)
        self.assertEqual(cocovid_image.json["height"], height)
        self.assertEqual(cocovid_image.width, width)
        self.assertEqual(cocovid_image.json["width"], width)
        self.assertEqual(len(cocovid_image.annotations), 2)
        self.assertEqual(cocovid_image.annotations[0].category_id, category_id1)
        self.assertEqual(cocovid_image.annotations[0].category_name, category_name1)
        self.assertEqual(cocovid_image.annotations[0].image_id, image_id1)
        self.assertEqual(cocovid_image.annotations[0].bbox, bbox1)

        self.assertEqual(cocovid_image.annotations[1].category_id, category_id2)
        self.assertEqual(cocovid_image.annotations[1].category_name, category_name2)
        self.assertEqual(cocovid_image.annotations[1].instance_id, instance_id2)
        self.assertEqual(cocovid_image.annotations[1].bbox, bbox2)

    def test_coco_video(self):
        from sahi.utils.coco import CocoVidAnnotation, CocoVideo, CocoVidImage

        # init coco image
        file_name = "tests/data/small-vehicles1.jpeg"
        height1 = 519
        width1 = 1067
        cocovid_image = CocoVidImage(file_name, height1, width1)

        # create and add first annotation
        bbox1 = [1, 1, 324, 199]
        category_id1 = 3
        category_name1 = "car"
        image_id1 = 13
        instance_id1 = 22
        iscrowd1 = 0
        cocovid_annotation_1 = CocoVidAnnotation(
            bbox=bbox1,
            category_id=category_id1,
            category_name=category_name1,
            image_id=image_id1,
            instance_id=instance_id1,
            iscrowd=iscrowd1,
        )
        cocovid_image.add_annotation(cocovid_annotation_1)

        # init coco video
        name = "small-vehicles"
        height2 = 580
        width2 = 1068
        coco_video = CocoVideo(name=name, height=height2, width=width2)

        # add first image
        coco_video.add_cocovidimage(cocovid_image)

        # compare
        self.assertEqual(coco_video.name, name)
        self.assertEqual(coco_video.json["name"], name)
        self.assertEqual(coco_video.height, height2)
        self.assertEqual(coco_video.json["height"], height2)
        self.assertEqual(coco_video.width, width2)
        self.assertEqual(coco_video.json["width"], width2)
        self.assertEqual(len(coco_video.images), 1)
        self.assertEqual(coco_video.images[0].file_name, file_name)
        self.assertEqual(coco_video.images[0].json["file_name"], file_name)
        self.assertEqual(coco_video.images[0].height, height1)
        self.assertEqual(coco_video.images[0].json["height"], height1)
        self.assertEqual(coco_video.images[0].width, width1)
        self.assertEqual(coco_video.images[0].json["width"], width1)
        self.assertEqual(coco_video.images[0].annotations[0].bbox, bbox1)
        self.assertEqual(coco_video.images[0].annotations[0].json["bbox"], bbox1)

    def test_coco(self):
        from sahi.utils.coco import Coco

        category_mapping = {1: "human", 2: "car"}
        # init coco
        coco_path = "tests/data/coco_utils/terrain_all_coco.json"
        coco_dict = load_json(coco_path)
        coco1 = Coco.from_coco_dict_or_path(coco_dict)
        coco2 = Coco.from_coco_dict_or_path(coco_path)

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

    def test_coco2yolo(self):
        from sahi.utils.coco import Coco

        coco_dict_path = "tests/data/coco_utils/combined_coco.json"
        image_dir = "tests/data/coco_utils/"
        output_dir = "tests/data/coco2yolo/"
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        coco = Coco.from_coco_dict_or_path(coco_dict_path)
        coco.export_as_yolov5(
            image_dir, output_dir=output_dir, train_split_rate=0.5, numpy_seed=0
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

    def test_merge_from_list(self):
        from sahi.utils.coco import merge_from_list

        # load coco files to be combined
        coco_path1 = "tests/data/coco_utils/terrain1_coco.json"
        coco_path2 = "tests/data/coco_utils/terrain2_coco.json"
        coco_path3 = "tests/data/coco_utils/terrain3_coco.json"
        coco_dict1 = load_json(coco_path1)
        self.assertEqual(len(coco_dict1["images"]), 1)
        self.assertEqual(len(coco_dict1["annotations"]), 7)
        self.assertEqual(len(coco_dict1["categories"]), 1)

        coco_dict2 = load_json(coco_path2)
        self.assertEqual(len(coco_dict2["images"]), 1)
        self.assertEqual(len(coco_dict2["annotations"]), 5)
        self.assertEqual(len(coco_dict2["categories"]), 1)

        coco_dict3 = load_json(coco_path3)
        self.assertEqual(len(coco_dict3["images"]), 1)
        self.assertEqual(len(coco_dict3["annotations"]), 10)
        self.assertEqual(len(coco_dict3["categories"]), 1)

        # merge without desired_name2id
        merged_coco_dict = merge_from_list([coco_dict1, coco_dict2, coco_dict3])
        self.assertEqual(len(merged_coco_dict["images"]), 3)
        self.assertEqual(len(merged_coco_dict["annotations"]), 22)
        self.assertEqual(len(merged_coco_dict["categories"]), 2)
        self.assertEqual(
            merged_coco_dict["annotations"][12]["bbox"],
            coco_dict3["annotations"][0]["bbox"],
        )
        self.assertEqual(
            merged_coco_dict["annotations"][12]["id"],
            13,
        )
        self.assertEqual(
            merged_coco_dict["annotations"][12]["category_id"],
            coco_dict3["annotations"][0]["category_id"],
        )
        self.assertEqual(
            merged_coco_dict["annotations"][12]["image_id"],
            3,
        )
        self.assertEqual(
            merged_coco_dict["annotations"][12]["category_id"],
            coco_dict3["annotations"][0]["category_id"],
        )
        self.assertEqual(
            merged_coco_dict["annotations"][9]["category_id"],
            2,
        )
        self.assertEqual(
            merged_coco_dict["annotations"][9]["image_id"],
            2,
        )

    def test_multi_coco_init(self):
        from sahi.utils.coco import Coco

        # load coco files to be combined
        coco_path1 = "tests/data/coco_utils/terrain1_coco.json"
        coco_path2 = "tests/data/coco_utils/terrain2_coco.json"
        coco_path3 = "tests/data/coco_utils/terrain3_coco.json"
        coco = Coco.from_coco_dict_or_path([coco_path1, coco_path2, coco_path3])
        self.assertEqual(len(coco.json["images"]), 3)
        self.assertEqual(len(coco.json["annotations"]), 22)
        self.assertEqual(len(coco.json["categories"]), 2)
        self.assertEqual(len(coco.images), 3)

        self.assertEqual(
            coco.json["annotations"][12]["id"],
            13,
        )
        self.assertEqual(
            coco.json["annotations"][12]["image_id"],
            3,
        )
        self.assertEqual(
            coco.json["annotations"][9]["category_id"],
            2,
        )
        self.assertEqual(
            coco.json["annotations"][9]["image_id"],
            2,
        )

    def test_cocovid(self):
        from sahi.utils.coco import CocoVid

        # TODO


if __name__ == "__main__":
    unittest.main()
