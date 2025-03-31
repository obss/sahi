# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
import os
import shutil
import unittest

from sahi.utils.coco import merge, update_categories
from sahi.utils.file import load_json

logger = logging.getLogger(__name__)


class TestCocoUtils(unittest.TestCase):
    def test_coco_categories(self):
        from sahi.utils.coco import CocoCategory

        category_id = 0
        category_name = "human"
        supercategory = "human"
        coco_category1 = CocoCategory(id=category_id, name=category_name, supercategory=supercategory)
        coco_category2 = CocoCategory(id=category_id, name=category_name)
        coco_category3 = CocoCategory.from_coco_category(
            {
                "id": category_id,
                "name": category_name,
                "supercategory": supercategory,
            }
        )
        coco_category4 = CocoCategory.from_coco_category(
            {
                "id": category_id,
                "name": category_name,
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

        self.assertEqual(coco_category4.id, category_id)
        self.assertEqual(coco_category4.name, category_name)
        self.assertEqual(coco_category4.supercategory, category_name)

    def test_coco_annotation(self):
        from sahi.utils.coco import CocoAnnotation

        coco_segmentation = [[1, 1, 325, 125, 250, 200, 5, 200]]
        category_id = 3
        category_name = "car"
        coco_annotation = CocoAnnotation.from_coco_segmentation(
            segmentation=coco_segmentation, category_id=category_id, category_name=category_name
        )

        self.assertAlmostEqual(coco_annotation.area, 41177, 1)
        self.assertEqual(coco_annotation.bbox, [1, 1, 324, 199])
        self.assertEqual(coco_annotation.category_id, category_id)
        self.assertEqual(coco_annotation.category_name, category_name)
        self.assertEqual(coco_annotation.segmentation, coco_segmentation)

        coco_bbox = [1, 1, 100, 100]
        category_id = 3
        coco_annotation = CocoAnnotation.from_coco_bbox(
            bbox=coco_bbox, category_id=category_id, category_name=category_name
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
        from sahi.utils.coco import CocoAnnotation, CocoImage, CocoPrediction

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
            segmentation=coco_segmentation, category_id=category_id, category_name=category_name
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

        # create and add first prediction
        prediction_coco_segmentation = [[4, 3, 315, 124, 265, 198, 5, 198]]
        score = 0.983425
        category_id = 3
        category_name = "car"
        coco_prediction_1 = CocoPrediction.from_coco_segmentation(
            segmentation=prediction_coco_segmentation, category_id=category_id, category_name=category_name, score=score
        )
        coco_image.add_prediction(coco_prediction_1)

        # create and add second prediction
        prediction_coco_bbox = [2, 5, 103, 98]
        score = 0.683465
        category_id = 2
        category_name = "bus"
        coco_prediction_2 = CocoPrediction.from_coco_bbox(
            bbox=prediction_coco_bbox, category_id=category_id, category_name=category_name, score=score
        )
        coco_image.add_prediction(coco_prediction_2)

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

        self.assertEqual(len(coco_image.predictions), 2)
        self.assertEqual(coco_image.predictions[0].category_id, 3)
        self.assertEqual(coco_image.predictions[0].category_name, "car")
        self.assertEqual(coco_image.predictions[0].segmentation, prediction_coco_segmentation)
        self.assertEqual(coco_image.predictions[0].score, 0.983425)
        self.assertEqual(coco_image.predictions[1].category_id, 2)
        self.assertEqual(coco_image.predictions[1].category_name, "bus")
        self.assertEqual(coco_image.predictions[1].bbox, prediction_coco_bbox)
        self.assertEqual(coco_image.predictions[1].score, 0.683465)

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
        self.assertEqual(coco1.images[1].annotations[1].segmentation, [[501, 451, 622, 451, 622, 543, 501, 543]])
        self.assertEqual(coco2.images[1].annotations[1].segmentation, [[501, 451, 622, 451, 622, 543, 501, 543]])
        self.assertEqual(coco1.category_mapping, category_mapping)
        self.assertEqual(coco2.category_mapping, category_mapping)
        self.assertEqual(coco1.stats, coco2.stats)
        assert coco1.stats
        self.assertEqual(coco1.stats["num_images"], len(coco1.images))
        self.assertEqual(coco1.stats["num_annotations"], len(coco1.json["annotations"]))

    def test_split_coco_as_train_val(self):
        from sahi.utils.coco import Coco

        coco_dict_path = "tests/data/coco_utils/combined_coco.json"
        image_dir = "tests/data/coco_utils/"
        coco = Coco.from_coco_dict_or_path(coco_dict_path, image_dir=image_dir)
        result = coco.split_coco_as_train_val(train_split_rate=0.5, numpy_seed=1)
        assert len(coco.images) == 2
        # NOTE: the split uses a seed. The splitting was changed from numpy to the std
        # random.shuffle package, and the seed of 0 changed the output.
        if len(result["train_coco"].json["annotations"]) == 7:
            result = coco.split_coco_as_train_val(train_split_rate=0.5, numpy_seed=0)
        self.assertEqual(len(result["train_coco"].json["images"]), 1)
        self.assertEqual(len(result["train_coco"].json["annotations"]), 5)
        self.assertEqual(result["train_coco"].json["images"][0]["height"], 682)
        self.assertEqual(result["train_coco"].image_dir, image_dir)
        assert result["train_coco"].stats
        self.assertEqual(result["train_coco"].stats["num_images"], len(result["train_coco"].images))
        self.assertEqual(result["train_coco"].stats["num_annotations"], len(result["train_coco"].json["annotations"]))

        self.assertEqual(len(result["val_coco"].json["images"]), 1)
        self.assertEqual(len(result["val_coco"].json["annotations"]), 7)
        self.assertEqual(result["val_coco"].json["images"][0]["height"], 1365)
        self.assertEqual(result["val_coco"].image_dir, image_dir)
        assert result["val_coco"].stats
        self.assertEqual(result["val_coco"].stats["num_images"], len(result["val_coco"].images))
        self.assertEqual(result["val_coco"].stats["num_annotations"], len(result["val_coco"].json["annotations"]))

    def test_coco2yolo(self):
        from sahi.utils.coco import Coco

        coco_dict_path = "tests/data/coco_utils/combined_coco.json"
        image_dir = "tests/data/coco_utils/"
        output_dir = "tests/data/coco2yolo/"
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        coco = Coco.from_coco_dict_or_path(coco_dict_path, image_dir=image_dir)
        coco.export_as_yolo(output_dir=output_dir, train_split_rate=0.5, numpy_seed=0)

    def test_update_categories(self):
        coco_path = "tests/data/coco_utils/terrain2_coco.json"
        source_coco_dict = load_json(coco_path)

        self.assertEqual(len(source_coco_dict["annotations"]), 5)
        self.assertEqual(len(source_coco_dict["images"]), 1)
        self.assertEqual(len(source_coco_dict["categories"]), 1)
        self.assertEqual(source_coco_dict["categories"], [{"id": 1, "name": "car", "supercategory": "car"}])
        self.assertEqual(source_coco_dict["annotations"][1]["category_id"], 1)

        # update categories
        desired_name2id = {"human": 1, "car": 2, "big_vehicle": 3}
        target_coco_dict = update_categories(desired_name2id=desired_name2id, coco_dict=source_coco_dict)

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

    def test_coco_update_categories(self):
        from sahi.utils.coco import Coco

        coco_path = "tests/data/coco_utils/terrain2_coco.json"
        image_dir = "tests/data/coco_utils/"
        coco = Coco.from_coco_dict_or_path(coco_path, image_dir=image_dir)

        self.assertEqual(len(coco.json["annotations"]), 5)
        self.assertEqual(len(coco.json["images"]), 1)
        self.assertEqual(len(coco.json["categories"]), 1)
        self.assertEqual(coco.json["categories"], [{"id": 1, "name": "car", "supercategory": "car"}])
        self.assertEqual(coco.json["annotations"][1]["category_id"], 1)
        self.assertEqual(coco.image_dir, image_dir)
        assert coco.stats
        self.assertEqual(coco.stats["num_images"], len(coco.images))
        self.assertEqual(coco.stats["num_annotations"], len(coco.json["annotations"]))

        # update categories
        desired_name2id = {"human": 1, "car": 2, "big_vehicle": 3}
        coco.update_categories(desired_name2id=desired_name2id)

        self.assertEqual(len(coco.json["annotations"]), 5)
        self.assertEqual(len(coco.json["images"]), 1)
        self.assertEqual(len(coco.json["categories"]), 3)
        self.assertEqual(
            coco.json["categories"],
            [
                {"id": 1, "name": "human", "supercategory": "human"},
                {"id": 2, "name": "car", "supercategory": "car"},
                {"id": 3, "name": "big_vehicle", "supercategory": "big_vehicle"},
            ],
        )
        self.assertEqual(coco.json["annotations"][1]["category_id"], 2)
        self.assertEqual(coco.image_dir, image_dir)
        self.assertEqual(coco.stats["num_images"], len(coco.images))
        self.assertEqual(coco.stats["num_annotations"], len(coco.json["annotations"]))

    def test_get_imageid2annotationlist_mapping(self):
        from sahi.utils.coco import get_imageid2annotationlist_mapping

        coco_path = "tests/data/coco_utils/combined_coco.json"
        coco_dict = load_json(coco_path)
        imageid2annotationlist_mapping = get_imageid2annotationlist_mapping(coco_dict)
        self.assertEqual(len(imageid2annotationlist_mapping), 2)

        def check_image_id(image_id):
            logger.debug(f"{type(imageid2annotationlist_mapping[image_id][0])}")
            image_ids = []
            for annotationlist in imageid2annotationlist_mapping[image_id]:
                # TODO: get_imageid2annotationlist_mapping is supposed to get CocoAnnotation, not a dict
                assert isinstance(annotationlist, dict)
                image_ids.append(annotationlist["image_id"])
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
        self.assertEqual(merged_coco_dict["annotations"][7]["image_id"], 3)
        self.assertEqual(merged_coco_dict["annotations"][7]["id"], 9)

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
        self.assertEqual(merged_coco_dict["annotations"][12]["bbox"], coco_dict3["annotations"][0]["bbox"])
        self.assertEqual(merged_coco_dict["annotations"][12]["id"], 15)
        self.assertEqual(merged_coco_dict["annotations"][12]["image_id"], 5)
        self.assertEqual(merged_coco_dict["annotations"][9]["category_id"], 1)
        self.assertEqual(merged_coco_dict["annotations"][9]["image_id"], 3)

    def test_coco_merge(self):
        from sahi.utils.coco import Coco

        # load coco files to be combined
        coco_path1 = "tests/data/coco_utils/terrain1_coco.json"
        coco_path2 = "tests/data/coco_utils/terrain2_coco.json"
        coco_path3 = "tests/data/coco_utils/terrain3_coco.json"
        image_dir = "tests/data/coco_utils/"
        coco1 = Coco.from_coco_dict_or_path(coco_path1, image_dir=image_dir)
        coco2 = Coco.from_coco_dict_or_path(coco_path2, image_dir=image_dir)
        coco3 = Coco.from_coco_dict_or_path(coco_path3, image_dir=image_dir)
        coco1.merge(coco2)
        coco1.merge(coco3)
        self.assertEqual(len(coco1.json["images"]), 3)
        self.assertEqual(len(coco1.json["annotations"]), 22)
        self.assertEqual(len(coco1.json["categories"]), 2)
        self.assertEqual(len(coco1.images), 3)

        self.assertEqual(coco1.json["annotations"][12]["id"], 13)
        self.assertEqual(coco1.json["annotations"][12]["image_id"], 3)
        self.assertEqual(coco1.json["annotations"][9]["category_id"], 1)
        self.assertEqual(coco1.json["annotations"][9]["image_id"], 2)
        self.assertEqual(coco1.image_dir, image_dir)
        self.assertEqual(coco2.image_dir, image_dir)
        assert coco2.stats
        self.assertEqual(coco2.stats["num_images"], len(coco2.images))
        self.assertEqual(coco2.stats["num_annotations"], len(coco2.json["annotations"]))

    def test_get_subsampled_coco(self):
        from sahi.utils.coco import Coco

        coco_path = "tests/data/coco_utils/visdrone2019-det-train-first50image.json"
        image_dir = "tests/data/coco_utils/"
        SUBSAMPLE_RATIO = 5
        coco = Coco.from_coco_dict_or_path(coco_path, image_dir=image_dir)
        subsampled_coco = coco.get_subsampled_coco(subsample_ratio=SUBSAMPLE_RATIO)
        self.assertEqual(len(coco.json["images"]), 50)
        self.assertEqual(len(subsampled_coco.json["images"]), 10)
        self.assertEqual(len(coco.images[5].annotations), len(subsampled_coco.images[1].annotations))
        self.assertEqual(len(coco.images[5].annotations), len(subsampled_coco.images[1].annotations))
        self.assertEqual(coco.image_dir, image_dir)
        self.assertEqual(subsampled_coco.image_dir, image_dir)
        assert subsampled_coco.stats
        self.assertEqual(subsampled_coco.stats["num_images"], len(subsampled_coco.images))
        self.assertEqual(subsampled_coco.stats["num_annotations"], len(subsampled_coco.json["annotations"]))

        vehicle_subsampled_coco = coco.get_subsampled_coco(subsample_ratio=SUBSAMPLE_RATIO, category_id=1)
        assert vehicle_subsampled_coco.stats
        assert coco.stats
        self.assertEqual(
            vehicle_subsampled_coco.stats["num_images_per_category"]["vehicle"],
            int(coco.stats["num_images_per_category"]["vehicle"] / SUBSAMPLE_RATIO) + 1,
        )

        negative_subsampled_coco = coco.get_subsampled_coco(subsample_ratio=SUBSAMPLE_RATIO, category_id=-1)
        assert negative_subsampled_coco.stats
        self.assertEqual(
            negative_subsampled_coco.stats["num_images_per_category"]["vehicle"],
            coco.stats["num_images_per_category"]["vehicle"],
        )
        self.assertEqual(
            negative_subsampled_coco.stats["num_negative_images"],
            int(coco.stats["num_negative_images"] / SUBSAMPLE_RATIO) + 1,
        )

    def test_get_upsampled_coco(self):
        from sahi.utils.coco import Coco

        coco_path = "tests/data/coco_utils/visdrone2019-det-train-first50image.json"
        image_dir = "tests/data/coco_utils/"
        coco = Coco.from_coco_dict_or_path(coco_path, image_dir=image_dir)
        UPSAMPLE_RATIO = 5
        upsampled_coco = coco.get_upsampled_coco(upsample_ratio=UPSAMPLE_RATIO)
        self.assertEqual(len(coco.json["images"]), 50)
        self.assertEqual(len(upsampled_coco.json["images"]), 250)
        self.assertEqual(len(coco.images[5].annotations), len(upsampled_coco.images[5 + len(coco.images)].annotations))
        self.assertEqual(coco.image_dir, image_dir)
        self.assertEqual(upsampled_coco.image_dir, image_dir)
        assert upsampled_coco.stats
        assert coco.stats
        self.assertEqual(
            upsampled_coco.stats["num_images_per_category"]["vehicle"],
            coco.stats["num_images_per_category"]["vehicle"] * UPSAMPLE_RATIO,
        )
        self.assertEqual(upsampled_coco.stats["num_images"], len(upsampled_coco.images))
        self.assertEqual(upsampled_coco.stats["num_annotations"], len(upsampled_coco.json["annotations"]))

        vehicle_upsampled_coco = coco.get_upsampled_coco(upsample_ratio=UPSAMPLE_RATIO, category_id=1)
        assert vehicle_upsampled_coco.stats
        self.assertEqual(
            vehicle_upsampled_coco.stats["num_images_per_category"]["vehicle"],
            coco.stats["num_images_per_category"]["vehicle"] * UPSAMPLE_RATIO,
        )
        self.assertNotEqual(
            vehicle_upsampled_coco.stats["num_images_per_category"]["human"],
            coco.stats["num_images_per_category"]["human"] * UPSAMPLE_RATIO,
        )

        negative_upsampled_coco = coco.get_upsampled_coco(upsample_ratio=UPSAMPLE_RATIO, category_id=-1)
        assert negative_upsampled_coco.stats
        self.assertEqual(
            negative_upsampled_coco.stats["num_images_per_category"]["vehicle"],
            coco.stats["num_images_per_category"]["vehicle"],
        )
        self.assertEqual(
            negative_upsampled_coco.stats["num_negative_images"], coco.stats["num_negative_images"] * UPSAMPLE_RATIO
        )

    def test_get_area_filtered_coco(self):
        from sahi.utils.coco import Coco

        coco_path = "tests/data/coco_utils/visdrone2019-det-train-first50image.json"
        image_dir = "tests/data/coco_utils/"
        min_area = 50
        max_area = 10000
        coco = Coco.from_coco_dict_or_path(coco_path, image_dir=image_dir)
        area_filtered_coco = coco.get_area_filtered_coco(min=min_area, max=max_area)
        self.assertEqual(len(coco.json["images"]), 50)
        self.assertEqual(len(area_filtered_coco.json["images"]), 17)
        assert area_filtered_coco.stats
        self.assertGreater(area_filtered_coco.stats["min_annotation_area"], min_area)
        self.assertLess(area_filtered_coco.stats["max_annotation_area"], max_area)
        self.assertEqual(area_filtered_coco.image_dir, image_dir)
        self.assertEqual(area_filtered_coco.stats["num_images"], len(area_filtered_coco.images))
        self.assertEqual(area_filtered_coco.stats["num_annotations"], len(area_filtered_coco.json["annotations"]))

        intervals_per_category = {
            "human": {"min": 20, "max": 10000},
            "vehicle": {"min": 50, "max": 15000},
        }
        area_filtered_coco = coco.get_area_filtered_coco(intervals_per_category=intervals_per_category)

        self.assertEqual(len(coco.json["images"]), 50)
        self.assertEqual(len(area_filtered_coco.json["images"]), 24)
        assert area_filtered_coco.stats
        self.assertGreater(
            area_filtered_coco.stats["min_annotation_area"],
            min(intervals_per_category["human"]["min"], intervals_per_category["vehicle"]["min"]),
        )
        self.assertLess(
            area_filtered_coco.stats["max_annotation_area"],
            max(intervals_per_category["human"]["max"], intervals_per_category["vehicle"]["max"]),
        )
        self.assertEqual(area_filtered_coco.image_dir, image_dir)
        self.assertEqual(area_filtered_coco.stats["num_images"], len(area_filtered_coco.images))
        self.assertEqual(area_filtered_coco.stats["num_annotations"], len(area_filtered_coco.json["annotations"]))

        intervals_per_category = {
            "human": {"min": 20, "max": 10000},
            "vehicle": {"min": 50, "max": 15000},
        }
        area_filtered_coco = coco.get_area_filtered_coco(intervals_per_category=intervals_per_category)

        self.assertEqual(len(coco.json["images"]), 50)
        self.assertEqual(len(area_filtered_coco.json["images"]), 24)
        assert area_filtered_coco.stats
        self.assertGreater(
            area_filtered_coco.stats["min_annotation_area"],
            min(intervals_per_category["human"]["min"], intervals_per_category["vehicle"]["min"]),
        )
        self.assertLess(
            area_filtered_coco.stats["max_annotation_area"],
            max(intervals_per_category["human"]["max"], intervals_per_category["vehicle"]["max"]),
        )
        self.assertEqual(area_filtered_coco.image_dir, image_dir)
        self.assertEqual(area_filtered_coco.stats["num_images"], len(area_filtered_coco.images))
        self.assertEqual(area_filtered_coco.stats["num_annotations"], len(area_filtered_coco.json["annotations"]))

    def test_export_coco_as_yolov5(self):
        from sahi.utils.coco import Coco, export_coco_as_yolo

        coco_dict_path = "tests/data/coco_utils/combined_coco.json"
        image_dir = "tests/data/coco_utils/"
        output_dir = "tests/data/export_coco_as_yolov5/"
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)
        coco = Coco.from_coco_dict_or_path(coco_dict_path, image_dir=image_dir)
        export_coco_as_yolo(output_dir=output_dir, train_coco=coco, val_coco=coco, numpy_seed=0)

    def test_cocovid(self):
        # from sahi.utils.coco import CocoVid
        # TODO
        pass

    def test_bbox_clipping(self):
        from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage

        coco = Coco()
        coco.add_category(CocoCategory(id=0, name="box", supercategory="box"))
        cocoimg = CocoImage(file_name="bboxes.jpg", height=100, width=100)
        cocoimg2 = CocoImage(file_name="sample_photo.png", height=1080, width=1920)  # negative img
        cocoann = CocoAnnotation(
            bbox=[60, 20, 10, 70], category_id=0, category_name="box", image_id=1
        )  # bbox totally inside img
        cocoann2 = CocoAnnotation(
            bbox=[120, 110, 60, 30], category_id=0, category_name="box", image_id=1
        )  # bbox totally outside img
        cocoann3 = CocoAnnotation(bbox=[-50, -20, 80, 80], category_id=0, category_name="box", image_id=1)  # x<0   y<0
        cocoann4 = CocoAnnotation(
            bbox=[50, -50, 60, 60], category_id=0, category_name="box", image_id=1
        )  #  y<0   x+w > imwidth
        cocoann5 = CocoAnnotation(
            bbox=[-50, 50, 70, 70], category_id=0, category_name="box", image_id=1
        )  #  x<0   y+h > imheight
        cocoann6 = CocoAnnotation(
            bbox=[80, 80, 50, 50], category_id=0, category_name="box", image_id=1
        )  # x+w > imwidth y+h > imheight
        cocoann7 = CocoAnnotation(
            bbox=[-70, -70, 200, 200], category_id=0, category_name="box", image_id=1
        )  # bbox totally enclosing img

        cocoimg.add_annotation(cocoann)
        cocoimg.add_annotation(cocoann2)
        cocoimg.add_annotation(cocoann3)
        cocoimg.add_annotation(cocoann4)
        cocoimg.add_annotation(cocoann5)
        cocoimg.add_annotation(cocoann6)
        cocoimg.add_annotation(cocoann7)
        coco.add_image(cocoimg)
        coco.add_image(cocoimg2)

        coco_with_clipped_bboxes = coco.get_coco_with_clipped_bboxes()

        self.assertEqual(coco_with_clipped_bboxes.images[0].annotations[0].bbox, [60, 20, 10, 70])
        self.assertEqual(coco_with_clipped_bboxes.images[0].annotations[1].bbox, [0, 0, 30, 60])
        self.assertEqual(coco_with_clipped_bboxes.images[0].annotations[2].bbox, [50, 0, 50, 10])
        self.assertEqual(coco_with_clipped_bboxes.images[0].annotations[3].bbox, [0, 50, 20, 50])
        self.assertEqual(coco_with_clipped_bboxes.images[0].annotations[4].bbox, [80, 80, 20, 20])
        self.assertEqual(coco_with_clipped_bboxes.images[0].annotations[5].bbox, [0, 0, 100, 100])
        self.assertIsNotNone(coco_with_clipped_bboxes.images[1])
        self.assertEqual(len(coco_with_clipped_bboxes.images[1].annotations), 0)


if __name__ == "__main__":
    unittest.main()
