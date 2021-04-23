# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.
# Modified by Sinan O Altinuc, 2020.

import copy
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
from sahi.utils.file import get_base_filename, load_json, save_json
from sahi.utils.shapely import ShapelyAnnotation, get_shapely_multipolygon
from tqdm import tqdm


class CocoCategory:
    """
    COCO formatted category.
    """

    def __init__(self, id=None, name=None, supercategory=None):
        self.id = int(id)
        self.name = name
        self.supercategory = supercategory if supercategory else name

    @classmethod
    def from_coco_category(cls, category):
        """
        Creates CocoCategory object using coco category.

        Args:
            category: Dict
                {"supercategory": "person", "id": 1, "name": "person"},
        """
        return cls(
            id=category["id"],
            name=category["name"],
            supercategory=category["supercategory"],
        )

    @property
    def json(self):
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }

    def __repr__(self):
        return f"""CocoCategory<
    id: {self.id},
    name: {self.name},
    supercategory: {self.supercategory}>"""


class CocoAnnotation:
    """
    COCO formatted annotation.
    """

    @classmethod
    def from_coco_segmentation(
        cls, segmentation, category_id, category_name, iscrowd=0
    ):
        """
        Creates CocoAnnotation object using coco segmentation.

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            iscrowd: int
                0 or 1
        """
        return cls(
            segmentation=segmentation,
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_bbox(cls, bbox, category_id, category_name, iscrowd=0):
        """
        Creates CocoAnnotation object using coco bbox

        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            iscrowd: int
                0 or 1
        """
        return cls(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_annotation_dict(cls, category_name, annotation_dict):
        """
        Creates CocoAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            category_name: str
                Category name of the annotation
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
        """
        if annotation_dict["segmentation"]:
            return cls(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )
        else:
            return cls(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )

    def __init__(
        self,
        segmentation=None,
        bbox=None,
        category_id=None,
        category_name=None,
        image_id=None,
        iscrowd=0,
    ):
        """
        Creates coco annotation object using bbox or segmentation

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            iscrowd: int
                0 or 1
        """
        assert bbox or segmentation, "you must provide a bbox or polygon"

        self._segmentation = segmentation
        self._bbox = [int(coord) for coord in bbox] if bbox else bbox
        self._category_id = category_id
        self._category_name = category_name
        self._image_id = image_id
        self._iscrowd = iscrowd

        if self._segmentation:
            shapely_annotation = ShapelyAnnotation.from_coco_segmentation(
                segmentation=self._segmentation
            )
        else:
            shapely_annotation = ShapelyAnnotation.from_coco_bbox(bbox=self._bbox)
        self._shapely_annotation = shapely_annotation

    @property
    def area(self):
        """
        Returns area of annotation polygon (or bbox if no polygon available)
        """
        return self._shapely_annotation.area

    @property
    def bbox(self):
        """
        Returns coco formatted bbox of the annotation as [xmin, ymin, width, height]
        """
        return self._shapely_annotation.to_coco_bbox()

    @property
    def segmentation(self):
        """
        Returns coco formatted segmentation of the annotation as [[1, 1, 325, 125, 250, 200, 5, 200]]
        """
        if self._segmentation:
            return self._shapely_annotation.to_coco_segmentation()
        else:
            return []

    @property
    def category_id(self):
        """
        Returns category id of the annotation as int
        """
        return self._category_id

    @category_id.setter
    def category_id(self, i):
        if not isinstance(i, int):
            raise Exception("category_id must be an integer")
        self._category_id = i

    @property
    def image_id(self):
        """
        Returns image id of the annotation as int
        """
        return self._image_id

    @image_id.setter
    def image_id(self, i):
        if not isinstance(i, int):
            raise Exception("image_id must be an integer")
        self._image_id = i

    @property
    def category_name(self):
        """
        Returns category name of the annotation as str
        """
        return self._category_name

    @category_name.setter
    def category_name(self, n):
        if not isinstance(n, str):
            raise Exception("category_name must be a string")
        self._category_name = n

    @property
    def iscrowd(self):
        """
        Returns iscrowd info of the annotation
        """
        return self._iscrowd

    @property
    def json(self):
        return {
            "image_id": self.image_id,
            "bbox": self.bbox,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "segmentation": self.segmentation,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    def serialize(self):
        print(".serialize() is deprectaed, use .json instead")

    def __repr__(self):
        return f"""CocoAnnotation<
    image_id: {self.image_id},
    bbox: {self.bbox},
    segmentation: {self.segmentation},
    category_id: {self.category_id},
    category_name: {self.category_name},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""


class CocoPrediction(CocoAnnotation):
    """
    Class for handling predictions in coco format.
    """

    @classmethod
    def from_coco_segmentation(
        cls,
        segmentation,
        category_id,
        category_name,
        score,
        iscrowd=0,
    ):
        """
        Creates CocoAnnotation object using coco segmentation.

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            score: float
                Prediction score between 0 and 1
            iscrowd: int
                0 or 1
        """
        return cls(
            segmentation=segmentation,
            category_id=category_id,
            category_name=category_name,
            score=score,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_bbox(cls, bbox, category_id, category_name, score, iscrowd=0):
        """
        Creates CocoAnnotation object using coco bbox

        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            score: float
                Prediction score between 0 and 1
            iscrowd: int
                0 or 1
        """
        return cls(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            score=score,
            iscrowd=iscrowd,
        )

    @classmethod
    def from_coco_annotation_dict(cls, category_name, annotation_dict, score):
        """
        Creates CocoAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            category_name: str
                Category name of the annotation
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
            score: float
                Prediction score between 0 and 1
        """
        if annotation_dict["segmentation"]:
            return cls(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                score=score,
            )
        else:
            return cls(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
            )

    def __init__(
        self,
        segmentation=None,
        bbox=None,
        category_id=None,
        category_name=None,
        image_id=None,
        score=None,
        iscrowd=0,
    ):
        """

        Args:
            segmentation: List[List]
                [[1, 1, 325, 125, 250, 200, 5, 200]]
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            score: float
                Prediction score between 0 and 1
            iscrowd: int
                0 or 1
        """
        self.score = score
        super().__init__(
            segmentation=segmentation,
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            image_id=image_id,
            iscrowd=iscrowd,
        )

    @property
    def json(self):
        return {
            "image_id": self.image_id,
            "bbox": self.bbox,
            "score": self.score,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "segmentation": self.segmentation,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    def serialize(self):
        print(".serialize() is deprectaed, use .json instead")

    def __repr__(self):
        return f"""CocoPrediction<
    image_id: {self.image_id},
    bbox: {self.bbox},
    segmentation: {self.segmentation},
    score: {self.score},
    category_id: {self.category_id},
    category_name: {self.category_name},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""


class CocoVidAnnotation(CocoAnnotation):
    """
    COCOVid formatted annotation.
    https://github.com/open-mmlab/mmtracking/blob/master/docs/tutorials/customize_dataset.md#the-cocovid-annotation-file
    """

    def __init__(
        self,
        bbox=None,
        category_id=None,
        category_name=None,
        image_id=None,
        instance_id=None,
        iscrowd=0,
        id=None,
    ):
        """
        Args:
            bbox: List
                [xmin, ymin, width, height]
            category_id: int
                Category id of the annotation
            category_name: str
                Category name of the annotation
            image_id: int
                Image ID of the annotation
            instance_id: int
                Used for tracking
            iscrowd: int
                0 or 1
            id: int
                Annotation id
        """
        super(CocoVidAnnotation, self).__init__(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            image_id=image_id,
            iscrowd=iscrowd,
        )
        self.instance_id = instance_id
        self.id = id

    @property
    def json(self):
        return {
            "id": self.id,
            "image_id": self.image_id,
            "bbox": self.bbox,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "instance_id": self.instance_id,
            "iscrowd": self.iscrowd,
            "area": self.area,
        }

    def __repr__(self):
        return f"""CocoAnnotation<
    id: {self.id},
    image_id: {self.image_id},
    bbox: {self.bbox},
    category_id: {self.category_id},
    category_name: {self.category_name},
    instance_id: {self.instance_id},
    iscrowd: {self.iscrowd},
    area: {self.area}>"""


class CocoImage:
    @classmethod
    def from_coco_image_dict(cls, image_dict):
        """
        Creates CocoImage object from COCO formatted image dict (with fields "id", "file_name", "height" and "weight").

        Args:
            image_dict: dict
                COCO formatted image dict (with fields "id", "file_name", "height" and "weight")
        """
        return cls(
            id=image_dict["id"],
            file_name=image_dict["file_name"],
            height=image_dict["height"],
            width=image_dict["width"],
        )

    def __init__(self, file_name: str, height: int, width: int, id: int = None):
        """
        Creates CocoImage object

        Args:
            id : int
                Image id
            file_name : str
                Image path
            height : int
                Image height in pixels
            width : int
                Image width in pixels
        """
        self.id = int(id) if id else id
        self.file_name = file_name
        self.height = int(height)
        self.width = int(width)
        self.annotations = []  # list of CocoAnnotation that belong to this image

    def add_annotation(self, annotation):
        """
        Adds annotation to this CocoImage instance

        annotation : CocoAnnotation
        """

        assert isinstance(
            annotation, CocoAnnotation
        ), "annotation must be a CocoAnnotation instance"
        self.annotations.append(annotation)

    def __repr__(self):
        return f"""CocoImage<
    id: {self.id},
    file_name: {self.file_name},
    height: {self.height},
    width: {self.width},
    annotations: List[CocoAnnotation]>"""


class CocoVidImage(CocoImage):
    """
    COCOVid formatted image.
    https://github.com/open-mmlab/mmtracking/blob/master/docs/tutorials/customize_dataset.md#the-cocovid-annotation-file
    """

    def __init__(
        self,
        file_name,
        height,
        width,
        video_id=None,
        frame_id=None,
        id=None,
    ):
        """
        Creates CocoVidImage object

        Args:
            id: int
                Image id
            file_name: str
                Image path
            height: int
                Image height in pixels
            width: int
                Image width in pixels
            frame_id: int
                0-indexed frame id
            video_id: int
                Video id
        """
        super(CocoVidImage, self).__init__(
            file_name=file_name, height=height, width=width, id=id
        )
        self.frame_id = frame_id
        self.video_id = video_id

    @classmethod
    def from_coco_image(cls, coco_image, video_id=None, frame_id=None):
        """
        Creates CocoVidImage object using CocoImage object.
        Args:
            coco_image: CocoImage
            frame_id: int
                0-indexed frame id
            video_id: int
                Video id

        """
        return cls(
            file_name=coco_image.file_name,
            height=coco_image.height,
            width=coco_image.width,
            id=coco_image.id,
            video_id=video_id,
            frame_id=frame_id,
        )

    def add_annotation(self, annotation):
        """
        Adds annotation to this CocoImage instance
        annotation : CocoVidAnnotation
        """

        assert (
            type(annotation) == CocoVidAnnotation
        ), "annotation must be a CocoVidAnnotation instance"
        self.annotations.append(annotation)

    @property
    def json(self):
        return {
            "file_name": self.file_name,
            "height": self.height,
            "width": self.width,
            "id": self.id,
            "video_id": self.video_id,
            "frame_id": self.frame_id,
        }

    def __repr__(self):
        return f"""CocoVidImage<
    file_name: {self.file_name},
    height: {self.height},
    width: {self.width},
    id: {self.id},
    video_id: {self.video_id},
    frame_id: {self.frame_id},
    annotations: List[CocoVidAnnotation]>"""


class CocoVideo:
    """
    COCO formatted video.
    https://github.com/open-mmlab/mmtracking/blob/master/docs/tutorials/customize_dataset.md#the-cocovid-annotation-file
    """

    def __init__(
        self,
        name: str,
        id: int = None,
        fps: float = None,
        height: int = None,
        width: int = None,
    ):
        """
        Creates CocoVideo object

        Args:
            name: str
                Video name
            id: int
                Video id
            fps: float
                Video fps
            height: int
                Video height in pixels
            width: int
                Video width in pixels
        """
        self.name = name
        self.id = id
        self.fps = fps
        self.height = height
        self.width = width
        self.images = []  # list of CocoImage that belong to this video

    def add_image(self, image):
        """
        Adds image to this CocoVideo instance
        Args:
            image: CocoImage
        """

        assert type(image) == CocoImage, "image must be a CocoImage instance"

        self.images.append(CocoVidImage.from_coco_image(image))

    def add_cocovidimage(self, cocovidimage):
        """
        Adds CocoVidImage to this CocoVideo instance
        Args:
            cocovidimage: CocoVidImage
        """

        assert (
            type(cocovidimage) == CocoVidImage
        ), "cocovidimage must be a CocoVidImage instance"

        self.images.append(cocovidimage)

    @property
    def json(self):
        return {
            "name": self.name,
            "id": self.id,
            "fps": self.fps,
            "height": self.height,
            "width": self.width,
        }

    def __repr__(self):
        return f"""CocoVideo<
    id: {self.id},
    name: {self.name},
    fps: {self.fps},
    height: {self.height},
    width: {self.width},
    images: List[CocoVidImage]>"""


class Coco:
    def __init__(self, name=None, remapping_dict=None):
        """
        Creates Coco object.

        Args:
            name: str
                Name of the Coco dataset, it determines exported json name.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
        """
        self.name = name
        self.remapping_dict = remapping_dict  # TODO: utilize remapping_dict
        self.categories = []
        self.images = []

    def add_categories_from_coco_category_list(self, coco_category_list):
        """
        Creates CocoCategory object using coco category list.

        Args:
            coco_category_list: List[Dict]
                [
                    {"supercategory": "person", "id": 1, "name": "person"},
                    {"supercategory": "vehicle", "id": 2, "name": "bicycle"}
                ]
        """

        for coco_category in coco_category_list:
            if self.remapping_dict is not None:
                for source_id in self.remapping_dict.keys():
                    if coco_category["id"] == source_id:
                        target_id = self.remapping_dict[source_id]
                        coco_category["id"] = target_id

            self.add_category(CocoCategory.from_coco_category(coco_category))

    def add_category(self, category):
        """
        Adds category to this Coco instance

        Args:
            category: CocoCategory
        """

        assert (
            type(category) == CocoCategory
        ), "category must be a CocoCategory instance"

        self.categories.append(category)

    def add_image(self, image):
        """
        Adds image to this Coco instance

        Args:
            image: CocoImage
        """

        assert type(image) == CocoImage, "image must be a CocoImage instance"

        self.images.append(image)

    @classmethod
    def from_coco_dict_or_path(cls, coco_dict_or_path, desired_name2id=None):
        """
        Creates coco object from COCO formatted dict or COCO dataset file path.

        Args:
            coco_dict_or_path: dict/str or List[dict/str]
                COCO formatted dict or COCO dataset file path
                List of COCO formatted dict or COCO dataset file path
            desired_name2id : dict
                {"human": 1, "car": 2, "big_vehicle": 3}

        Properties:
            images: list of CocoImage
            category_mapping: dict
        """
        # init coco object
        coco = cls()

        if type(coco_dict_or_path) == list:  # merge coco datasets if given as list
            # create coco_dict_list
            coco_dict_list = []
            coco_dict_or_path_list = copy.deepcopy(coco_dict_or_path)
            for coco_dict_or_path in coco_dict_or_path_list:
                # load coco dataset dict
                if type(coco_dict_or_path) == str:
                    coco_dict = load_json(coco_dict_or_path)
                else:
                    coco_dict = coco_dict_or_path
                # append to list
                coco_dict_list.append(coco_dict)
            # merge coco dicts
            coco_dict = merge_from_list(coco_dict_list, desired_name2id=None)
        else:
            # load coco dict if path is given
            if type(coco_dict_or_path) == str:
                coco_dict = load_json(coco_dict_or_path)
            else:
                coco_dict = coco_dict_or_path

        # arrange image id to annotation id mapping
        coco.add_categories_from_coco_category_list(coco_dict["categories"])
        imageid2annotationlist = get_imageid2annotationlist_mapping(coco_dict)
        category_mapping = coco.category_mapping

        coco_image_list = []
        for coco_image_dict in coco_dict["images"]:
            coco_image = CocoImage.from_coco_image_dict(coco_image_dict)
            annotation_list = imageid2annotationlist[coco_image_dict["id"]]
            for coco_annotation_dict in annotation_list:
                category_name = category_mapping[coco_annotation_dict["category_id"]]
                coco_annotation = CocoAnnotation.from_coco_annotation_dict(
                    category_name=category_name, annotation_dict=coco_annotation_dict
                )
                coco_image.add_annotation(coco_annotation)
            coco_image_list.append(coco_image)

        coco.images = coco_image_list
        return coco

    @property
    def json_categories(self):
        categories = []
        for category in self.categories:
            categories.append(category.json)
        return categories

    @property
    def category_mapping(self):
        category_mapping = {}
        for category in self.categories:
            category_mapping[category.id] = category.name
        return category_mapping

    @property
    def imageid2annotationlist(self):
        return get_imageid2annotationlist_mapping(self.json)

    @property
    def json(self):
        return create_coco_dict(
            images=self.images,
            categories=self.json_categories,
            ignore_negative_samples=True,
        )

    def split_coco_as_train_val(
        self, file_name=None, target_dir=None, train_split_rate=0.9, numpy_seed=0
    ):
        """
        Split images into train-val and saves as seperate coco dataset files.

        Args:
            file_name: str
            target_dir: str
            train_split_rate: float
            numpy_seed: int
                To fix the numpy seed.

        Returns:
            result : dict
                {
                    "train_dict": "",
                    "val_dict": "",
                    "train_path": "",
                    "val_path": "",
                }
        """
        # fix numpy numpy seed
        np.random.seed(numpy_seed)

        # set output coco file name
        if file_name:
            None
        elif target_dir:
            raise ValueError("file_name should be specified.")

        # divide images
        num_images = len(self.images)
        shuffled_images = copy.deepcopy(self.images)
        np.random.shuffle(shuffled_images)
        num_train = int(num_images * train_split_rate)
        train_images = shuffled_images[:num_train]
        val_images = shuffled_images[num_train:]

        # form train val coco dicts
        train_coco_dict = create_coco_dict(
            images=train_images,
            categories=self.json_categories,
            ignore_negative_samples=False,
        )
        val_coco_dict = create_coco_dict(
            images=val_images,
            categories=self.json_categories,
            ignore_negative_samples=False,
        )
        # return result
        if not target_dir:
            return {
                "train_dict": train_coco_dict,
                "val_dict": val_coco_dict,
                "train_path": "",
                "val_path": "",
            }
        else:
            train_coco_dict_path = os.path.join(target_dir, file_name + "_train.json")
            save_json(train_coco_dict, train_coco_dict_path)
            val_coco_dict_path = os.path.join(target_dir, file_name + "_val.json")
            save_json(val_coco_dict, val_coco_dict_path)
            return {
                "train_dict": train_coco_dict,
                "val_dict": val_coco_dict,
                "train_path": train_coco_dict_path,
                "val_path": val_coco_dict_path,
            }

    def export_as_yolov5(self, image_dir, output_dir, train_split_rate=1, numpy_seed=0):
        """
        Exports current COCO dataset in ultralytics/yolov5 format.
        Creates train val folders with image symlinks and txt files and a data yaml file.

        Args:
            image_dir: str
                Source image directory that contains coco images.
            output_dir: str
                Export directory.
            train_split_rate: float
                If given 1, will be exported as train split.
                If given 0, will be exported as val split.
                If in between 0-1, both train/val splits will be calculated and exported.
            numpy_seed: int
                To fix the numpy seed.
        """
        import yaml

        # set split_mode
        if 0 < train_split_rate and train_split_rate < 1:
            split_mode = "TRAINVAL"
        elif train_split_rate == 0:
            split_mode = "VAL"
        elif train_split_rate == 1:
            split_mode = "TRAIN"
        else:
            ValueError("train_split_rate cannot be <0 or >1")

        # split dataset
        if split_mode == "TRAINVAL":
            result = self.split_coco_as_train_val(
                file_name=None,
                target_dir=None,
                train_split_rate=train_split_rate,
                numpy_seed=numpy_seed,
            )
            train_coco_dict = result["train_dict"]
            val_coco_dict = result["val_dict"]
        elif split_mode == "TRAIN":
            train_coco_dict = self.json
            val_coco_dict = None
        elif split_mode == "VAL":
            train_coco_dict = None
            val_coco_dict = self.json

        # create train val image dirs
        train_dir = ""
        val_dir = ""
        if split_mode in ["TRAINVAL", "TRAIN"]:
            train_dir = Path(os.path.abspath(output_dir)) / "train/"
            train_dir.mkdir(parents=True, exist_ok=True)  # create dir
        if split_mode in ["TRAINVAL", "VAL"]:
            val_dir = Path(os.path.abspath(output_dir)) / "val/"
            val_dir.mkdir(parents=True, exist_ok=True)  # create dir

        # create image symlinks and annotation txts
        if split_mode in ["TRAINVAL", "TRAIN"]:
            export_yolov5_images_and_txts_from_coco_dict(
                image_dir, output_dir=train_dir, coco_dict_or_path=train_coco_dict
            )
        if split_mode in ["TRAINVAL", "VAL"]:
            export_yolov5_images_and_txts_from_coco_dict(
                image_dir, output_dir=val_dir, coco_dict_or_path=val_coco_dict
            )

        # create yolov5 data yaml
        data = {
            "train": train_dir,
            "val": val_dir,
            "nc": len(self.category_mapping),
            "names": list(self.category_mapping.values()),
        }
        yaml_path = str(Path(output_dir) / "data.yml")
        with open(yaml_path, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=None)


def export_yolov5_images_and_txts_from_coco_dict(
    image_dir, output_dir, coco_dict_or_path
):
    """
    Creates image symlinks and annotation txts in yolo format from coco dataset.

    Args:
        image_dir: str
            Source image directory that contains coco images.
        output_dir: str
            Export directory.
        coco_dict_or_path: str or dict
            Path for the coco dataset file or coco dataset as python dictionary.
    """
    # create coco instance from coco_dict_or_path
    coco = Coco.from_coco_dict_or_path(coco_dict_or_path)

    for image in coco.images:
        # Create a symbolic link pointing to src named dst
        src = os.path.abspath(os.path.join(image_dir, image.file_name))
        dst = os.path.join(output_dir, image.file_name)
        os.symlink(src, dst)
        # calculate annotation normalization ratios
        width = image.width
        height = image.height
        dw = 1.0 / (width)
        dh = 1.0 / (height)
        # set annotation filepath
        file_name = get_base_filename(image.file_name)[1]
        yolo_annotation_filepath = "{}.txt".format(os.path.join(output_dir, file_name))
        # create annotation file
        annotations = image.annotations
        if len(annotations) > 0:
            for annotation in annotations:
                # convert coco bbox to yolo bbox
                x_center = annotation.bbox[0] + annotation.bbox[2] / 2.0
                y_center = annotation.bbox[1] + annotation.bbox[3] / 2.0
                bbox_width = annotation.bbox[2]
                bbox_height = annotation.bbox[3]
                x_center = x_center * dw
                y_center = y_center * dh
                bbox_width = bbox_width * dw
                bbox_height = bbox_height * dh
                category_id = annotation.category_id
                yolo_bbox = (x_center, y_center, bbox_width, bbox_height)
                # save yolo annotation
                with open(yolo_annotation_filepath, "w") as outfile:
                    outfile.write(
                        str(category_id)
                        + " "
                        + " ".join([str(value) for value in yolo_bbox])
                        + "\n"
                    )


def update_categories(desired_name2id: dict, coco_dict: dict) -> dict:
    """
    Rearranges category mapping of given COCO dictionary based on given category_mapping.
    Can also be used to filter some of the categories.

    Arguments:
    ---------
        desired_name2id : dict
            {"big_vehicle": 1, "car": 2, "human": 3}
        coco_dict : dict
            COCO formatted dictionary.
    Returns:
    ---------
        coco_target : dict
            COCO dict with updated/filtred categories.
    """
    # so that original variable doesnt get affected
    coco_source = copy.deepcopy(coco_dict)

    # init target coco dict
    coco_target = {"images": [], "annotations": [], "categories": []}

    # init vars
    currentid2desiredid_mapping = {}
    # create category id mapping (currentid2desiredid_mapping)
    for category in coco_source["categories"]:
        current_category_id = category["id"]
        current_category_name = category["name"]
        if current_category_name in desired_name2id.keys():
            currentid2desiredid_mapping[current_category_id] = desired_name2id[
                current_category_name
            ]
        else:
            # ignore categories that are not included in desired_name2id
            currentid2desiredid_mapping[current_category_id] = -1

    # update annotations
    for annotation in coco_source["annotations"]:
        current_category_id = annotation["category_id"]
        desired_category_id = currentid2desiredid_mapping[current_category_id]
        # append annotations with category id present in desired_name2id
        if desired_category_id != -1:
            # update cetegory id
            annotation["category_id"] = desired_category_id
            # append updated annotation to target coco dict
            coco_target["annotations"].append(annotation)

    # create desired categories
    categories = []
    for name in desired_name2id.keys():
        category = {}
        category["name"] = category["supercategory"] = name
        category["id"] = desired_name2id[name]
        categories.append(category)

    # update categories
    coco_target["categories"] = categories

    # update images
    coco_target["images"] = coco_source["images"]

    return coco_target


def update_categories_from_file(
    desired_name2id: dict, coco_path: str, save_path: str
) -> None:
    """
    Rearranges category mapping of a COCO dictionary in coco_path based on given category_mapping.
    Can also be used to filter some of the categories.
    Arguments:
    ---------
        desired_name2id : dict
            {"human": 1, "car": 2, "big_vehicle": 3}
        coco_path : str
            "dirname/coco.json"
    """
    # load source coco dict
    coco_source = load_json(coco_path)

    # update categories
    coco_target = update_categories(desired_name2id, coco_source)

    # save modified coco file
    save_json(coco_target, save_path)


def merge(coco_dict1: dict, coco_dict2: dict, desired_name2id: dict = None) -> dict:
    """
    Combines 2 coco formatted annotations dicts, and returns the combined coco dict.

    Arguments:
    ---------
        coco_dict1 : dict
            First coco dictionary.
        coco_dict2 : dict
            Second coco dictionary.
        desired_name2id : dict
            {"human": 1, "car": 2, "big_vehicle": 3}
    Returns:
    ---------
        merged_coco_dict : dict
            Merged COCO dict.
    """

    # copy input dicts so that original dicts are not affected
    temp_coco_dict1 = copy.deepcopy(coco_dict1)
    temp_coco_dict2 = copy.deepcopy(coco_dict2)

    # rearrange categories if any desired_name2id mapping is given
    if desired_name2id is not None:
        temp_coco_dict1 = update_categories(desired_name2id, temp_coco_dict1)
        temp_coco_dict2 = update_categories(desired_name2id, temp_coco_dict2)

    # rearrange categories of the second coco based on first, if their categories are not the same
    if temp_coco_dict1["categories"] != temp_coco_dict2["categories"]:
        desired_name2id = {
            category["name"]: category["id"]
            for category in temp_coco_dict1["categories"]
        }
        temp_coco_dict2 = update_categories(desired_name2id, temp_coco_dict2)

    # calculate first image and annotation index of the second coco file
    last_image_id = coco_dict1["images"][-1]["id"]
    last_annotation_id = coco_dict1["annotations"][-1]["id"]

    merged_coco_dict = temp_coco_dict1

    for image in temp_coco_dict2["images"]:
        image["id"] += last_image_id
        merged_coco_dict["images"].append(image)

    for annotation in temp_coco_dict2["annotations"]:
        annotation["image_id"] += last_image_id
        annotation["id"] += last_annotation_id
        merged_coco_dict["annotations"].append(annotation)

    return merged_coco_dict


def merge_from_list(coco_dict_list, desired_name2id=None, verbose=1):
    """
    Combines a list of coco formatted annotations dicts, and returns the combined coco dict.

    Arguments:
    ---------
        coco_dict)list : list of dict
            A list of coco dicts
        desired_name2id : dict
            {"human": 1, "car": 2, "big_vehicle": 3}
    Returns:
    ---------
        merged_coco_dict : dict
            Merged COCO dict.
    """
    if verbose:
        if not desired_name2id:
            print("'desired_name2id' is not specified, combining all categories.")

    # create desired_name2id by combinin all categories, if desired_name2id is not specified
    if desired_name2id is None:
        desired_name2id = {}
        for ind, coco_dict in enumerate(coco_dict_list):
            temp_categories = copy.deepcopy(coco_dict["categories"])
            for temp_category in temp_categories:
                if temp_category["name"] not in desired_name2id:
                    desired_name2id[temp_category["name"]] = ind + 1
                else:
                    continue

    for ind, coco_dict in enumerate(coco_dict_list):
        if ind == 0:
            merged_coco_dict = copy.deepcopy(coco_dict)
        else:
            merged_coco_dict = merge(merged_coco_dict, coco_dict, desired_name2id)

    # print categories
    if verbose:
        print(
            "Categories are formed as:\n",
            merged_coco_dict["categories"],
        )

    return merged_coco_dict


def merge_from_file(coco_path1: str, coco_path2: str, save_path: str):
    """
    Combines 2 coco formatted annotations files given their paths, and saves the combined file to save_path.

    Arguments:
    ---------
        coco_path1 : str
            Path for the first coco file.
        coco_path2 : str
            Path for the second coco file.
        save_path : str
            "dirname/coco.json"
    """

    # load coco files to be combined
    coco_dict1 = load_json(coco_path1)
    coco_dict2 = load_json(coco_path2)

    # merge coco dicts
    merged_coco_dict = merge(coco_dict1, coco_dict2)

    # save merged coco dict
    save_json(merged_coco_dict, save_path)


def get_imageid2annotationlist_mapping(coco_dict: dict) -> dict:
    """
    Slice a large image into smaller windows.

    Arguments
    ---------
        coco_dict : dict
            coco dict with fields "images", "annotations", "categories"
    Returns
    -------
        imageid2annotationlist_mapping : dict
        {
            1: [COCOAnnotation, COCOAnnotation, COCOAnnotation],
            2: [COCOAnnotation]
        }

        where
        COCOAnnotation = {
            'area': 2795520,
            'bbox': [491.0, 1035.0, 153.0, 182.0],
            'category_id': 1,
            'id': 1,
            'image_id': 1,
            'iscrowd': 0,
            'segmentation': [[491.0, 1035.0, 644.0, 1035.0, 644.0, 1217.0, 491.0, 1217.0]]
        }
    """
    imageid2annotationlist_mapping = {}
    for image in coco_dict["images"]:
        image_id = image["id"]
        imageid2annotationlist_mapping[image_id] = []

        for annotation in coco_dict["annotations"]:
            if annotation["image_id"] == image_id:
                imageid2annotationlist_mapping[image_id].append(annotation)

    return imageid2annotationlist_mapping


def create_coco_dict(images, categories, ignore_negative_samples=True):
    """
    Creates COCO dict with fields "images", "annotations", "categories".

    Arguments
    ---------
        images : List of CocoImage containing a list of CocoAnnotation
        categories : List of Dict
            COCO categories
        ignore_negative_samples : Bool
            If True, images without annotations are ignored
    Returns
    -------
        coco_dict : Dict
            COCO dict with fields "images", "annotations", "categories"
    """
    out_images = []
    out_annotations = []
    out_categories = categories

    num_images = len(images)
    image_id = 1
    annotation_id = 1
    for image_ind in range(num_images):
        # get coco image and its coco annotations
        coco_image = images[image_ind]
        coco_annotations = coco_image.annotations
        # get num annotations
        num_annotations = len(coco_annotations)
        # if ignore_negative_samples is True and no annotations, skip image
        if ignore_negative_samples and num_annotations == 0:
            continue
        else:
            # create coco image object
            out_image = {
                "height": coco_image.height,
                "width": coco_image.width,
                "id": image_id,
                "file_name": coco_image.file_name,
            }
            out_images.append(out_image)

            # do the same for image annotations
            for coco_annotation in coco_annotations:
                # create coco annotation object
                out_annotation = {
                    "iscrowd": 0,
                    "image_id": image_id,
                    "bbox": coco_annotation.bbox,
                    "segmentation": coco_annotation.segmentation,
                    "category_id": coco_annotation.category_id,
                    "id": annotation_id,
                    "area": coco_annotation.area,
                }
                out_annotations.append(out_annotation)
                annotation_id = annotation_id + 1

            # increment annotation id
            image_id = image_id + 1

    # form coco dict
    coco_dict = {
        "images": out_images,
        "annotations": out_annotations,
        "categories": out_categories,
    }
    # return coco dict
    return coco_dict


def split_coco_as_train_val(
    coco_file_path_or_dict,
    file_name=None,
    target_dir=None,
    train_split_rate=0.9,
    numpy_seed=0,
):
    """
    Takes single coco dataset file path, split images into train-val and saves as seperate coco dataset files.

    Args:
        coco_file_path_or_dict: str or dict
        file_name: str
        target_dir: str
        train_split_rate: float
        numpy_seed: int

    Returns:
        result : dict
            {
                "train_dict": "",
                "val_dict": "",
                "train_path": "",
                "val_path": "",
            }
    """
    # fix numpy numpy seed
    np.random.seed(numpy_seed)

    # set output coco file name
    if file_name:
        None
    elif isinstance(coco_file_path_or_dict, str):
        file_name = os.path.basename(coco_file_path_or_dict).replace(".json", "")
    elif target_dir:
        raise ValueError("file_name should be specified.")

    # read coco dict
    if isinstance(coco_file_path_or_dict, str):
        coco_dict = load_json(coco_file_path_or_dict)
    else:
        coco_dict = coco_file_path_or_dict

    # divide coco dict into train val coco dicts
    num_images = len(coco_dict["images"])
    random_indices = np.random.permutation(num_images).tolist()
    random_indices = [random_indice + 1 for random_indice in random_indices]
    image_ids = {a["image_id"] for a in coco_dict["annotations"]}
    image_id_2_idx = {i_id: i for i, i_id in enumerate(image_ids)}
    num_train = int(num_images * train_split_rate)

    # divide images
    train_indices = random_indices[:num_train]
    val_indices = random_indices[num_train:]
    train_images = np.array(coco_dict["images"])[
        (np.array(train_indices) - 1).tolist()
    ].tolist()
    val_images = np.array(coco_dict["images"])[
        (np.array(val_indices) - 1).tolist()
    ].tolist()
    # divide annotations
    train_annotations = list()
    val_annotations = list()
    for annotation in tqdm(coco_dict["annotations"]):
        image_index_for_annotation = image_id_2_idx[annotation["image_id"]]
        if image_index_for_annotation in train_indices:
            train_annotations.append(annotation)
        elif image_index_for_annotation in val_indices:
            val_annotations.append(annotation)
    # form train val coco dicts
    train_coco_dict = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_dict["categories"],
    }
    val_coco_dict = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_dict["categories"],
    }
    # return result
    if not target_dir:
        return {
            "train_dict": train_coco_dict,
            "val_dict": val_coco_dict,
            "train_path": "",
            "val_path": "",
        }
    else:
        train_coco_dict_path = os.path.join(target_dir, file_name + "_train.json")
        save_json(train_coco_dict, train_coco_dict_path)
        val_coco_dict_path = os.path.join(target_dir, file_name + "_val.json")
        save_json(val_coco_dict, val_coco_dict_path)
        return {
            "train_dict": train_coco_dict,
            "val_dict": val_coco_dict,
            "train_path": train_coco_dict_path,
            "val_path": val_coco_dict_path,
        }


def add_bbox_and_area_to_coco(
    source_coco_path: str = "",
    target_coco_path: str = "",
    add_bbox: bool = True,
    add_area: bool = True,
) -> dict:
    """
    Takes single coco dataset file path, calculates and fills bbox and area fields of the annotations
    and exports the updated coco dict.
    Returns:
    coco_dict : dict
        Updated coco dict
    """
    coco_dict = load_json(source_coco_path)
    coco_dict = copy.deepcopy(coco_dict)

    annotations = coco_dict["annotations"]
    for ind, annotation in enumerate(annotations):
        # assign annotation bbox
        if add_bbox:
            coco_polygons = []
            [
                coco_polygons.extend(coco_polygon)
                for coco_polygon in annotation["segmentation"]
            ]
            minx, miny, maxx, maxy = list(
                [
                    min(coco_polygons[0::2]),
                    min(coco_polygons[1::2]),
                    max(coco_polygons[0::2]),
                    max(coco_polygons[1::2]),
                ]
            )
            x, y, width, height = (
                int(minx),
                int(miny),
                int(maxx - minx),
                int(maxy - miny),
            )
            annotations[ind]["bbox"] = [x, y, width, height]

        # assign annotation area
        if add_area:
            shapely_multipolygon = get_shapely_multipolygon(
                coco_segmentation=annotation["segmentation"]
            )
            annotations[ind]["area"] = shapely_multipolygon.area

    coco_dict["annotations"] = annotations
    save_json(coco_dict, target_coco_path)
    return coco_dict


@dataclass
class DatasetClassCounts:
    """Stores the number of images that include each category in a dataset"""

    counts: dict
    total_images: int

    def frequencies(self):
        """calculates the frequenct of images that contain each category"""
        return {cid: count / self.total_images for cid, count in self.counts.items()}

    def __add__(self, o):
        total = self.total_images + o.total_images
        exclusive_keys = set(o.counts.keys()) - set(self.counts.keys())
        counts = {}
        for k, v in self.counts.items():
            counts[k] = v + o.counts.get(k, 0)
        for k in exclusive_keys:
            counts[k] = o.counts[k]
        return DatasetClassCounts(counts, total)


def count_images_with_category(coco_file_path):
    """Reads a coco dataset file and returns an DatasetClassCounts object
     that stores the number of images that include each category in a dataset
    Returns: DatasetClassCounts object
    coco_file_path : str
        path to coco dataset file
    """

    image_id_2_category_2_count = defaultdict(lambda: defaultdict(lambda: 0))
    coco = load_json(coco_file_path)
    for annotation in coco["annotations"]:
        image_id = annotation["image_id"]
        cid = annotation["category_id"]
        image_id_2_category_2_count[image_id][cid] = (
            image_id_2_category_2_count[image_id][cid] + 1
        )

    category_2_count = defaultdict(lambda: 0)
    for image_id, image_category_2_count in image_id_2_category_2_count.items():
        for cid, count in image_category_2_count.items():
            if count > 0:
                category_2_count[cid] = category_2_count[cid] + 1

    category_2_count = dict(category_2_count)
    total_images = len(image_id_2_category_2_count.keys())
    return DatasetClassCounts(category_2_count, total_images)


class CocoVid:
    def __init__(self, name=None, remapping_dict=None):
        """
        Creates CocoVid object.

        Args:
            name: str
                Name of the CocoVid dataset, it determines exported json name.
            remapping_dict: dict
                {1:0, 2:1} maps category id 1 to 0 and category id 2 to 1
        """
        self.name = name
        self.remapping_dict = remapping_dict
        self.categories = []
        self.videos = []

    def add_categories_from_coco_category_list(self, coco_category_list):
        """
        Creates CocoCategory object using coco category list.

        Args:
            coco_category_list: List[Dict]
                [
                    {"supercategory": "person", "id": 1, "name": "person"},
                    {"supercategory": "vehicle", "id": 2, "name": "bicycle"}
                ]
        """

        for coco_category in coco_category_list:
            if self.remapping_dict is not None:
                for source_id in self.remapping_dict.keys():
                    if coco_category["id"] == source_id:
                        target_id = self.remapping_dict[source_id]
                        coco_category["id"] = target_id

            self.add_category(CocoCategory.from_coco_category(coco_category))

    def add_category(self, category):
        """
        Adds category to this CocoVid instance

        Args:
            category: CocoCategory
        """

        assert (
            type(category) == CocoCategory
        ), "category must be a CocoCategory instance"

        self.categories.append(category)

    @property
    def json_categories(self):
        categories = []
        for category in self.categories:
            categories.append(category.json)
        return categories

    @property
    def category_mapping(self):
        category_mapping = {}
        for category in self.categories:
            category_mapping[category.id] = category.name
        return category_mapping

    def add_video(self, video):
        """
        Adds video to this CocoVid instance

        Args:
            video: CocoVideo
        """

        assert type(video) == CocoVideo, "video must be a CocoVideo instance"

        self.videos.append(video)

    @property
    def json(self):
        coco_dict = {
            "videos": [],
            "images": [],
            "annotations": [],
            "categories": self.json_categories,
        }
        annotation_id = 1
        image_id = 1
        video_id = 1
        global_instance_id = 1
        for coco_video in self.videos:
            coco_video.id = video_id
            coco_dict["videos"].append(coco_video.json)

            frame_id = 0
            instance_id_set = set()
            for cocovid_image in coco_video.images:
                cocovid_image.id = image_id
                cocovid_image.frame_id = frame_id
                cocovid_image.video_id = coco_video.id
                coco_dict["images"].append(cocovid_image.json)

                for cocovid_annotation in cocovid_image.annotations:
                    instance_id_set.add(cocovid_annotation.instance_id)
                    cocovid_annotation.instance_id += global_instance_id

                    cocovid_annotation.id = annotation_id
                    cocovid_annotation.image_id = cocovid_image.id
                    coco_dict["annotations"].append(cocovid_annotation.json)

                    # increment annotation_id
                    annotation_id = copy.deepcopy(annotation_id + 1)
                # increment image_id and frame_id
                image_id = copy.deepcopy(image_id + 1)
                frame_id = copy.deepcopy(frame_id + 1)
            # increment video_id and global_instance_id
            video_id = copy.deepcopy(video_id + 1)
            global_instance_id += len(instance_id_set)

        return coco_dict
