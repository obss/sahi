# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import copy
from typing import Dict, List

import imantics
import numpy as np

from sahi.utils.coco import CocoAnnotation, CocoPrediction
from sahi.utils.cv import (
    get_bbox_from_bool_mask,
    get_bool_mask_from_coco_segmentation,
    get_coco_segmentation_from_bool_mask,
)
from sahi.utils.shapely import ShapelyAnnotation


class BoundingBox:
    """
    Bounding box of the annotation.
    """

    def __init__(self, box: list, shift_amount: list = [0, 0]):
        """
        Args:
            box: List
                [minx, miny, maxx, maxy]
            shift_amount: List
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
        """
        if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
            raise Exception("Box coords [minx, miny, maxx, maxy] cannot be negative")
        self.minx = int(box[0])
        self.miny = int(box[1])
        self.maxx = int(box[2])
        self.maxy = int(box[3])

        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]

    def get_expanded_box(self, ratio=0.1, max_x=None, max_y=None):
        w = self.maxx - self.minx
        h = self.maxy - self.miny
        y_mar = int(w * ratio)
        x_mar = int(h * ratio)
        maxx = min(max_x, self.maxx + x_mar) if max_x else self.maxx + x_mar
        minx = max(0, self.minx - x_mar)
        maxy = min(max_y, self.maxy + y_mar) if max_y else self.maxy + y_mar
        miny = max(0, self.miny - y_mar)
        box = [minx, miny, maxx, maxy]
        return BoundingBox(box)

    def to_coco_bbox(self):
        """
        Returns: [xmin, ymin, width, height]
        """
        return [self.minx, self.miny, self.maxx - self.minx, self.maxy - self.miny]

    def to_voc_bbox(self):
        """
        Returns: [xmin, ymin, xmax, ymax]
        """
        return [self.minx, self.miny, self.maxx, self.maxy]

    def get_shifted_box(self):
        """
        Returns: shifted BoundingBox
        """
        box = [
            self.minx + self.shift_x,
            self.miny + self.shift_y,
            self.maxx + self.shift_x,
            self.maxy + self.shift_y,
        ]
        return BoundingBox(box)

    def __repr__(self):
        return f"BoundingBox: <{(self.minx, self.miny, self.maxx, self.maxy)}, w: {self.maxx - self.minx}, h: {self.maxy - self.miny}>"


class Category:
    """
    Category of the annotation.
    """

    def __init__(self, id=None, name=None):
        """
        Args:
            id: int
                ID of the object category
            name: str
                Name of the object category
        """
        assert isinstance(id, int), "id should be integer"
        assert isinstance(name, str), "name should be string"
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Category: <id: {self.id}, name: {self.name}>"


class Mask:
    @classmethod
    def from_float_mask(
        cls,
        mask,
        full_image_size=None,
        mask_threshold: float = 0.5,
        shift_amount: list = [0, 0],
    ):
        """
        Args:
            mask: np.ndarray of np.float elements
                Mask values between 0 and 1 (should have a shape of height*width)
            mask_threshold: float
                Value to threshold mask pixels between 0 and 1
            shift_amount: List
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_image_size: List
                Size of the full image after shifting, should be in the form of [height, width]
        """
        bool_mask = mask > mask_threshold
        return cls(
            bool_mask=bool_mask,
            shift_amount=shift_amount,
            full_image_size=full_image_size,
        )

    @classmethod
    def from_coco_segmentation(
        cls,
        segmentation,
        full_image_size=None,
        shift_amount: list = [0, 0],
    ):
        """
        Init Mask from coco segmentation representation.

        Args:
            segmentation : List[List]
                [
                    [x1, y1, x2, y2, x3, y3, ...],
                    [x1, y1, x2, y2, x3, y3, ...],
                    ...
                ]
            full_image_size: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        # confirm full_image_size is given
        assert full_image_size is not None, "full_image_size must be provided"

        bool_mask = get_bool_mask_from_coco_segmentation(
            segmentation, height=full_image_size[0], width=full_image_size[1]
        )
        return cls(
            bool_mask=bool_mask,
            shift_amount=shift_amount,
            full_image_size=full_image_size,
        )

    def __init__(
        self,
        bool_mask=None,
        full_image_size=None,
        shift_amount: list = [0, 0],
    ):
        """
        Args:
            bool_mask: np.ndarray with np.bool elements
                2D mask of object, should have a shape of height*width
            full_image_size: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """

        if len(bool_mask) > 0:
            has_bool_mask = True
        else:
            has_bool_mask = False

        if has_bool_mask:
            self.bool_mask = bool_mask.astype(np.bool)
        else:
            self.bool_mask = None

        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]

        if full_image_size:
            self.full_image_height = full_image_size[0]
            self.full_image_width = full_image_size[1]
        elif has_bool_mask:
            self.full_image_height = self.bool_mask.shape[0]
            self.full_image_width = self.bool_mask.shape[1]
        else:
            self.full_image_height = None
            self.full_image_width = None

    def get_shifted_mask(self):
        # Confirm full_image_size is specified
        assert (self.full_image_height is not None) and (
            self.full_image_width is not None
        ), "full_image_size is None"
        # init full mask
        mask_fullsize = np.full(
            (
                self.full_image_height,
                self.full_image_width,
            ),
            0,
            dtype="float32",
        )

        # arrange starting ending indexes
        starting_pixel = [self.shift_x, self.shift_y]
        ending_pixel = [
            min(starting_pixel[0] + self.bool_mask.shape[1], self.full_image_width),
            min(starting_pixel[1] + self.bool_mask.shape[0], self.full_image_height),
        ]

        # convert sliced mask to full mask
        mask_fullsize[
            starting_pixel[1] : ending_pixel[1], starting_pixel[0] : ending_pixel[0]
        ] = self.bool_mask[
            : ending_pixel[1] - starting_pixel[1], : ending_pixel[0] - starting_pixel[0]
        ]

        return Mask(
            mask_fullsize,
            shift_amount=[0, 0],
            full_image_size=self.get_full_image_size(),
        )

    def get_full_image_size(self):
        """
        Returns image size as [height, width]
        """
        return [self.full_image_height, self.full_image_width]

    def to_coco_segmentation(self):
        """
        Returns boolean mask as coco segmentation:
        [
            [x1, y1, x2, y2, x3, y3, ...],
            [x1, y1, x2, y2, x3, y3, ...],
            ...
        ]
        """
        coco_segmentation = get_coco_segmentation_from_bool_mask(self.bool_mask)
        return coco_segmentation


class ObjectAnnotation:
    """
    All about an annotation such as Mask, Category, BoundingBox.
    """

    @classmethod
    def from_bool_mask(
        cls,
        bool_mask,
        category_id=None,
        category_name=None,
        shift_amount: list = [0, 0],
        full_image_size=None,
    ):
        """
        Creates ObjectAnnotation from bool_mask (2D np.ndarray)

        Args:
            bool_mask: np.ndarray with np.bool elements
                2D mask of object, should have a shape of height*width
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_image_size: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=[0, 0],
            full_image_size=full_image_size,
        )

    @classmethod
    def from_coco_segmentation(
        cls,
        segmentation,
        category_id=None,
        category_name=None,
        shift_amount: list = [0, 0],
        full_image_size=None,
    ):
        """
        Creates ObjectAnnotation from coco segmentation:
        [
            [x1, y1, x2, y2, x3, y3, ...],
            [x1, y1, x2, y2, x3, y3, ...],
            ...
        ]

        Args:
            segmentation: List[List]
                [
                    [x1, y1, x2, y2, x3, y3, ...],
                    [x1, y1, x2, y2, x3, y3, ...],
                    ...
                ]
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_image_size: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        bool_mask = get_bool_mask_from_coco_segmentation(segmentation)
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=[0, 0],
            full_image_size=full_image_size,
        )

    @classmethod
    def from_coco_bbox(
        cls,
        bbox,
        category_id=None,
        category_name=None,
        shift_amount: list = [0, 0],
        full_image_size=None,
    ):
        """
        Creates ObjectAnnotation from coco bbox [minx, miny, width, height]

        Args:
            bbox: List
                [minx, miny, width, height]
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_image_size: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        return cls(
            category_id=category_id,
            bbox=bbox,
            category_name=category_name,
            shift_amount=[0, 0],
            full_image_size=full_image_size,
        )

    @classmethod
    def from_shapely_annotation(
        cls,
        annotation,
        category_id=None,
        category_name=None,
        shift_amount: list = [0, 0],
        full_image_size=None,
    ):
        """
        Creates ObjectAnnotation from shapely_utils.ShapelyAnnotation

        Args:
            annotation: shapely_utils.ShapelyAnnotation
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_image_size: List
                Size of the full image, should be in the form of [height, width]
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        bool_mask = get_bool_mask_from_coco_segmentation(
            annotation.to_coco_segmentation()
        )
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=[0, 0],
            full_image_size=full_image_size,
        )

    @classmethod
    def from_imantics_annotation(
        cls,
        annotation,
        shift_amount: list = [0, 0],
        full_image_size=None,
    ):
        """
        Creates ObjectAnnotation from imantics.annotation.Annotation

        Args:
            annotation: imantics.annotation.Annotation
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
            full_image_size: List
                Size of the full image, should be in the form of [height, width]
        """
        return cls(
            category_id=annotation.category.id,
            bool_mask=annotation.mask.array,
            category_name=annotation.category.name,
            shift_amount=[0, 0],
            full_image_size=full_image_size,
        )

    def __init__(
        self,
        bbox=None,
        bool_mask=None,
        category_id=None,
        category_name=None,
        shift_amount: list = [0, 0],
        full_image_size=None,
    ):
        """
        Args:
            bbox: List
                [minx, miny, maxx, maxy]
            bool_mask: np.ndarray with np.bool elements
                2D mask of object, should have a shape of height*width
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            shift_amount: List
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_image_size: List
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        assert isinstance(category_id, int), "category_id must be an integer"
        assert (bbox is not None) or (
            bool_mask is not None
        ), "you must provide a bbox or bool_mask"

        if bool_mask is None:
            self.mask = None
            self.bbox = BoundingBox(bbox, shift_amount)
        else:
            self.mask = Mask(
                bool_mask=bool_mask,
                shift_amount=shift_amount,
                full_image_size=full_image_size,
            )
            bbox = get_bbox_from_bool_mask(bool_mask)
            self.bbox = BoundingBox(bbox, shift_amount)
        category_name = category_name if category_name else str(category_id)
        self.category = Category(
            id=category_id,
            name=category_name,
        )

        self.merged = None

    def to_coco_annotation(self):
        """
        Returns sahi.utils.coco.CocoAnnotation representation of ObjectAnnotation.
        """
        if self.mask:
            coco_annotation = CocoAnnotation.from_coco_segmentation(
                segmentation=self.mask.to_coco_segmentation(),
                category_id=self.category.id,
                category_name=self.category.name,
            )
        else:
            coco_annotation = CocoAnnotation.from_coco_bbox(
                bbox=self.bbox.to_coco_bbox(),
                category_id=self.category.id,
                category_name=self.category.name,
            )
        return coco_annotation

    def to_coco_prediction(self):
        """
        Returns sahi.utils.coco.CocoPrediction representation of ObjectAnnotation.
        """
        if self.mask:
            coco_prediction = CocoPrediction.from_coco_segmentation(
                segmentation=self.mask.to_coco_segmentation(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.score,
            )
        else:
            coco_prediction = CocoPrediction.from_coco_bbox(
                bbox=self.bbox.to_coco_bbox(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.score,
            )
        return coco_prediction

    def to_shapely_annotation(self):
        """
        Returns sahi.utils.shapely.ShapelyAnnotation representation of ObjectAnnotation.
        """
        if self.mask:
            shapely_annotation = ShapelyAnnotation.from_coco_segmentation(
                segmentation=self.mask.to_coco_segmentation(),
            )
        else:
            shapely_annotation = ShapelyAnnotation.from_coco_bbox(
                bbox=self.bbox.to_coco_bbox(),
            )
        return shapely_annotation

    def to_imantics_annotation(self):
        """
        Returns imantics.annotation.Annotation representation of ObjectAnnotation.
        """
        imantics_category = imantics.Category(
            id=self.category.id, name=self.category.name
        )
        imantics_mask = imantics.Mask.create(self.mask.bool_mask)
        imantics_annotation = imantics.annotation.Annotation.from_mask(
            mask=imantics_mask, category=imantics_category
        )
        return imantics_annotation

    def deepcopy(self):
        """
        Returns: deepcopy of current ObjectAnnotation instance
        """
        return copy.deepcopy(self)

    @classmethod
    def get_empty_mask(cls):
        return Mask(bool_mask=None)

    def get_shifted_object_annotation(self):
        if self.mask:
            return ObjectAnnotation(
                bbox=self.bbox.get_shifted_box().to_voc_bbox(),
                category_id=self.category.id,
                bool_mask=self.mask.get_shifted_mask().bool_mask,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_image_size=self.mask.get_shifted_mask().get_full_image_size(),
            )
        else:
            return ObjectAnnotation(
                bbox=self.bbox.get_shifted_box().to_voc_bbox(),
                category_id=self.category.id,
                bool_mask=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_image_size=None,
            )

    def __repr__(self):
        return f"""ObjectAnnotation<
    bbox: {self.bbox},
    mask: {self.mask},
    category: {self.category}>"""
