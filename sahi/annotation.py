# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import copy
from typing import Dict, List, Optional

import numpy as np

from sahi.utils.coco import CocoAnnotation, CocoPrediction
from sahi.utils.cv import (
    get_bbox_from_bool_mask,
    get_bool_mask_from_coco_segmentation,
    get_coco_segmentation_from_bool_mask,
)
from sahi.utils.shapely import ShapelyAnnotation

try:
    from pycocotools import mask as mask_utils

    use_rle = True
except ImportError:
    use_rle = False


class BoundingBox:
    """
    Bounding box of the annotation.
    """

    def __init__(self, box: List[float], offset_amount: List[int] = [0, 0]):
        """
        Args:
            box: List[float]
                [minx, miny, maxx, maxy]
            offset_amount: List[int]
                To remap the box and mask predictions from sliced image
                to full sized image, should be in the form of [offset_x, offset_y]
        """
        if box[0] < 0 or box[1] < 0 or box[2] < 0 or box[3] < 0:
            raise Exception("Box coords [minx, miny, maxx, maxy] cannot be negative")
        self.minx = box[0]
        self.miny = box[1]
        self.maxx = box[2]
        self.maxy = box[3]

        self.offset_x = offset_amount[0]
        self.offset_y = offset_amount[1]

    @property
    def offset_amount(self):
        """
        Returns the offset amount of the bbox slice as [offset_x, offset_y]
        """
        return [self.offset_x, self.offset_y]

    @property
    def area(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny)

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

    def to_xywh(self):
        """
        Returns: [xmin, ymin, width, height]
        """
        return [self.minx, self.miny, self.maxx - self.minx, self.maxy - self.miny]

    def to_coco_bbox(self):
        """
        Returns: [xmin, ymin, width, height]
        """
        return self.to_xywh()

    def to_xyxy(self):
        """
        Returns: [xmin, ymin, xmax, ymax]
        """
        return [self.minx, self.miny, self.maxx, self.maxy]

    def to_voc_bbox(self):
        """
        Returns: [xmin, ymin, xmax, ymax]
        """
        return self.to_xyxy()

    def remap(self, inplace: bool = False):
        """
        Remaps bbox coords using offset_amount

        Args:
            inplace: bool
                If True, remaps the bbox coords inplace, else returns a new BoundingBox object

        Returns:
            BoundingBox: remapped bbox
        """
        minx = self.minx + self.offset_x
        miny = self.miny + self.offset_y
        maxx = self.maxx + self.offset_x
        maxy = self.maxy + self.offset_y

        if inplace:
            self.minx = minx
            self.miny = miny
            self.maxx = maxx
            self.maxy = maxy

            self.offset_x = 0
            self.offset_y = 0

            return self
        else:
            box = [minx, miny, maxx, maxy]
            return BoundingBox(box, offset_amount=[0, 0])

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
        if not isinstance(id, int):
            raise TypeError("id should be integer")
        if not isinstance(name, str):
            raise TypeError("name should be string")
        self.id = id
        self.name = name

    def __repr__(self):
        return f"Category: <id: {self.id}, name: {self.name}>"


class Mask:
    @classmethod
    def from_float_mask(
        cls,
        mask,
        full_shape=None,
        mask_threshold: float = 0.5,
        offset_amount: list = [0, 0],
    ):
        """
        Args:
            mask: np.ndarray of np.float elements
                Mask values between 0 and 1 (should have a shape of height*width)
            mask_threshold: float
                Value to threshold mask pixels between 0 and 1
            offset_amount: List
                To remap the predicted mask from sliced image to full
                sized image, should be in the form of [offset_x, offset_y]
            full_shape: List
                Size of the full image after remapping, should be in the form of [height, width]
        """
        bool_mask = mask > mask_threshold
        return cls(
            bool_mask=bool_mask,
            offset_amount=offset_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_segmentation(
        cls,
        segmentation,
        full_shape=None,
        offset_amount: list = [0, 0],
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
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            offset_amount: List
                To remap the predicted mask from sliced image to full
                sized image, should be in the form of [offset_x, offset_y]
        """
        # confirm full_shape is given
        if full_shape is None:
            raise ValueError("full_shape must be provided")
        bool_mask = get_bool_mask_from_coco_segmentation(segmentation, height=full_shape[0], width=full_shape[1])
        return cls(
            bool_mask=bool_mask,
            offset_amount=offset_amount,
            full_shape=full_shape,
        )

    def __init__(
        self,
        bool_mask=None,
        full_shape=None,
        offset_amount: list = [0, 0],
    ):
        """
        Args:
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            offset_amount: List
                To remap the predicted mask from sliced image to full
                sized image, should be in the form of [offset_x, offset_y]
        """

        if len(bool_mask) > 0:
            has_bool_mask = True
        else:
            has_bool_mask = False

        if has_bool_mask:
            self._mask = self.encode_bool_mask(bool_mask)
        else:
            self._mask = None

        self.offset_x = offset_amount[0]
        self.offset_y = offset_amount[1]

        if full_shape:
            self.full_shape_height = full_shape[0]
            self.full_shape_width = full_shape[1]
        elif has_bool_mask:
            self.full_shape_height = self.bool_mask.shape[0]
            self.full_shape_width = self.bool_mask.shape[1]
        else:
            self.full_shape_height = None
            self.full_shape_width = None

    def encode_bool_mask(self, bool_mask):
        _mask = bool_mask
        if use_rle:
            _mask = mask_utils.encode(np.asfortranarray(bool_mask.astype(np.uint8)))
        return _mask

    def decode_bool_mask(self, bool_mask):
        _mask = bool_mask
        if use_rle:
            _mask = mask_utils.decode(bool_mask).astype(bool)
        return _mask

    @property
    def bool_mask(self):
        return self.decode_bool_mask(self._mask)

    @property
    def shape(self):
        """
        Returns mask shape as [height, width]
        """
        return [self.bool_mask.shape[0], self.bool_mask.shape[1]]

    @property
    def full_shape(self):
        """
        Returns full mask shape after remapping as [height, width]
        """
        return [self.full_shape_height, self.full_shape_width]

    @property
    def offset_amount(self):
        """
        Returns the offset amount of the mask slice as [offset_x, offset_y]
        """
        return [self.offset_x, self.offset_y]

    def remap(self, inplace: bool = False):
        """
        Remap the mask to full sized image.

        Args:
            inplace: bool
                If True, remap the mask in place, otherwise return a new Mask object

        Returns:
            Mask: remapped mask
        """
        # Confirm full_shape is specified
        if (self.full_shape_height is None) or (self.full_shape_width is None):
            raise ValueError("full_shape is None")
        # init full mask
        mask_fullsized = np.full(
            (
                self.full_shape_height,
                self.full_shape_width,
            ),
            0,
            dtype="float32",
        )

        # arrange starting ending indexes
        starting_pixel = [self.offset_x, self.offset_y]
        ending_pixel = [
            min(starting_pixel[0] + self.bool_mask.shape[1], self.full_shape_width),
            min(starting_pixel[1] + self.bool_mask.shape[0], self.full_shape_height),
        ]

        # convert sliced mask to full mask
        mask_fullsized[starting_pixel[1] : ending_pixel[1], starting_pixel[0] : ending_pixel[0]] = self.bool_mask[
            : ending_pixel[1] - starting_pixel[1], : ending_pixel[0] - starting_pixel[0]
        ]

        if inplace:
            self.bool_mask = mask_fullsized
            self.offset_x = 0
            self.offset_y = 0

            return self
        else:
            return Mask(
                bool_mask=mask_fullsized,
                offset_amount=[0, 0],
                full_shape=self.full_shape,
            )

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
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        offset_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from bool_mask (2D np.ndarray)

        Args:
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            offset_amount: List
                To remap the box and mask predictions from sliced image to full
                sized image, should be in the form of [offset_x, offset_y]
        """
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            offset_amount=offset_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_segmentation(
        cls,
        segmentation,
        full_shape: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        offset_amount: Optional[List[int]] = [0, 0],
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
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            offset_amount: List
                To remap the box and mask predictions from sliced image to full
                sized image, should be in the form of [offset_x, offset_y]
        """
        bool_mask = get_bool_mask_from_coco_segmentation(segmentation, width=full_shape[1], height=full_shape[0])
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            offset_amount=offset_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_bbox(
        cls,
        bbox: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        offset_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
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
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            offset_amount: List
                To remap the box and mask predictions from sliced image to full
                sized image, should be in the form of [offset_x, offset_y]
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
            offset_amount=offset_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_annotation_dict(
        cls,
        annotation_dict: Dict,
        full_shape: List[int],
        category_name: str = None,
        offset_amount: Optional[List[int]] = [0, 0],
    ):
        """
        Creates ObjectAnnotation object from category name and COCO formatted
        annotation dict (with fields "bbox", "segmentation", "category_id").

        Args:
            annotation_dict: dict
                COCO formatted annotation dict (with fields "bbox", "segmentation", "category_id")
            category_name: str
                Category name of the annotation
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            offset_amount: List
                To remap the box and mask predictions from sliced image to full
                sized image, should be in the form of [soffset_x, offset_y]
        """
        if annotation_dict["segmentation"]:
            return cls.from_coco_segmentation(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                offset_amount=offset_amount,
                full_shape=full_shape,
            )
        else:
            return cls.from_coco_bbox(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                offset_amount=offset_amount,
                full_shape=full_shape,
            )

    @classmethod
    def from_shapely_annotation(
        cls,
        annotation,
        full_shape: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        offset_amount: Optional[List[int]] = [0, 0],
    ):
        """
        Creates ObjectAnnotation from shapely_utils.ShapelyAnnotation

        Args:
            annotation: shapely_utils.ShapelyAnnotation
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            offset_amount: List
                To remap the box and mask predictions from sliced image to full
                sized image, should be in the form of [offset_x, offset_y]
        """
        bool_mask = get_bool_mask_from_coco_segmentation(
            annotation.to_coco_segmentation(), width=full_shape[1], height=full_shape[0]
        )
        return cls(
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            offset_amount=offset_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_imantics_annotation(
        cls,
        annotation,
        offset_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from imantics.annotation.Annotation

        Args:
            annotation: imantics.annotation.Annotation
            offset_amount: List
                To remap the box and mask predictions from sliced image to full
                sized image, should be in the form of [offset_x, offset_y]
            full_shape: List
                Size of the full image, should be in the form of [height, width]
        """
        return cls(
            category_id=annotation.category.id,
            bool_mask=annotation.mask.array,
            category_name=annotation.category.name,
            offset_amount=offset_amount,
            full_shape=full_shape,
        )

    def __init__(
        self,
        bbox: Optional[List[int]] = None,
        bool_mask: Optional[np.ndarray] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        offset_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Args:
            bbox: List
                [minx, miny, maxx, maxy]
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            offset_amount: List
                To remap the box and mask predictions from sliced image
                to full sized image, should be in the form of [offset_x, offset_y]
            full_shape: List
                Size of the full image after remapping, should be in
                the form of [height, width]
        """
        if not isinstance(category_id, int):
            raise ValueError("category_id must be an integer")
        if (bbox is None) and (bool_mask is None):
            raise ValueError("you must provide a bbox or bool_mask")

        if bool_mask is not None:
            self.mask = Mask(
                bool_mask=bool_mask,
                offset_amount=offset_amount,
                full_shape=full_shape,
            )
            bbox_from_bool_mask = get_bbox_from_bool_mask(bool_mask)
            # https://github.com/obss/sahi/issues/235
            if bbox_from_bool_mask is not None:
                bbox = bbox_from_bool_mask
            else:
                raise ValueError("Invalid boolean mask.")
        else:
            self.mask = None

        # if bbox is a numpy object, convert it to python List[float]
        if type(bbox).__module__ == "numpy":
            bbox = copy.deepcopy(bbox).tolist()

        # make sure bbox coords lie inside [0, image_size]
        xmin = max(bbox[0], 0)
        ymin = max(bbox[1], 0)
        if full_shape:
            xmax = min(bbox[2], full_shape[1])
            ymax = min(bbox[3], full_shape[0])
        else:
            xmax = bbox[2]
            ymax = bbox[3]
        bbox = [xmin, ymin, xmax, ymax]
        # set bbox
        self.bbox = BoundingBox(bbox, offset_amount)

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
                bbox=self.bbox.to_xywh(),
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
                score=1,
            )
        else:
            coco_prediction = CocoPrediction.from_coco_bbox(
                bbox=self.bbox.to_xywh(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=1,
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
                bbox=self.bbox.to_xywh(),
            )
        return shapely_annotation

    def to_imantics_annotation(self):
        """
        Returns imantics.annotation.Annotation representation of ObjectAnnotation.
        """
        try:
            import imantics
        except ImportError:
            raise ImportError(
                'Please run "pip install -U imantics" ' "to install imantics first for imantics conversion."
            )

        imantics_category = imantics.Category(id=self.category.id, name=self.category.name)
        if self.mask is not None:
            imantics_mask = imantics.Mask.create(self.mask.bool_mask)
            imantics_annotation = imantics.annotation.Annotation.from_mask(
                mask=imantics_mask, category=imantics_category
            )
        else:
            imantics_bbox = imantics.BBox.create(self.bbox.to_xyxy())
            imantics_annotation = imantics.annotation.Annotation.from_bbox(
                bbox=imantics_bbox, category=imantics_category
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

    def remap(self, inplace: bool = False):
        """
        Remaps the box and mask annotations from sliced image to full sized image.

        Args:
            inplace: bool
                If True, remaps the current instance, otherwise returns a new instance.

        Returns:
            ObjectAnnotation: remapped ObjectAnnotation
        """

        if inplace:
            self.bbox.remap(inplace=True)

            if self.mask:
                self.mask.remap(inplace=True)

            return self
        else:
            new_instance = self.deepcopy()
            new_instance.bbox = self.bbox.remap(inplace=False)
            if self.mask:
                new_instance.mask = self.mask.remap(inplace=False)
            return new_instance

    def __repr__(self):
        return f"""ObjectAnnotation<
    bbox: {self.bbox},
    mask: {self.mask},
    category: {self.category}>"""
