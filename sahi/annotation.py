import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from sahi.utils.coco import CocoAnnotation, CocoPrediction
from sahi.utils.cv import (
    get_bbox_from_coco_segmentation,
    get_bool_mask_from_coco_segmentation,
    get_coco_segmentation_from_bool_mask,
)
from sahi.utils.shapely import ShapelyAnnotation


@dataclass(frozen=True)
class BoundingBox:
    """
    BoundingBox represents a rectangular region in 2D space, typically used for object detection annotations.

    Attributes:
        box (Tuple[float, float, float, float]): The bounding box coordinates in the format (minx, miny, maxx, maxy).
            - minx (float): Minimum x-coordinate (left).
            - miny (float): Minimum y-coordinate (top).
            - maxx (float): Maximum x-coordinate (right).
            - maxy (float): Maximum y-coordinate (bottom).
        shift_amount (Tuple[int, int], optional): The amount to shift the bounding box in the x and y directions.
            Defaults to (0, 0).

    !!! example "BoundingBox Usage Example"
        ```python
        bbox = BoundingBox((10.0, 20.0, 50.0, 80.0))
        area = bbox.area
        expanded_bbox = bbox.get_expanded_box(ratio=0.2)
        shifted_bbox = bbox.get_shifted_box()
        coco_format = bbox.to_coco_bbox()
        ```

    """

    box: Union[Tuple[float, float, float, float], List[float]]
    shift_amount: Tuple[int, int] = (0, 0)

    def __post_init__(self):
        if len(self.box) != 4 or any(coord < 0 for coord in self.box):
            raise ValueError("box must be 4 non-negative floats: [minx, miny, maxx, maxy]")
        if len(self.shift_amount) != 2:
            raise ValueError("shift_amount must be 2 integers: [shift_x, shift_y]")

    @property
    def minx(self):
        return self.box[0]

    @property
    def miny(self):
        return self.box[1]

    @property
    def maxx(self):
        return self.box[2]

    @property
    def maxy(self):
        return self.box[3]

    @property
    def shift_x(self):
        return self.shift_amount[0]

    @property
    def shift_y(self):
        return self.shift_amount[1]

    @property
    def area(self):
        return (self.maxx - self.minx) * (self.maxy - self.miny)

    def get_expanded_box(self, ratio: float = 0.1, max_x: int = None, max_y: int = None):
        """
        Returns an expanded bounding box by increasing its size by a given ratio.
        The expansion is applied equally in all directions. Optionally, the expanded box
        can be clipped to maximum x and y boundaries.

        Args:
            ratio (float, optional): The proportion by which to expand the box size.
                Default is 0.1 (10%).
            max_x (int, optional): The maximum allowed x-coordinate for the expanded box.
                If None, no maximum is applied.
            max_y (int, optional): The maximum allowed y-coordinate for the expanded box.
                If None, no maximum is applied.

        Returns:
            BoundingBox: A new BoundingBox instance representing the expanded box.
        """

        w = self.maxx - self.minx
        h = self.maxy - self.miny
        y_mar = int(w * ratio)
        x_mar = int(h * ratio)
        maxx = min(max_x, self.maxx + x_mar) if max_x else self.maxx + x_mar
        minx = max(0, self.minx - x_mar)
        maxy = min(max_y, self.maxy + y_mar) if max_y else self.maxy + y_mar
        miny = max(0, self.miny - y_mar)
        box: list[float] = [minx, miny, maxx, maxy]
        return BoundingBox(box)

    def to_xywh(self):
        """
        Returns [xmin, ymin, width, height]

        Returns:
            List[float]: A list containing the bounding box in the format [xmin, ymin, width, height].
        """

        return [self.minx, self.miny, self.maxx - self.minx, self.maxy - self.miny]

    def to_coco_bbox(self):
        """
        Returns the bounding box in COCO format: [xmin, ymin, width, height]

        Returns:
            List[float]: A list containing the bounding box in COCO format.
        """
        return self.to_xywh()

    def to_xyxy(self):
        """
        Returns: [xmin, ymin, xmax, ymax]

        Returns:
            List[float]: A list containing the bounding box in the format [xmin, ymin, xmax, ymax].
        """
        return [self.minx, self.miny, self.maxx, self.maxy]

    def to_voc_bbox(self):
        """
        Returns the bounding box in VOC format: [xmin, ymin, xmax, ymax]

        Returns:
            List[float]: A list containing the bounding box in VOC format.
        """
        return self.to_xyxy()

    def get_shifted_box(self):
        """
        Returns shifted BoundingBox

        Returns:
            BoundingBox: A new BoundingBox instance representing the shifted box.
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


@dataclass(frozen=True)
class Category:
    """
    Category of the annotation.

    Attributes:
        id (int): Unique identifier for the category.
        name (str): Name of the category.

    """

    id: int
    name: str

    def __post_init__(self):
        if not isinstance(self.id, int):
            raise TypeError("id should be integer")
        if not isinstance(self.name, str):
            raise TypeError("name should be string")

    def __repr__(self):
        return f"Category: <id: {self.id}, name: {self.name}>"


class Mask:
    """
    Init Mask from coco segmentation representation.

    Args:
        segmentation : List[List]
            [
                [x1, y1, x2, y2, x3, y3, ...],
                [x1, y1, x2, y2, x3, y3, ...],
                ...
            ]
        full_shape: List[int]
            Size of the full image, should be in the form of [height, width]
        shift_amount: List[int]
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y]
    """

    def __init__(
        self,
        segmentation: List[List[float]],
        full_shape: List[int],
        shift_amount: list = [0, 0],
    ):
        if full_shape is None:
            raise ValueError("full_shape must be provided")  # pyright: ignore[reportUnreachable]

        self.shift_x = shift_amount[0]
        self.shift_y = shift_amount[1]
        self.full_shape_height = full_shape[0]
        self.full_shape_width = full_shape[1]
        self.segmentation = segmentation

    @classmethod
    def from_float_mask(
        cls,
        mask: np.ndarray,
        full_shape: List[int],
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
            full_shape: List[int]
                Size of the full image after shifting, should be in the form of [height, width]
        """
        bool_mask = mask > mask_threshold
        return cls(
            segmentation=get_coco_segmentation_from_bool_mask(bool_mask),
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_bool_mask(
        cls,
        bool_mask: np.ndarray,
        full_shape: List[int],
        shift_amount: list = [0, 0],
    ):
        """
        Args:
            bool_mask: np.ndarray with bool elements
                2D mask of object, should have a shape of height*width
            full_shape: List[int]
                Size of the full image, should be in the form of [height, width]
            shift_amount: List[int]
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        return cls(
            segmentation=get_coco_segmentation_from_bool_mask(bool_mask),
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @property
    def bool_mask(self) -> np.ndarray:
        return get_bool_mask_from_coco_segmentation(
            self.segmentation, width=self.full_shape[1], height=self.full_shape[0]
        )

    @property
    def shape(self) -> List[int]:
        """
        Returns mask shape as [height, width]
        """
        return [self.bool_mask.shape[0], self.bool_mask.shape[1]]

    @property
    def full_shape(self) -> List[int]:
        """
        Returns full mask shape after shifting as [height, width]
        """
        return [self.full_shape_height, self.full_shape_width]

    @property
    def shift_amount(self):
        """
        Returns the shift amount of the mask slice as [shift_x, shift_y]
        """
        return [self.shift_x, self.shift_y]

    def get_shifted_mask(self) -> "Mask":
        # Confirm full_shape is specified
        if (self.full_shape_height is None) or (self.full_shape_width is None):
            raise ValueError("full_shape is None")
        shifted_segmentation = []
        for s in self.segmentation:
            xs = [min(self.shift_x + s[i], self.full_shape_width) for i in range(0, len(s) - 1, 2)]
            ys = [min(self.shift_y + s[i], self.full_shape_height) for i in range(1, len(s), 2)]
            shifted_segmentation.append([j for i in zip(xs, ys) for j in i])
        return Mask(
            segmentation=shifted_segmentation,
            shift_amount=[0, 0],
            full_shape=self.full_shape,
        )


class ObjectAnnotation:
    """
    All about an annotation such as Mask, Category, BoundingBox.
    """

    def __init__(
        self,
        bbox: Optional[List[int]] = None,
        segmentation: Optional[np.ndarray] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Args:
            bbox: List
                [minx, miny, maxx, maxy]
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
            shift_amount: List
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        if not isinstance(category_id, int):
            raise ValueError("category_id must be an integer")
        if (bbox is None) and (segmentation is None):
            raise ValueError("you must provide a bbox or segmentation")

        self.mask: Mask | None = None
        if segmentation is not None:
            self.mask = Mask(
                segmentation=segmentation,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
            bbox_from_segmentation = get_bbox_from_coco_segmentation(segmentation)
            # https://github.com/obss/sahi/issues/235
            if bbox_from_segmentation is not None:
                bbox = bbox_from_segmentation
            else:
                raise ValueError("Invalid segmentation mask.")

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
        self.bbox = BoundingBox(bbox, shift_amount)

        category_name = category_name if category_name else str(category_id)
        self.category = Category(
            id=category_id,
            name=category_name,
        )

        self.merged = None

    @classmethod
    def from_bool_mask(
        cls,
        bool_mask,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
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
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
        return cls(
            category_id=category_id,
            segmentation=segmentation,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_segmentation(
        cls,
        segmentation,
        full_shape: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
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
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        return cls(
            category_id=category_id,
            segmentation=segmentation,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_bbox(
        cls,
        bbox: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
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
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_coco_annotation_dict(
        cls,
        annotation_dict: Dict,
        full_shape: List[int],
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
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
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        if annotation_dict["segmentation"]:
            return cls.from_coco_segmentation(
                segmentation=annotation_dict["segmentation"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )
        else:
            return cls.from_coco_bbox(
                bbox=annotation_dict["bbox"],
                category_id=annotation_dict["category_id"],
                category_name=category_name,
                shift_amount=shift_amount,
                full_shape=full_shape,
            )

    @classmethod
    def from_shapely_annotation(
        cls,
        annotation: ShapelyAnnotation,
        full_shape: List[int],
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        shift_amount: Optional[List[int]] = [0, 0],
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
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
        """
        return cls(
            category_id=category_id,
            segmentation=annotation.to_coco_segmentation(),
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    @classmethod
    def from_imantics_annotation(
        cls,
        annotation,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectAnnotation from imantics.annotation.Annotation

        Args:
            annotation: imantics.annotation.Annotation
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image, should be in the form of [height, width]
        """
        return cls(
            category_id=annotation.category.id,
            bool_mask=annotation.mask.array,
            category_name=annotation.category.name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def to_coco_annotation(self) -> CocoAnnotation:
        """
        Returns sahi.utils.coco.CocoAnnotation representation of ObjectAnnotation.
        """
        if self.mask:
            coco_annotation = CocoAnnotation.from_coco_segmentation(
                segmentation=self.mask.segmentation,
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

    def to_coco_prediction(self) -> CocoPrediction:
        """
        Returns sahi.utils.coco.CocoPrediction representation of ObjectAnnotation.
        """
        if self.mask:
            coco_prediction = CocoPrediction.from_coco_segmentation(
                segmentation=self.mask.segmentation,
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

    def to_shapely_annotation(self) -> ShapelyAnnotation:
        """
        Returns sahi.utils.shapely.ShapelyAnnotation representation of ObjectAnnotation.
        """
        if self.mask:
            shapely_annotation = ShapelyAnnotation.from_coco_segmentation(
                segmentation=self.mask.segmentation,
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
            raise ImportError('Please run "pip install -U imantics" to install imantics first for imantics conversion.')

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

    def get_shifted_object_annotation(self):
        if self.mask:
            shifted_mask = self.mask.get_shifted_mask()
            return ObjectAnnotation(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                segmentation=shifted_mask.segmentation,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=shifted_mask.full_shape,
            )
        else:
            return ObjectAnnotation(
                bbox=self.bbox.get_shifted_box().to_xyxy(),
                category_id=self.category.id,
                bool_mask=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def __repr__(self):
        return f"""ObjectAnnotation<
    bbox: {self.bbox},
    mask: {self.mask},
    category: {self.category}>"""
