from __future__ import annotations

from typing import Any

import numpy as np

from sahi.slicing import CocoImage, SliceImageResult, SlicedImage, get_slice_bboxes
from sahi.utils.cv import read_image_as_pil


class BaseSlicer:
    """Abstract base class for image slicing backends.

    Subclasses only need to implement ``_slice_region`` to extract a crop
    from the image array.  All common logic (image reading, bbox
    calculation, result assembly) lives here.
    """

    def __init__(
        self,
        slice_height: int | None = 512,
        slice_width: int | None = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        auto_slice_resolution: bool = True,
    ) -> None:
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio
        self.auto_slice_resolution = auto_slice_resolution

    def slice_image(
        self,
        image: Any,
        verbose: bool = False,
        exif_fix: bool = True,
    ) -> SliceImageResult:
        """Slice an image into tiles.

        This is the main entry point.  It reads the image, computes
        slice bounding boxes, delegates pixel extraction to
        ``_slice_region``, and assembles the result.

        Args:
            image: Image path, PIL Image, or numpy array.
            verbose: Whether to print logs.
            exif_fix: Whether to apply EXIF orientation fix.

        Returns:
            SliceImageResult with all sliced tiles.
        """
        image_pil = read_image_as_pil(image, exif_fix=exif_fix)
        image_width, image_height = image_pil.size

        slice_bboxes = get_slice_bboxes(
            image_height=image_height,
            image_width=image_width,
            auto_slice_resolution=self.auto_slice_resolution,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )

        result = SliceImageResult(original_image_size=[image_height, image_width])
        image_arr = np.asarray(image_pil)

        for bbox in slice_bboxes:
            tlx, tly, brx, bry = bbox
            cropped = self._slice_region(image_arr, tly, bry, tlx, brx)
            coco_image = CocoImage(file_name="", height=bry - tly, width=brx - tlx)
            result.add_sliced_image(
                SlicedImage(image=cropped, coco_image=coco_image, starting_pixel=[tlx, tly])
            )

        return result

    def _slice_region(
        self, image_arr: np.ndarray, y_min: int, y_max: int, x_min: int, x_max: int
    ) -> np.ndarray:
        """Extract a rectangular region from the image array.

        Subclasses override this to provide accelerated implementations.

        Args:
            image_arr: Full image as HWC numpy array.
            y_min, y_max: Vertical slice bounds.
            x_min, x_max: Horizontal slice bounds.

        Returns:
            Cropped image region as numpy array.
        """
        return image_arr[y_min:y_max, x_min:x_max]
