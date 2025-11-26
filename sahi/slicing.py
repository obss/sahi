from __future__ import annotations

import concurrent.futures
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from shapely.errors import TopologicalError
from tqdm import tqdm

from sahi.annotation import BoundingBox, Mask
from sahi.logger import logger
from sahi.utils.coco import Coco, CocoAnnotation, CocoImage, create_coco_dict
from sahi.utils.cv import IMAGE_EXTENSIONS_LOSSLESS, IMAGE_EXTENSIONS_LOSSY, read_image_as_pil
from sahi.utils.file import load_json, save_json

_CPU_COUNT = os.cpu_count() or 4
MAX_WORKERS = max(1, min(32, _CPU_COUNT * 2))


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int | None = None,
    slice_width: int | None = None,
    auto_slice_resolution: bool | None = True,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
) -> list[list[int]]:
    """Generate bounding boxes for slicing an image into crops.

    The function calculates the coordinates for each slice based on the provided
    image dimensions, slice size, and overlap ratios. If slice size is not provided
    and auto_slice_resolution is True, the function will automatically determine
    appropriate slice parameters.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these parameters from image resolution and orientation.

    Returns:
        List[List[int]]: List of 4 corner coordinates for each N slices.
            [
                [slice_0_left, slice_0_top, slice_0_right, slice_0_bottom],
                ...
                [slice_N_left, slice_N_top, slice_N_right, slice_N_bottom]
            ]
    """
    slice_bboxes = []
    y_max = y_min = 0

    if slice_height and slice_width:
        if overlap_height_ratio is not None and overlap_height_ratio >= 1.0:
            raise ValueError("Overlap ratio must be less than 1.0")
        if overlap_width_ratio is not None and overlap_width_ratio >= 1.0:
            raise ValueError("Overlap ratio must be less than 1.0")
        y_overlap = int((overlap_height_ratio if overlap_height_ratio is not None else 0.2) * slice_height)
        x_overlap = int((overlap_width_ratio if overlap_width_ratio is not None else 0.2) * slice_width)
    elif auto_slice_resolution:
        x_overlap, y_overlap, slice_width, slice_height = get_auto_slice_params(height=image_height, width=image_width)
    else:
        raise ValueError("Compute type is not auto and slice width and height are not provided.")

    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def annotation_inside_slice(annotation: dict, slice_bbox: list[int]) -> bool:
    """Check whether annotation coordinates lie inside slice coordinates.

    Args:
        annotation (dict): Single annotation entry in COCO format.
        slice_bbox (List[int]): Generated from `get_slice_bboxes`.
            Format for each slice bbox: [x_min, y_min, x_max, y_max].

    Returns:
        (bool): True if any annotation coordinate lies inside slice.
    """
    left, top, width, height = annotation["bbox"]

    right = left + width
    bottom = top + height

    if left >= slice_bbox[2]:
        return False
    if top >= slice_bbox[3]:
        return False
    if right <= slice_bbox[0]:
        return False
    if bottom <= slice_bbox[1]:
        return False

    return True


def process_coco_annotations(
    coco_annotation_list: list[CocoAnnotation], slice_bbox: list[int], min_area_ratio
) -> list[CocoAnnotation]:
    """Slices and filters given list of CocoAnnotation objects with given 'slice_bbox' and 'min_area_ratio'.

    Args:
        coco_annotation_list (List[CocoAnnotation])
        slice_bbox (List[int]): Generated from `get_slice_bboxes`.
            Format for each slice bbox: [x_min, y_min, x_max, y_max].
        min_area_ratio (float): If the cropped annotation area to original
            annotation ratio is smaller than this value, the annotation is
            filtered out. Default 0.1.

    Returns:
        (List[CocoAnnotation]): Sliced annotations.
    """

    sliced_coco_annotation_list: list[CocoAnnotation] = []
    for coco_annotation in coco_annotation_list:
        if annotation_inside_slice(coco_annotation.json, slice_bbox):
            sliced_coco_annotation = coco_annotation.get_sliced_coco_annotation(slice_bbox)
            if sliced_coco_annotation.area / coco_annotation.area >= min_area_ratio:
                sliced_coco_annotation_list.append(sliced_coco_annotation)
    return sliced_coco_annotation_list


class SlicedImage:
    def __init__(self, image, coco_image, starting_pixel):
        """
        image: np.array
            Sliced image.
        coco_image: CocoImage
            Coco styled image object that belong to sliced image.
        starting_pixel: list of list of int
            Starting pixel coordinates of the sliced image.
        """
        self.image = image
        self.coco_image = coco_image
        self.starting_pixel = starting_pixel


class SliceImageResult:
    def __init__(self, original_image_size: list[int], image_dir: str | None = None):
        """
        image_dir: str
            Directory of the sliced image exports.
        original_image_size: list of int
            Size of the unsliced original image in [height, width]
        """
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]
        self.image_dir = image_dir

        self._sliced_image_list: list[SlicedImage] = []

    def add_sliced_image(self, sliced_image: SlicedImage):
        if not isinstance(sliced_image, SlicedImage):
            raise TypeError("sliced_image must be a SlicedImage instance")

        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self):
        return self._sliced_image_list

    @property
    def images(self):
        """Returns sliced images.

        Returns:
            images: a list of np.array
        """
        images = []
        for sliced_image in self._sliced_image_list:
            images.append(sliced_image.image)
        return images

    @property
    def coco_images(self) -> list[CocoImage]:
        """Returns CocoImage representation of SliceImageResult.

        Returns:
            coco_images: a list of CocoImage
        """
        coco_images: list = []
        for sliced_image in self._sliced_image_list:
            coco_images.append(sliced_image.coco_image)
        return coco_images

    @property
    def starting_pixels(self) -> list[int]:
        """Returns a list of starting pixels for each slice.

        Returns:
            starting_pixels: a list of starting pixel coords [x,y]
        """
        starting_pixels = []
        for sliced_image in self._sliced_image_list:
            starting_pixels.append(sliced_image.starting_pixel)
        return starting_pixels

    @property
    def filenames(self) -> list[int]:
        """Returns a list of filenames for each slice.

        Returns:
            filenames: a list of filenames as str
        """
        filenames = []
        for sliced_image in self._sliced_image_list:
            filenames.append(sliced_image.coco_image.file_name)
        return filenames

    def __getitem__(self, i):
        def _prepare_ith_dict(i):
            return {
                "image": self.images[i],
                "coco_image": self.coco_images[i],
                "starting_pixel": self.starting_pixels[i],
                "filename": self.filenames[i],
            }

        if isinstance(i, np.ndarray):
            i = i.tolist()

        if isinstance(i, int):
            return _prepare_ith_dict(i)
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return [_prepare_ith_dict(i) for i in range(start, stop, step)]
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(_prepare_ith_dict, i)
            return list(accessed_mapping)
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self):
        return len(self._sliced_image_list)


def slice_image(
    image: str | Image.Image,
    coco_annotation_list: list[CocoAnnotation] | None = None,
    output_file_name: str | None = None,
    output_dir: str | None = None,
    slice_height: int | None = None,
    slice_width: int | None = None,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
    auto_slice_resolution: bool | None = True,
    min_area_ratio: float | None = 0.1,
    out_ext: str | None = None,
    verbose: bool | None = False,
    exif_fix: bool = True,
) -> SliceImageResult:
    """Slice a large image into smaller windows. If output_file_name and output_dir is given, export sliced images.

    Args:
        image (str or PIL.Image): File path of image or Pillow Image to be sliced.
        coco_annotation_list (List[CocoAnnotation], optional): List of CocoAnnotation objects.
        output_file_name (str, optional): Root name of output files (coordinates will
            be appended to this)
        output_dir (str, optional): Output directory
        slice_height (int, optional): Height of each slice. Default None.
        slice_width (int, optional): Width of each slice. Default None.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        auto_slice_resolution (bool, optional): if not set slice parameters such as slice_height and slice_width,
            it enables automatically calculate these params from image resolution and orientation.
        min_area_ratio (float, optional): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix for lossless image formats and png for lossy formats ('.jpg','.jpeg').
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.
        exif_fix (bool): Whether to apply an EXIF fix to the image.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                image_dir: str
                                    Directory of the sliced image exports.
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
    """

    # define verboseprint
    verboselog = logger.info if verbose else lambda *a, **k: None

    def _export_single_slice(image: np.ndarray, output_dir: str, slice_file_name: str):
        image_pil = read_image_as_pil(image, exif_fix=exif_fix)
        slice_file_path = str(Path(output_dir) / slice_file_name)
        # export sliced image
        image_pil.save(slice_file_path)
        image_pil.close()  # to fix https://github.com/obss/sahi/issues/565
        verboselog("sliced image path: " + slice_file_path)

    # create outdir if not present
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # read image
    image_pil = read_image_as_pil(image, exif_fix=exif_fix)
    verboselog("image.shape: " + str(image_pil.size))

    image_width, image_height = image_pil.size
    if not (image_width != 0 and image_height != 0):
        raise RuntimeError(f"invalid image size: {image_pil.size} for 'slice_image'.")
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        auto_slice_resolution=auto_slice_resolution,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    n_ims = 0

    # init images and annotations lists
    sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width], image_dir=output_dir)

    image_pil_arr = np.asarray(image_pil)
    # iterate over slices
    for slice_bbox in slice_bboxes:
        n_ims += 1

        # extract image
        tlx = slice_bbox[0]
        tly = slice_bbox[1]
        brx = slice_bbox[2]
        bry = slice_bbox[3]
        image_pil_slice = image_pil_arr[tly:bry, tlx:brx]

        # set image file suffixes
        slice_suffixes = "_".join(map(str, slice_bbox))
        if out_ext:
            suffix = out_ext
        elif hasattr(image_pil, "filename"):
            suffix = Path(getattr(image_pil, "filename")).suffix
            if suffix in IMAGE_EXTENSIONS_LOSSY:
                suffix = ".png"
            elif suffix in IMAGE_EXTENSIONS_LOSSLESS:
                suffix = Path(image_pil.filename).suffix
        else:
            suffix = ".png"

        # set image file name and path
        slice_file_name = f"{output_file_name}_{slice_suffixes}{suffix}"

        # create coco image
        slice_width = slice_bbox[2] - slice_bbox[0]
        slice_height = slice_bbox[3] - slice_bbox[1]
        coco_image = CocoImage(file_name=slice_file_name, height=slice_height, width=slice_width)

        # append coco annotations (if present) to coco image
        if coco_annotation_list is not None:
            for sliced_coco_annotation in process_coco_annotations(coco_annotation_list, slice_bbox, min_area_ratio):
                coco_image.add_annotation(sliced_coco_annotation)

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=image_pil_slice, coco_image=coco_image, starting_pixel=[slice_bbox[0], slice_bbox[1]]
        )
        sliced_image_result.add_sliced_image(sliced_image)

    # export slices if output directory is provided
    if output_file_name and output_dir:
        # Use a context-managed ThreadPoolExecutor for clean shutdown and
        # limit workers based on CPU count to avoid oversubscription.
        max_workers = min(MAX_WORKERS, len(sliced_image_result))
        max_workers = max(1, max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # map will schedule tasks and wait for completion when the context exits
            list(
                executor.map(
                    _export_single_slice,
                    sliced_image_result.images,
                    [output_dir] * len(sliced_image_result),
                    sliced_image_result.filenames,
                )
            )

    verboselog(
        "Num slices: " + str(n_ims) + " slice_height: " + str(slice_height) + " slice_width: " + str(slice_width)
    )

    return sliced_image_result


def slice_coco(
    coco_annotation_file_path: str,
    image_dir: str,
    output_coco_annotation_file_name: str,
    output_dir: str | None = None,
    ignore_negative_samples: bool | None = False,
    slice_height: int | None = 512,
    slice_width: int | None = 512,
    overlap_height_ratio: float | None = 0.2,
    overlap_width_ratio: float | None = 0.2,
    min_area_ratio: float | None = 0.1,
    out_ext: str | None = None,
    verbose: bool | None = False,
    exif_fix: bool = True,
) -> list[dict | str]:
    """Slice large images given in a directory, into smaller windows. If output_dir is given, export sliced images and
    coco file.

    Args:
        coco_annotation_file_path (str): Location of the coco annotation file
        image_dir (str): Base directory for the images
        output_coco_annotation_file_name (str): File name of the exported coco
            dataset json.
        output_dir (str, optional): Output directory
        ignore_negative_samples (bool, optional): If True, images without annotations
            are ignored. Defaults to False.
        slice_height (int, optional): Height of each slice. Default 512.
        slice_width (int, optional): Width of each slice. Default 512.
        overlap_height_ratio (float, optional): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float, optional): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
        exif_fix (bool, optional): Whether to apply an EXIF fix to the image.

    Returns:
        coco_dict: dict
            COCO dict for sliced images and annotations
        save_path: str
            Path to the saved coco file
    """

    # read coco file
    coco_dict: dict = load_json(coco_annotation_file_path)
    # create image_id_to_annotation_list mapping
    coco = Coco.from_coco_dict_or_path(coco_dict)
    # init sliced coco_utils.CocoImage list
    sliced_coco_images: list = []

    # iterate over images and slice
    for idx, coco_image in enumerate(tqdm(coco.images)):
        # get image path
        image_path: str = os.path.join(image_dir, coco_image.file_name)
        # get annotation json list corresponding to selected coco image
        # slice image
        try:
            slice_image_result = slice_image(
                image=image_path,
                coco_annotation_list=coco_image.annotations,
                output_file_name=f"{Path(coco_image.file_name).stem}_{idx}",
                output_dir=output_dir,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                min_area_ratio=min_area_ratio,
                out_ext=out_ext,
                verbose=verbose,
                exif_fix=exif_fix,
            )
            # append slice outputs
            sliced_coco_images.extend(slice_image_result.coco_images)
        except TopologicalError:
            logger.warning(f"Invalid annotation found, skipping this image: {image_path}")

    # create and save coco dict
    coco_dict = create_coco_dict(
        sliced_coco_images, coco_dict["categories"], ignore_negative_samples=ignore_negative_samples
    )
    save_path = ""
    if output_coco_annotation_file_name and output_dir:
        save_path = Path(output_dir) / (output_coco_annotation_file_name + "_coco.json")
        save_json(coco_dict, save_path)

    return coco_dict, save_path


def calc_ratio_and_slice(orientation: Literal["vertical", "horizontal", "square"], slide: int = 1, ratio: float = 0.1):
    """
    According to image resolution calculation overlap params
    Args:
        orientation: image capture angle
        slide: sliding window
        ratio: buffer value

    Returns:
        overlap params
    """
    if orientation == "vertical":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide * 2, ratio, ratio
    elif orientation == "horizontal":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide * 2, slide, ratio, ratio
    elif orientation == "square":
        slice_row, slice_col, overlap_height_ratio, overlap_width_ratio = slide, slide, ratio, ratio
    else:
        raise ValueError(f"Invalid orientation: {orientation}. Must be one of 'vertical', 'horizontal', or 'square'.")

    return slice_row, slice_col, overlap_height_ratio, overlap_width_ratio


def calc_resolution_factor(resolution: int) -> int:
    """
    According to image resolution calculate power(2,n) and return the closest smaller `n`.
    Args:
        resolution: the width and height of the image multiplied. such as 1024x720 = 737280

    Returns:

    """
    expo = 0
    while np.power(2, expo) < resolution:
        expo += 1

    return expo - 1


def calc_aspect_ratio_orientation(width: int, height: int) -> str:
    """

    Args:
        width:
        height:

    Returns:
        image capture orientation
    """

    if width < height:
        return "vertical"
    elif width > height:
        return "horizontal"
    else:
        return "square"


def calc_slice_and_overlap_params(
    resolution: str, height: int, width: int, orientation: str
) -> tuple[int, int, int, int]:
    """
    This function calculate according to image resolution slice and overlap params.
    Args:
        resolution: str
        height: int
        width: int
        orientation: str

    Returns:
        x_overlap, y_overlap, slice_width, slice_height
    """

    if resolution == "medium":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=1, ratio=0.8
        )

    elif resolution == "high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=2, ratio=0.4
        )

    elif resolution == "ultra-high":
        split_row, split_col, overlap_height_ratio, overlap_width_ratio = calc_ratio_and_slice(
            orientation, slide=4, ratio=0.4
        )
    else:  # low condition
        split_col = 1
        split_row = 1
        overlap_width_ratio = 1
        overlap_height_ratio = 1

    slice_height = height // split_col
    slice_width = width // split_row

    x_overlap = int(slice_width * overlap_width_ratio)
    y_overlap = int(slice_height * overlap_height_ratio)

    return x_overlap, y_overlap, slice_width, slice_height


def get_resolution_selector(res: str, height: int, width: int) -> tuple[int, int, int, int]:
    """

    Args:
        res: resolution of image such as low, medium
        height:
        width:

    Returns:
        trigger slicing params function and return overlap params
    """
    orientation = calc_aspect_ratio_orientation(width=width, height=height)
    x_overlap, y_overlap, slice_width, slice_height = calc_slice_and_overlap_params(
        resolution=res, height=height, width=width, orientation=orientation
    )

    return x_overlap, y_overlap, slice_width, slice_height


def get_auto_slice_params(height: int, width: int) -> tuple[int, int, int, int]:
    """
    According to Image HxW calculate overlap sliding window and buffer params
    factor is the power value of 2 closest to the image resolution.
        factor <= 18: low resolution image such as 300x300, 640x640
        18 < factor <= 21: medium resolution image such as 1024x1024, 1336x960
        21 < factor <= 24: high resolution image such as 2048x2048, 2048x4096, 4096x4096
        factor > 24: ultra-high resolution image such as 6380x6380, 4096x8192
    Args:
        height:
        width:

    Returns:
        slicing overlap params x_overlap, y_overlap, slice_width, slice_height
    """
    resolution = height * width
    factor = calc_resolution_factor(resolution)
    if factor <= 18:
        return get_resolution_selector("low", height=height, width=width)
    elif 18 <= factor < 21:
        return get_resolution_selector("medium", height=height, width=width)
    elif 21 <= factor < 24:
        return get_resolution_selector("high", height=height, width=width)
    else:
        return get_resolution_selector("ultra-high", height=height, width=width)


def shift_bboxes(bboxes, offset: Sequence[int]):
    """Shift bboxes w.r.t offset.

    Suppo

    Args:
        bboxes (Tensor, np.ndarray, list): The bboxes need to be translated. Its shape can
            be (n, 4), which means (x, y, x, y).
        offset (Sequence[int]): The translation offsets with shape of (2, ).
    Returns:
        Tensor, np.ndarray, list: Shifted bboxes.
    """
    shifted_bboxes = []

    if type(bboxes).__module__ == "torch":
        bboxes_is_torch_tensor = True
    else:
        bboxes_is_torch_tensor = False

    for bbox in bboxes:
        if bboxes_is_torch_tensor or isinstance(bbox, np.ndarray):
            bbox = bbox.tolist()
        bbox = BoundingBox(bbox, shift_amount=offset)
        bbox = bbox.get_shifted_box()
        shifted_bboxes.append(bbox.to_xyxy())

    if isinstance(bboxes, np.ndarray):
        return np.stack(shifted_bboxes, axis=0)
    elif bboxes_is_torch_tensor:
        return bboxes.new_tensor(shifted_bboxes)
    else:
        return shifted_bboxes


def shift_masks(masks: np.ndarray, offset: Sequence[int], full_shape: Sequence[int]) -> np.ndarray:
    """Shift masks to the original image.

    Args:
        masks (np.ndarray): masks that need to be shifted.
        offset (Sequence[int]): The offset to translate with shape of (2, ).
        full_shape (Sequence[int]): A (height, width) tuple of the huge image's shape.
    Returns:
        np.ndarray: Shifted masks.
    """
    # empty masks
    if masks is None:
        return masks

    shifted_masks = []
    for mask in masks:
        mask = Mask(segmentation=mask, shift_amount=offset, full_shape=full_shape)
        mask = mask.get_shifted_mask()
        shifted_masks.append(mask.bool_mask)

    return np.stack(shifted_masks, axis=0)


# ============================================================================
# ADVANCED SLICING STRATEGIES AND UTILITIES
# ============================================================================


def calculate_optimal_slice_size(
    image_height: int,
    image_width: int,
    model_input_size: int = 640,
    min_overlap_ratio: float = 0.2,
    max_slices: int = 100,
    target_object_size: int = None,
) -> tuple:
    """Calculate optimal slice size and overlap based on image and model characteristics.
    
    This function dynamically determines the best slicing parameters to balance
    detection accuracy and computational efficiency.
    
    Args:
        image_height: Height of the input image
        image_width: Width of the input image
        model_input_size: Expected input size of the detection model
        min_overlap_ratio: Minimum overlap ratio between adjacent slices
        max_slices: Maximum number of slices to generate
        target_object_size: Expected size of objects to detect (for optimization)
    
    Returns:
        Tuple of (slice_height, slice_width, overlap_height_ratio, overlap_width_ratio)
    """
    # Calculate initial slice size based on model input
    base_slice_size = model_input_size
    
    # Determine number of slices needed
    min_overlap = int(base_slice_size * min_overlap_ratio)
    
    # Calculate optimal number of slices per dimension
    num_h_slices = max(1, int(np.ceil((image_height - min_overlap) / (base_slice_size - min_overlap))))
    num_w_slices = max(1, int(np.ceil((image_width - min_overlap) / (base_slice_size - min_overlap))))
    
    total_slices = num_h_slices * num_w_slices
    
    # Adjust if exceeds max_slices
    if total_slices > max_slices:
        scale_factor = np.sqrt(max_slices / total_slices)
        num_h_slices = max(1, int(num_h_slices * scale_factor))
        num_w_slices = max(1, int(num_w_slices * scale_factor))
    
    # Calculate actual slice sizes
    slice_height = int(np.ceil(image_height / num_h_slices) + min_overlap)
    slice_width = int(np.ceil(image_width / num_w_slices) + min_overlap)
    
    # Calculate overlap ratios
    overlap_height_ratio = min_overlap / slice_height
    overlap_width_ratio = min_overlap / slice_width
    
    # Adjust for target object size if provided
    if target_object_size:
        # Ensure slices are at least 3x the target object size
        min_slice_size = target_object_size * 3
        slice_height = max(slice_height, min_slice_size)
        slice_width = max(slice_width, min_slice_size)
    
    return slice_height, slice_width, overlap_height_ratio, overlap_width_ratio


def generate_adaptive_grid(
    image_height: int,
    image_width: int,
    density_map: np.ndarray = None,
    base_slice_size: int = 640,
    high_density_threshold: float = 0.7,
    min_slice_size: int = 320,
    max_slice_size: int = 1280,
) -> list:
    """Generate adaptive slicing grid based on object density.
    
    Creates variable-sized slices with smaller slices in high-density regions
    and larger slices in low-density regions for computational efficiency.
    
    Args:
        image_height: Height of the input image
        image_width: Width of the input image
        density_map: Optional density map indicating object distribution (0-1)
        base_slice_size: Base slice size for medium density regions
        high_density_threshold: Threshold for high-density regions
        min_slice_size: Minimum slice size for high-density regions
        max_slice_size: Maximum slice size for low-density regions
    
    Returns:
        List of slice coordinates [(x, y, width, height), ...]
    """
    slices = []
    
    # If no density map provided, use uniform grid
    if density_map is None:
        y = 0
        while y < image_height:
            x = 0
            while x < image_width:
                slice_w = min(base_slice_size, image_width - x)
                slice_h = min(base_slice_size, image_height - y)
                slices.append((x, y, slice_w, slice_h))
                x += base_slice_size
            y += base_slice_size
        return slices
    
    # Adaptive grid based on density map
    grid_resolution = 10  # Divide image into grid for density analysis
    grid_h = image_height // grid_resolution
    grid_w = image_width // grid_resolution
    
    # Resize density map to grid resolution
    from sahi.utils.cv import cv2
    density_grid = cv2.resize(density_map, (grid_resolution, grid_resolution))
    
    y = 0
    while y < image_height:
        x = 0
        while x < image_width:
            # Determine local density
            grid_y = min(int(y / grid_h), grid_resolution - 1)
            grid_x = min(int(x / grid_w), grid_resolution - 1)
            local_density = density_grid[grid_y, grid_x]
            
            # Adjust slice size based on density
            if local_density > high_density_threshold:
                # High density - use smaller slices
                slice_size = min_slice_size
            elif local_density < (1 - high_density_threshold):
                # Low density - use larger slices
                slice_size = max_slice_size
            else:
                # Medium density - use base slice size
                slice_size = base_slice_size
            
            slice_w = min(slice_size, image_width - x)
            slice_h = min(slice_size, image_height - y)
            slices.append((x, y, slice_w, slice_h))
            
            x += int(slice_size * 0.8)  # 20% overlap
        y += int(slice_size * 0.8)
    
    return slices


def calculate_slice_overlap_iou(slice1: tuple, slice2: tuple) -> float:
    """Calculate IoU (Intersection over Union) between two slices.
    
    Args:
        slice1: First slice coordinates (x, y, width, height)
        slice2: Second slice coordinates (x, y, width, height)
    
    Returns:
        IoU value between 0 and 1
    """
    x1_1, y1_1, w1, h1 = slice1
    x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    
    x1_2, y1_2, w2, h2 = slice2
    x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    union = (w1 * h1) + (w2 * h2) - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_overlapping_predictions(
    predictions_list: list,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.3,
) -> list:
    """Merge predictions from overlapping slices using weighted averaging.
    
    Args:
        predictions_list: List of ObjectPrediction lists from different slices
        iou_threshold: IoU threshold for considering predictions as duplicates
        score_threshold: Minimum confidence score to consider
    
    Returns:
        List of merged ObjectPrediction instances
    """
    from sahi.prediction import ObjectPrediction
    
    # Flatten all predictions
    all_predictions = []
    for preds in predictions_list:
        all_predictions.extend([p for p in preds if p.score.value >= score_threshold])
    
    if not all_predictions:
        return []
    
    # Sort by score descending
    all_predictions.sort(key=lambda x: x.score.value, reverse=True)
    
    merged = []
    used = [False] * len(all_predictions)
    
    for i, pred1 in enumerate(all_predictions):
        if used[i]:
            continue
        
        # Find all overlapping predictions of same class
        overlapping = [pred1]
        overlapping_indices = [i]
        
        for j, pred2 in enumerate(all_predictions[i+1:], start=i+1):
            if used[j]:
                continue
            
            if pred1.category.id != pred2.category.id:
                continue
            
            # Calculate IoU
            bbox1 = pred1.bbox.to_xyxy()
            bbox2 = pred2.bbox.to_xyxy()
            
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
            
            if x2 < x1 or y2 < y1:
                continue
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            union = area1 + area2 - intersection
            iou = intersection / union if union > 0 else 0
            
            if iou >= iou_threshold:
                overlapping.append(pred2)
                overlapping_indices.append(j)
        
        # Mark as used
        for idx in overlapping_indices:
            used[idx] = True
        
        # Merge overlapping predictions
        if len(overlapping) == 1:
            merged.append(overlapping[0])
        else:
            # Weighted average by confidence scores
            total_weight = sum(p.score.value for p in overlapping)
            weights = [p.score.value / total_weight for p in overlapping]
            
            # Average bbox
            bboxes = np.array([p.bbox.to_xyxy() for p in overlapping])
            avg_bbox = np.average(bboxes, axis=0, weights=weights)
            
            # Average score
            avg_score = np.average([p.score.value for p in overlapping], weights=weights)
            
            # Create merged prediction
            merged_pred = overlapping[0].deepcopy()
            merged_pred.bbox.minx = avg_bbox[0]
            merged_pred.bbox.miny = avg_bbox[1]
            merged_pred.bbox.maxx = avg_bbox[2]
            merged_pred.bbox.maxy = avg_bbox[3]
            merged_pred.score.value = float(avg_score)
            
            merged.append(merged_pred)
    
    return merged


def visualize_slicing_grid(
    image_height: int,
    image_width: int,
    slices: list,
    output_path: str = None,
    show_overlap: bool = True,
) -> np.ndarray:
    """Visualize slicing grid on image.
    
    Args:
        image_height: Height of the image
        image_width: Width of the image
        slices: List of slice coordinates [(x, y, width, height), ...]
        output_path: Optional path to save visualization
        show_overlap: Whether to highlight overlapping regions
    
    Returns:
        Visualization image as numpy array
    """
    from sahi.utils.cv import cv2
    
    # Create blank image
    vis = np.ones((image_height, image_width, 3), dtype=np.uint8) * 255
    
    # Draw slices
    for i, (x, y, w, h) in enumerate(slices):
        # Draw rectangle
        color = (0, 255, 0) if i % 2 == 0 else (0, 200, 0)
        cv2.rectangle(vis, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        
        # Add slice number
        cv2.putText(vis, str(i), (int(x + 10), int(y + 30)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Highlight overlap regions if requested
        if show_overlap and i > 0:
            for j in range(i):
                iou = calculate_slice_overlap_iou((x, y, w, h), slices[j])
                if iou > 0:
                    # Find overlap region
                    x2, y2, w2, h2 = slices[j]
                    x_overlap = max(x, x2)
                    y_overlap = max(y, y2)
                    x2_overlap = min(x + w, x2 + w2)
                    y2_overlap = min(y + h, y2 + h2)
                    
                    # Draw semi-transparent overlay
                    overlay = vis.copy()
                    cv2.rectangle(overlay,
                                (int(x_overlap), int(y_overlap)),
                                (int(x2_overlap), int(y2_overlap)),
                                (0, 0, 255), -1)
                    cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
    
    if output_path:
        cv2.imwrite(output_path, vis)
    
    return vis


def estimate_memory_usage(
    image_height: int,
    image_width: int,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
    bytes_per_pixel: int = 3,
    model_memory_mb: float = 500,
) -> dict:
    """Estimate memory usage for sliced inference.
    
    Args:
        image_height: Height of the input image
        image_width: Width of the input image
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_height_ratio: Overlap ratio in height dimension
        overlap_width_ratio: Overlap ratio in width dimension
        bytes_per_pixel: Number of bytes per pixel (3 for RGB)
        model_memory_mb: Estimated model memory usage in MB
    
    Returns:
        Dictionary with memory usage estimates
    """
    # Calculate number of slices
    overlap_height = int(slice_height * overlap_height_ratio)
    overlap_width = int(slice_width * overlap_width_ratio)
    
    stride_height = slice_height - overlap_height
    stride_width = slice_width - overlap_width
    
    num_h_slices = int(np.ceil((image_height - overlap_height) / stride_height))
    num_w_slices = int(np.ceil((image_width - overlap_width) / stride_width))
    num_slices = num_h_slices * num_w_slices
    
    # Calculate memory usage
    full_image_mb = (image_height * image_width * bytes_per_pixel) / (1024 ** 2)
    slice_image_mb = (slice_height * slice_width * bytes_per_pixel) / (1024 ** 2)
    
    # Estimate peak memory (image + model + slice)
    peak_memory_mb = full_image_mb + model_memory_mb + slice_image_mb
    
    # Estimate total processing memory
    total_memory_mb = peak_memory_mb * 1.5  # Add 50% buffer
    
    return {
        "num_slices": num_slices,
        "full_image_mb": full_image_mb,
        "slice_image_mb": slice_image_mb,
        "peak_memory_mb": peak_memory_mb,
        "total_memory_mb": total_memory_mb,
        "estimated_time_multiplier": num_slices,
    }


def optimize_slicing_parameters(
    image_height: int,
    image_width: int,
    model_input_size: int = 640,
    max_memory_mb: float = 8000,
    min_overlap_ratio: float = 0.1,
    max_overlap_ratio: float = 0.3,
    target_num_slices: int = None,
) -> dict:
    """Optimize slicing parameters for given constraints.
    
    Args:
        image_height: Height of the input image
        image_width: Width of the input image
        model_input_size: Expected input size of the detection model
        max_memory_mb: Maximum available memory in MB
        min_overlap_ratio: Minimum overlap ratio
        max_overlap_ratio: Maximum overlap ratio
        target_num_slices: Target number of slices (optional)
    
    Returns:
        Dictionary with optimized parameters
    """
    best_config = None
    best_score = -1
    
    # Try different configurations
    for overlap_ratio in np.linspace(min_overlap_ratio, max_overlap_ratio, 5):
        for scale_factor in [0.8, 1.0, 1.2, 1.5, 2.0]:
            slice_size = int(model_input_size * scale_factor)
            
            memory_est = estimate_memory_usage(
                image_height, image_width,
                slice_size, slice_size,
                overlap_ratio, overlap_ratio
            )
            
            # Check memory constraint
            if memory_est["peak_memory_mb"] > max_memory_mb:
                continue
            
            # Calculate score (balance between num_slices and coverage)
            num_slices = memory_est["num_slices"]
            
            if target_num_slices:
                # Prefer configurations close to target
                slice_score = 1.0 - abs(num_slices - target_num_slices) / target_num_slices
            else:
                # Prefer fewer slices with good overlap
                slice_score = 1.0 / (1.0 + num_slices) + overlap_ratio
            
            if slice_score > best_score:
                best_score = slice_score
                best_config = {
                    "slice_height": slice_size,
                    "slice_width": slice_size,
                    "overlap_height_ratio": overlap_ratio,
                    "overlap_width_ratio": overlap_ratio,
                    "num_slices": num_slices,
                    "memory_estimate": memory_est,
                }
    
    return best_config if best_config else {
        "slice_height": model_input_size,
        "slice_width": model_input_size,
        "overlap_height_ratio": 0.2,
        "overlap_width_ratio": 0.2,
    }
