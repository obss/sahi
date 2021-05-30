# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from tqdm import tqdm

from sahi.utils.coco import Coco, CocoAnnotation, CocoImage, create_coco_dict
from sahi.utils.cv import read_image_as_pil
from sahi.utils.file import create_dir, load_json, save_json


def get_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: int = 0.2,
    overlap_width_ratio: int = 0.2,
) -> List[List[int]]:
    """Slices `image_pil` in crops.
    Corner values of each slice will be generated using the `slice_height`,
    `slice_width`, `overlap_height_ratio` and `overlap_width_ratio` arguments.

    Args:
        image_height (int): Height of the original image.
        image_width (int): Width of the original image.
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio(float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio(float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.

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
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max - y_overlap < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max - x_overlap < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                ymax = min(image_height, y_max)
                xmax = min(image_width, x_max)
                slice_bboxes.append([xmax - slice_width, ymax - slice_height, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def annotation_inside_slice(annotation: Dict, slice_bbox: List[int]) -> bool:
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


def process_coco_annotations(coco_annotation_list: List[CocoAnnotation], slice_bbox: List[int], min_area_ratio) -> bool:
    """Slices and filters given list of CocoAnnotation objects with given
    'slice_bbox' and 'min_area_ratio'.

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

    sliced_coco_annotation_list: List[CocoAnnotation] = []
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
    def __init__(self, original_image_size=None, image_dir=None):
        """
        sliced_image_list: list of SlicedImage
        image_dir: str
            Directory of the sliced image exports.
        original_image_size: list of int
            Size of the unsliced original image in [height, width]
        """
        self._sliced_image_list = []
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]
        self.image_dir = image_dir

    def add_sliced_image(self, sliced_image: SlicedImage):
        assert type(sliced_image) == SlicedImage, "sliced_image must be a SlicedImage instance"
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
    def coco_images(self) -> List[CocoImage]:
        """Returns CocoImage representation of SliceImageResult.

        Returns:
            coco_images: a list of CocoImage
        """
        coco_images: List = []
        for sliced_image in self._sliced_image_list:
            coco_images.append(sliced_image.coco_image)
        return coco_images

    @property
    def starting_pixels(self) -> List[int]:
        """Returns a list of starting pixels for each slice.

        Returns:
            starting_pixels: a list of starting pixel coords [x,y]
        """
        starting_pixels = []
        for sliced_image in self._sliced_image_list:
            starting_pixels.append(sliced_image.starting_pixel)
        return starting_pixels

    def __len__(self):
        return len(self._sliced_image_list)


def slice_image(
    image: Union[str, Image.Image],
    coco_annotation_list: Optional[CocoAnnotation] = None,
    output_file_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
    out_ext: Optional[str] = None,
    verbose: bool = False,
) -> SliceImageResult:

    """Slice a large image into smaller windows. If output_file_name is given export
    sliced images.

    Args:
        image (str or PIL.Image): File path of image or Pillow Image to be sliced.
        coco_annotation_list (CocoAnnotation): List of CocoAnnotation objects.
        output_file_name (str, optional): Root name of output files (coordinates will
            be appended to this)
        output_dir (str, optional): Output directory
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio (float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.

    Returns:
        sliced_image_result: SliceImageResult:
                                sliced_image_list: list of SlicedImage
                                image_dir: str
                                    Directory of the sliced image exports.
                                original_image_size: list of int
                                    Size of the unsliced original image in [height, width]
        num_total_invalid_segmentation: int
            Number of invalid segmentation annotations.
    """

    # define verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    # create outdir if not present
    if output_dir:
        create_dir(output_dir)

    # read image
    image_pil = read_image_as_pil(image)
    verboseprint("image.shape:", image_pil.size)

    image_width, image_height = image_pil.size
    slice_bboxes = get_slice_bboxes(
        image_height=image_height,
        image_width=image_width,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    t0 = time.time()
    n_ims = 0

    # init images and annotations lists
    sliced_image_result = SliceImageResult(original_image_size=[image_height, image_width], image_dir=output_dir)

    # iterate over slices
    for slice_bbox in slice_bboxes:
        n_ims += 1

        # extract image
        image_pil_slice = image_pil.crop(slice_bbox)

        # process annotations if coco_annotations is given
        if coco_annotation_list is not None:
            sliced_coco_annotation_list = process_coco_annotations(coco_annotation_list, slice_bbox, min_area_ratio)

        # set image file suffixes
        slice_suffixes = "_".join(map(str, slice_bbox))
        if out_ext:
            suffix = out_ext
        else:
            try:
                suffix = Path(image_pil.filename).suffix
            except AttributeError:
                suffix = ".jpg"

        # set image file name and path
        slice_file_name = f"{output_file_name}_{slice_suffixes}{suffix}"
        # export image if output directory is provided
        if output_file_name and output_dir:
            slice_file_path = str(Path(output_dir) / slice_file_name)
            # export sliced image
            image_pil_slice.save(slice_file_path)
            verboseprint("sliced image path:", slice_file_path)

        # create coco image
        coco_image = CocoImage(file_name=slice_file_name, height=slice_height, width=slice_width)

        # append coco annotations (if present) to coco image
        if coco_annotation_list:
            for coco_annotation in sliced_coco_annotation_list:
                coco_image.add_annotation(coco_annotation)

        # create sliced image and append to sliced_image_result
        sliced_image = SlicedImage(
            image=np.asarray(image_pil_slice),
            coco_image=coco_image,
            starting_pixel=[slice_bbox[0], slice_bbox[1]],
        )
        sliced_image_result.add_sliced_image(sliced_image)

    verboseprint(
        "Num slices:",
        n_ims,
        "slice_height",
        slice_height,
        "slice_width",
        slice_width,
    )
    verboseprint("Time to slice", image, time.time() - t0, "seconds")

    return sliced_image_result


def slice_coco(
    coco_annotation_file_path: str,
    image_dir: str,
    output_coco_annotation_file_name: str,
    output_dir: Optional[str] = None,
    ignore_negative_samples: bool = True,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    min_area_ratio: float = 0.1,
    out_ext: Optional[str] = None,
    verbose: bool = False,
) -> List[Union[Dict, str]]:

    """
    Slice large images given in a directory, into smaller windows. If out_name is given export sliced images and coco file.

    Args:
        coco_annotation_file_pat (str): Location of the coco annotation file
        image_dir (str): Base directory for the images
        output_coco_annotation_file_name (str): File name of the exported coco
            datatset json.
        output_dir (str, optional): Output directory
        ignore_negative_samples (bool): If True, images without annotations
            are ignored. Defaults to True.
        slice_height (int): Height of each slice. Default 512.
        slice_width (int): Width of each slice. Default 512.
        overlap_height_ratio (float): Fractional overlap in height of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        overlap_width_ratio (float): Fractional overlap in width of each
            slice (e.g. an overlap of 0.2 for a slice of size 100 yields an
            overlap of 20 pixels). Default 0.2.
        min_area_ratio (float): If the cropped annotation area to original annotation
            ratio is smaller than this value, the annotation is filtered out. Default 0.1.
        out_ext (str, optional): Extension of saved images. Default is the
            original suffix.
        verbose (bool, optional): Switch to print relevant values to screen.
            Default 'False'.

    Returns:
        coco_dict: dict
            COCO dict for sliced images and annotations
        save_path: str
            Path to the saved coco file
    """

    # read coco file
    coco_dict: Dict = load_json(coco_annotation_file_path)
    # create image_id_to_annotation_list mapping
    coco = Coco.from_coco_dict_or_path(coco_dict)
    # init sliced coco_utils.CocoImage list
    sliced_coco_images: List = []

    # iterate over images and slice
    for coco_image in tqdm(coco.images):
        # get image path
        image_path: str = os.path.join(image_dir, coco_image.file_name)
        # get annotation json list corresponding to selected coco image
        # slice image
        slice_image_result = slice_image(
            image=image_path,
            coco_annotation_list=coco_image.annotations,
            output_file_name=Path(coco_image.file_name).stem,
            output_dir=output_dir,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            min_area_ratio=min_area_ratio,
            out_ext=out_ext,
            verbose=False,
        )
        # append slice outputs
        sliced_coco_images.extend(slice_image_result.coco_images)

    # create and save coco dict
    coco_dict = create_coco_dict(
        sliced_coco_images,
        coco_dict["categories"],
        ignore_negative_samples=ignore_negative_samples,
    )
    save_path = ""
    if output_coco_annotation_file_name and output_dir:
        save_path = os.path.join(output_dir, output_coco_annotation_file_name + "_coco.json")
        save_json(coco_dict, save_path)

    return coco_dict, save_path
