# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import time
from typing import Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from sahi.utils.coco import Coco, CocoAnnotation, CocoImage, create_coco_dict
from sahi.utils.cv import read_large_image
from sahi.utils.file import create_dir, load_json, save_json
from sahi.utils.shapely import (ShapelyAnnotation, get_bbox_from_shapely,
                                get_shapely_box, get_shapely_multipolygon)


def slice_coco_annotations_by_box(
    coco_annotation_list, box
) -> (List[CocoAnnotation], int):
    """
    Crops all annotations for a given grid. Returns processed annotation list.

    Args:
        coco_annotations: list of CocoAnnotation
            List of CocoAnnotation objects that belong to given COCO image
        box: list
            Points of slice box [x: int, y: int, width: int, height: int]
    Returns:
        sliced_annotations: list of CocoAnnotation
            Sliced list of CocoAnnotation objects that belong to given COCO image
    """

    sliced_annotations = []
    num_invalid_segmentation = 0
    for coco_annotation in coco_annotation_list:
        # calculate intersection polygon btw slice and annotation
        x, y, width, height = box
        shapely_box = get_shapely_box(x, y, width, height)

        shapely_annotation = coco_annotation._shapely_annotation

        invalid_segmentation = False
        if shapely_annotation.multipolygon.is_valid is True:
            intersection_shapely_annotation = shapely_annotation.get_intersection(
                shapely_box
            )
        else:
            try:
                shapely_annotation = shapely_annotation.get_buffered_shapely_annotation(
                    distance=3
                )
                intersection_shapely_annotation = shapely_annotation.get_intersection(
                    shapely_box
                )
            except:
                invalid_segmentation = True
                num_invalid_segmentation += 1

        # store intersection polygon if intersection area is greater than 0
        if (invalid_segmentation is False) and (
            intersection_shapely_annotation.area > 0
        ):
            new_annotation = CocoAnnotation.from_coco_segmentation(
                segmentation=intersection_shapely_annotation.to_coco_segmentation(),
                category_id=coco_annotation.category_id,
                category_name=coco_annotation.category_name,
            )

            sliced_annotations.append(new_annotation)

    return sliced_annotations, num_invalid_segmentation


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
        assert (
            type(sliced_image) == SlicedImage
        ), "sliced_image must be a SlicedImage instance"
        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self):
        return self._sliced_image_list

    @property
    def images(self):
        """
        Returns sliced images.

        Returns:
            images: a list of np.array
        """
        images = []
        for sliced_image in self._sliced_image_list:
            images.append(sliced_image.image)
        return images

    @property
    def coco_images(self):
        """
        Returns CocoImage representation of SliceImageResult.

        Returns:
            coco_images: a list of CocoImage
        """
        coco_images = []
        for sliced_image in self._sliced_image_list:
            coco_images.append(sliced_image.coco_image)
        return coco_images

    @property
    def starting_pixels(self):
        """
        Returns a list of starting pixels for each slice.

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
    image,
    coco_annotation_list=None,
    output_file_name: str = "",
    output_dir: str = "",
    slice_height: int = 256,
    slice_width: int = 256,
    max_allowed_zeros_ratio: float = 0.2,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    slice_sep: str = "_",
    out_ext: str = ".png",
    verbose: bool = False,
) -> (SliceImageResult, int):

    """
    Slice a large image into smaller windows. If output_file_name is given export sliced images.

    Args:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        coco_annotation_list: list of CocoAnnotation
            List of CocoAnnotation objects that belong to given COCO image.
        output_file_name: str
            Root name of output files (coordinates will be appended to this)
        output_dir: str
            Output directory
        slice_height: int
            Height of each slice. Defaults to ``256``.
        slice_width: int
            Width of each slice. Defaults to ``256``.
        max_allowed_zeros_ratio: float
            Maximum fraction of window that is allowed to be zeros or null.
            Defaults to ``0.2``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        slice_sep: str
            Character used to separate outname from coordinates in the saved
            windows. Defaults to ``|``
        out_ext: str
            Extension of saved images. Defaults to ``.png``.
        verbose: bool
            Switch to print relevant values to screen. Defaults to ``False``

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

    # read image if str image path is provided
    if isinstance(image, str):
        # read in image, cv2 fails on large files
        verboseprint("Read in image:", image)
        image0, use_cv2 = read_large_image(image)
    else:
        image0 = image
    verboseprint("image.shape:", image0.shape)

    if len(out_ext) == 0:
        ext = "." + image.split(".")[-1]
    else:
        ext = out_ext

    win_h, win_w = image0.shape[:2]

    # if slice sizes are large than image, pad the edges
    pad = 0
    if slice_height > win_h:
        pad = slice_height - win_h
    if slice_width > win_w:
        pad = max(pad, slice_width - win_w)
    # pad the edge of the image with black pixels
    if pad > 0:
        border_color = (0, 0, 0)
        image0 = cv2.copyMakeBorder(
            image0, 0, pad, 0, pad, cv2.BORDER_CONSTANT, value=border_color
        )

    win_size = slice_height * slice_width

    t0 = time.time()
    n_ims = 0
    n_ims_nonull = 0
    dx = int((1.0 - overlap_width_ratio) * slice_width)
    dy = int((1.0 - overlap_height_ratio) * slice_height)

    # init images and annotations lists
    sliced_image_result = SliceImageResult(
        original_image_size=[win_h, win_w], image_dir=output_dir
    )
    num_total_invalid_segmentation = 0

    # iterate over slices
    for y0 in range(0, image0.shape[0], dy):  # slice_height):
        for x0 in range(0, image0.shape[1], dx):  # slice_width):
            n_ims += 1

            if (n_ims % 50) == 0:
                verboseprint(n_ims)

            # make sure we don't have a tiny image on the edge
            if y0 + slice_height > image0.shape[0]:
                y = image0.shape[0] - slice_height
            else:
                y = y0
            if x0 + slice_width > image0.shape[1]:
                x = image0.shape[1] - slice_width
            else:
                x = x0

            # extract image
            window_c = image0[y : y + slice_height, x : x + slice_width]

            # process annotations if coco_annotations is given
            if coco_annotation_list:
                slice_box = [x, y, slice_width, slice_height]
                (
                    sliced_coco_annotation_list,
                    num_invalid_segmentation,
                ) = slice_coco_annotations_by_box(coco_annotation_list, box=slice_box)
                num_total_invalid_segmentation = (
                    num_total_invalid_segmentation + num_invalid_segmentation
                )

            # get black and white image
            window = cv2.cvtColor(window_c, cv2.COLOR_RGB2GRAY)

            # find threshold that's not black
            # https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html?highlight=threshold
            ret, thresh1 = cv2.threshold(window, 2, 255, cv2.THRESH_BINARY)
            non_zero_counts = cv2.countNonZero(thresh1)
            zero_counts = win_size - non_zero_counts
            zero_frac = float(zero_counts) / win_size
            # skip if image is mostly empty
            if zero_frac >= max_allowed_zeros_ratio:
                verboseprint("Zero frac too high at:", zero_frac)
                continue
            else:
                # save if out_name is given
                if output_file_name and output_dir:
                    outpath = os.path.join(
                        output_dir,
                        output_file_name
                        + slice_sep
                        + str(y)
                        + "_"
                        + str(x)
                        + "_"
                        + str(slice_height)
                        + "_"
                        + str(slice_width)
                        + "_"
                        + str(pad)
                        + "_"
                        + str(win_w)
                        + "_"
                        + str(win_h)
                        + ext,
                    )

                    verboseprint("outpath:", outpath)
                    # if large image, convert to bgr prior to saving
                    if not use_cv2:
                        try:
                            import skimage.io
                        except ImportError:
                            raise ImportError(
                                'Please run "pip install -U scikit-image" '
                                'to install scikit-image first for large image handling.')
                        skimage.io.imsave(outpath, window_c)
                    else:
                        window_c = cv2.cvtColor(window_c, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(outpath, window_c)
                    n_ims_nonull += 1

                    file_name = outpath.split(output_dir)[-1].replace(os.sep, "")
                else:
                    file_name = ""

                # create coco image
                coco_image = CocoImage(
                    file_name=file_name, height=slice_height, width=slice_width
                )

                # append coco annotations (if present) to coco image
                if coco_annotation_list:
                    for coco_annotation in sliced_coco_annotation_list:
                        coco_image.add_annotation(coco_annotation)

                # create sliced image and append to sliced_image_result
                sliced_image = SlicedImage(
                    image=window_c, coco_image=coco_image, starting_pixel=[x, y]
                )
                sliced_image_result.add_sliced_image(sliced_image)

    verboseprint(
        "Num slices:",
        n_ims,
        "Num non-null slices:",
        n_ims_nonull,
        "slice_height",
        slice_height,
        "slice_width",
        slice_width,
    )
    verboseprint("Time to slice", image, time.time() - t0, "seconds")

    return (
        sliced_image_result,
        num_total_invalid_segmentation,
    )


def slice_coco(
    coco_annotation_file_path: str,
    image_dir: str,
    output_coco_annotation_file_name: str = "",
    output_dir: str = "",
    ignore_negative_samples: bool = True,
    slice_height: int = 256,
    slice_width: int = 256,
    max_allowed_zeros_ratio: float = 0.2,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    slice_sep: str = "_",
    out_ext: str = ".png",
    verbose: bool = False,
):

    """
    Slice large images given in a directory, into smaller windows. If out_name is given export sliced images and coco file.

    Args:
        coco_annotation_file_path: str
            Location of the coco annotation file
        image_dir: str
            Base diectory for the images
        output_coco_annotation_file_name : str
            Root name of the exported coco datatset file
        output_dir: str
            Output directory
        ignore_negative_samples: bool
            If True, images without annotations are ignored. Defaults to True.
        slice_height: int
            Height of each slice.  Defaults to ``256``.
        slice_width: int
            Width of each slice.  Defaults to ``256``.
        max_allowed_zeros_ratio: float
            Maximum fraction of window that is allowed to be zeros or null.
            Defaults to ``0.2``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        slice_sep: str
            Character used to separate outname from coordinates in the saved
            windows. Defaults to ``|``
        out_ext: str
            Extension of saved images. Defaults to ``.png``.
        verbose: bool
            Switch to print relevant values to screen. Defaults to ``False``

    Returns:
        coco_dict: dict
            COCO dict for sliced images and annotations
        save_path: str
            Path to the saved coco file
    """
    # define verboseprint
    verboseprint = print if verbose else lambda *a, **k: None

    # read coco file
    coco_dict = load_json(coco_annotation_file_path)
    # create coco_utils.Coco object
    coco = Coco.from_coco_dict_or_path(coco_dict)
    # init sliced coco_utils.CocoImage list
    sliced_coco_images = []

    num_total_invalid_segmentation = 0

    # iterate over images and slice
    for coco_image in tqdm(coco.images):
        # get image path
        image_path = os.path.join(image_dir, coco_image.file_name)
        # get coco_utils.CocoAnnotation list corresponding to selected coco_utils.CocoImage
        coco_annotation_list = coco_image.annotations
        # slice image
        slice_image_result, num_invalid_segmentation = slice_image(
            image=image_path,
            coco_annotation_list=coco_annotation_list,
            output_file_name=os.path.basename(coco_image.file_name),
            output_dir=output_dir,
            slice_height=slice_height,
            slice_width=slice_width,
            max_allowed_zeros_ratio=max_allowed_zeros_ratio,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            slice_sep="_",
            out_ext=".png",
            verbose=False,
        )
        num_total_invalid_segmentation = (
            num_total_invalid_segmentation + num_invalid_segmentation
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
        save_path = os.path.join(
            output_dir, output_coco_annotation_file_name + "_coco.json"
        )
        save_json(coco_dict, save_path)
    verboseprint("There are", num_total_invalid_segmentation, "invalid segmentations")

    return coco_dict, save_path
