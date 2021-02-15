# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import time

from tqdm import tqdm

from sahi.postprocess.match import PredictionMatcher
from sahi.postprocess.merge import PredictionMerger, ScoreMergingPolicy
from sahi.postprocess.ops import box_intersection, box_ios, box_union
from sahi.prediction import ObjectPrediction, PredictionInput
from sahi.slicing import slice_image
from sahi.utils.cv import (
    crop_object_predictions,
    read_image,
    visualize_object_predictions,
)
from sahi.utils.file import get_base_filename, import_class, list_files, save_pickle
from sahi.utils.torch import to_float_tensor


def get_prediction(
    image,
    detection_model,
    shift_amount: list = [0, 0],
    full_shape=None,
    merger=None,
    matcher=None,
    verbose: int = 0,
):
    """
    Function for performing prediction for given image using given detection_model.

    Arguments:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionMode
        shift_amount: List
            To shift the box and mask predictions from sliced image to full
            sized image, should be in the form of [shift_x, shift_y]
        full_shape: List
            Size of the full image, should be in the form of [height, width]
        merger: postprocess.PredictionMerger
        matcher: postprocess.PredictionMatcher
        verbose: int
            0: no print (default)
            1: print prediction duration

    Returns:
        A dict with fields:
            object_prediction_list: a list of ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    durations_in_seconds = dict()

    # read image if image is str
    if isinstance(image, str):
        image = read_image(image)
    # get prediction
    time_start = time.time()
    detection_model.perform_inference(image)
    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    # process prediction
    time_start = time.time()
    # works only with 1 batch
    detection_model.convert_original_predictions(
        shift_amount=shift_amount,
        full_shape=full_shape,
    )
    object_prediction_list = detection_model.object_prediction_list
    # filter out predictions with lower score
    filtered_object_prediction_list = [
        object_prediction
        for object_prediction in object_prediction_list
        if object_prediction.score.score > detection_model.prediction_score_threshold
    ]
    # merge matching predictions
    if merger is not None:
        filtered_object_prediction_list = merger.merge_batch(
            matcher,
            filtered_object_prediction_list,
            merge_type="merge",
        )
    else:
        # init match merge instances
        merger = PredictionMerger(
            score_merging=ScoreMergingPolicy.LARGER_SCORE, box_merger=box_union
        )
        matcher = PredictionMatcher(threshold=0.5, scorer=box_ios)
        # merge matching predictions
        filtered_object_prediction_list = merger.merge_batch(
            matcher,
            filtered_object_prediction_list,
            merge_type="merge",
            ignore_class_label=True,
        )

    time_end = time.time() - time_start
    durations_in_seconds["postprocess"] = time_end

    if verbose == 1:
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    return {
        "object_prediction_list": filtered_object_prediction_list,
        "durations_in_seconds": durations_in_seconds,
    }


def get_sliced_prediction(
    image,
    detection_model=None,
    slice_height: int = 256,
    slice_width: int = 256,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    match_iou_threshold: float = 0.5,
    verbose: int = 1,
):
    """
    Function for slice image + get predicion for each slice + combine predictions in full image.

    Args:
        image: str or np.ndarray
            Location of image or numpy image matrix to slice
        detection_model: model.DetectionModel
        slice_height: int
            Height of each slice.  Defaults to ``256``.
        slice_width: int
            Width of each slice.  Defaults to ``256``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        match_iou_threshold: float
            Sliced predictions having higher iou than match_iou_threshold will be merged.
        verbose: int
            0: no print
            1: print number of slices (default)
            2: print number of slices and slice/prediction durations

    Returns:
        A Dict with fields:
            object_prediction_list: a list of sahi.prediction.ObjectPrediction
            durations_in_seconds: a dict containing elapsed times for profiling
    """
    # for profiling
    durations_in_seconds = dict()

    # currently only 1 batch supported
    num_batch = 1

    # create slices from full image
    time_start = time.time()
    slice_image_result, num_total_invalid_segmentation = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    num_slices = len(slice_image_result)
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end

    # init match merge instances
    merger = PredictionMerger(
        score_merging=ScoreMergingPolicy.LARGER_SCORE, box_merger=box_union
    )
    matcher = PredictionMatcher(threshold=match_iou_threshold, scorer=box_ios)

    # create prediction input
    num_group = int(num_slices / num_batch)
    if verbose == 1 or verbose == 2:
        print("Number of slices:", num_slices)
    object_prediction_list = []
    for group_ind in range(num_group):
        # prepare batch (currently supports only 1 batch)
        image_list = []
        shift_amount_list = []
        for image_ind in range(num_batch):
            image_list.append(
                slice_image_result.images[group_ind * num_batch + image_ind]
            )
            shift_amount_list.append(
                slice_image_result.starting_pixels[group_ind * num_batch + image_ind]
            )
        # perform batch prediction
        prediction_result = get_prediction(
            image=image_list[0],
            detection_model=detection_model,
            shift_amount=shift_amount_list[0],
            full_shape=[
                slice_image_result.original_image_height,
                slice_image_result.original_image_width,
            ],
        )
        object_prediction_list.extend(prediction_result["object_prediction_list"])

    # remove empty predictions
    object_prediction_list = [
        object_prediction
        for object_prediction in object_prediction_list
        if object_prediction
    ]
    # convert sliced predictions to full predictions
    full_object_prediction_list = []
    for object_prediction in object_prediction_list:
        full_object_prediction = object_prediction.get_shifted_object_prediction()
        full_object_prediction_list.append(full_object_prediction)

    time_end = time.time() - time_start
    durations_in_seconds["prediction"] = time_end

    if verbose == 2:
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )

    # merge matching predictions
    full_object_prediction_list = merger.merge_batch(
        matcher,
        full_object_prediction_list,
        merge_type="merge",
    )

    # return filtered elements of filtered full_object_prediction_list
    return {
        "object_prediction_list": full_object_prediction_list,
        "durations_in_seconds": durations_in_seconds,
    }


def predict_folder(
    model_name="MmdetDetectionModel",
    model_parameters=None,
    image_dir=None,
    visual_output_dir=None,
    pickle_dir=None,
    crop_dir=None,
    apply_sliced_prediction: bool = True,
    slice_height: int = 256,
    slice_width: int = 256,
    overlap_height_ratio: float = 0.1,
    overlap_width_ratio: float = 0.2,
    match_iou_threshold: float = 0.5,
    visual_bbox_thickness: int = 1,
    visual_text_size: float = 1,
    visual_text_thickness: int = 1,
    verbose: int = 1,
):
    """
    Performs prediction for all present images in given folder.

    Args:
        model_name: str
            Name of the implemented DetectionModel in model.py file.
        model_parameter: a dict with fields:
            model_path: str
                Path for the instance segmentation model weight
            config_path: str
                Path for the mmdetection instance segmentation model config file
            prediction_score_threshold: float
                All predictions with score < prediction_score_threshold will be discarded.
            device: str
                Torch device, "cpu" or "cuda"
            category_remapping: dict: str to int
                Remap category ids after performing inference
        image_dir: str
            Directory that contain images to be predicted.
        visual_output_dir: str
            Directory that prediction visuals are going to be exported. Set to None for no visuals.
        pickle_dir: str
            Directory that object prediction list pickles are going to be exported. Set to None for no pickles.
        crop_dir: str
            Directory that detected bounding boxes will be cropped&exported. Set to None for no crops.
        apply_sliced_prediction: bool
            Set to True if you want sliced prediction, set to False for full prediction.
        slice_height: int
            Height of each slice.  Defaults to ``256``.
        slice_width: int
            Width of each slice.  Defaults to ``256``.
        overlap_height_ratio: float
            Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        overlap_width_ratio: float
            Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
            of size 256 yields an overlap of 51 pixels).
            Default to ``0.2``.
        match_iou_threshold: float
            Sliced predictions having higher iou than match_iou_threshold will be merged.
        visual_bbox_thickness: int
        visual_text_size: float
        visual_text_thickness: int
        verbose: int
            0: no print
            1: print slice/prediction durations, number of slices, model loading/file exporting durations
    """
    # for profiling
    durations_in_seconds = dict()

    # list image files in directory
    time_start = time.time()
    image_path_list = list_files(
        directory=image_dir, contains=[".jpg", ".jpeg", ".png"], verbose=verbose
    )
    time_end = time.time() - time_start
    durations_in_seconds["list_files"] = time_end

    # init model instance
    time_start = time.time()
    DetectionModel = import_class(model_name)
    detection_model = DetectionModel(
        model_path=model_parameters["model_path"],
        config_path=model_parameters["config_path"],
        prediction_score_threshold=model_parameters["prediction_score_threshold"],
        device=model_parameters["device"],
        category_mapping=model_parameters["category_mapping"],
        category_remapping=model_parameters["category_remapping"],
        load_at_init=False,
    )
    detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0
    for image_path in tqdm(image_path_list):
        # get filename
        (
            filename_with_extension,
            filename_without_extension,
        ) = get_base_filename(path=image_path)
        # load image
        image = read_image(image_path)

        # perform prediction
        if apply_sliced_prediction:
            # get sliced prediction
            prediction_result = get_sliced_prediction(
                image=image,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                match_iou_threshold=match_iou_threshold,
                verbose=verbose,
            )
            object_prediction_list = prediction_result["object_prediction_list"]
            durations_in_seconds["slice"] += prediction_result["durations_in_seconds"][
                "slice"
            ]
        else:
            # get full sized prediction
            prediction_result = get_prediction(
                image=image,
                detection_model=detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                merger=None,
                matcher=None,
                verbose=0,
            )
            object_prediction_list = prediction_result["object_prediction_list"]

        durations_in_seconds["prediction"] += prediction_result["durations_in_seconds"][
            "prediction"
        ]

        time_start = time.time()
        # export prediction boxes
        if crop_dir:
            crop_object_predictions(
                image=image,
                object_prediction_list=object_prediction_list,
                output_dir=crop_dir,
                file_name=filename_without_extension,
            )
        # export prediction list as pickle
        if pickle_dir:
            save_path = os.path.join(
                pickle_dir,
                filename_without_extension + ".pickle",
            )
            save_pickle(data=object_prediction_list, save_path=save_path)
        # export visualization
        if visual_output_dir:
            visualize_object_predictions(
                image,
                object_prediction_list=object_prediction_list,
                rect_th=visual_bbox_thickness,
                text_size=visual_text_size,
                text_th=visual_text_thickness,
                output_dir=visual_output_dir,
                file_name=filename_without_extension,
            )
        time_end = time.time() - time_start
        durations_in_seconds["export_files"] = time_end

    # print prediction duration
    if verbose == 1:
        print(
            "Model loaded in",
            durations_in_seconds["model_load"],
            "seconds.",
        )
        print(
            "Slicing performed in",
            durations_in_seconds["slice"],
            "seconds.",
        )
        print(
            "Prediction performed in",
            durations_in_seconds["prediction"],
            "seconds.",
        )
        print(
            "Exporting performed in",
            durations_in_seconds["export_files"],
            "seconds.",
        )
