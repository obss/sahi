# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import os
import time

from tqdm import tqdm
from typing import Dict, Optional, List

from sahi.prediction import ObjectPrediction
from sahi.postprocess.combine import UnionMergePostprocess, PostprocessPredictions, NMSPostprocess
from sahi.slicing import slice_image
from sahi.utils.coco import Coco, CocoImage
from sahi.utils.cv import (
    crop_object_predictions,
    read_image,
    visualize_object_predictions,
)
from sahi.utils.file import (
    Path,
    import_class,
    increment_path,
    list_files,
    save_json,
    save_pickle,
)


def get_prediction(
    image,
    detection_model,
    shift_amount: list = [0, 0],
    full_shape=None,
    postprocess: Optional[PostprocessPredictions] = None,
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
        postprocess: sahi.postprocess.combine.PostprocessPredictions
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
    object_prediction_list: List[ObjectPrediction] = detection_model.object_prediction_list
    # filter out predictions with lower score
    filtered_object_prediction_list = [
        object_prediction
        for object_prediction in object_prediction_list
        if object_prediction.score.value > detection_model.prediction_score_threshold
    ]
    # postprocess matching predictions
    if postprocess is not None:
        filtered_object_prediction_list = postprocess(filtered_object_prediction_list)
    else:
        # init match merge instances
        postprocess = UnionMergePostprocess(match_threshold=0.9, match_metric="IOS", class_agnostic=True)
        # postprocess matching predictions
        filtered_object_prediction_list = postprocess(filtered_object_prediction_list)

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
    postprocess_type: str = "UNIONMERGE",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
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
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'UNIONMERGE' or 'NMS'. Default is 'UNIONMERGE'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
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
    slice_image_result = slice_image(
        image=image,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    num_slices = len(slice_image_result)
    time_end = time.time() - time_start
    durations_in_seconds["slice"] = time_end

    # init match postprocess instance
    if postprocess_type == "UNIONMERGE":
        postprocess = UnionMergePostprocess(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )
    elif postprocess_type == "NMS":
        postprocess = NMSPostprocess(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )
    else:
        raise ValueError(f"postprocess_type should be one of ['UNIOUNMERGE', 'NMS'] but given as {postprocess_type}")

    # create prediction input
    num_group = int(num_slices / num_batch)
    if verbose == 1 or verbose == 2:
        if num_slices > 0:
            print("Number of slices:", num_slices)
        else:
            print("Number of slices:", 1)
    object_prediction_list = []
    if num_slices > 0:  # if zero_frac < max_allowed_zeros_ratio from slice_image
        for group_ind in range(num_group):
            # prepare batch (currently supports only 1 batch)
            image_list = []
            shift_amount_list = []
            for image_ind in range(num_batch):
                image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
                shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
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
    else:  # if zero_frac >= max_allowed_zeros_ratio from slice_image
        prediction_result = get_prediction(
            image=image,
            detection_model=detection_model,
            shift_amount=[0, 0],
            full_shape=None,
            match_threshold=None,
            verbose=0,
        )
        object_prediction_list.extend(prediction_result["object_prediction_list"])

    # remove empty predictions
    object_prediction_list = [object_prediction for object_prediction in object_prediction_list if object_prediction]
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
    full_object_prediction_list = postprocess(full_object_prediction_list)
    # return filtered elements of filtered full_object_prediction_list
    return {
        "object_prediction_list": full_object_prediction_list,
        "durations_in_seconds": durations_in_seconds,
    }


def predict(
    model_name: str = "MmdetDetectionModel",
    model_parameters: Dict = None,
    source: str = None,
    apply_sliced_prediction: bool = True,
    slice_height: int = 256,
    slice_width: int = 256,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    postprocess_type: str = "UNIONMERGE",
    postprocess_match_metric: str = "IOS",
    postprocess_match_threshold: float = 0.5,
    postprocess_class_agnostic: bool = False,
    export_visual: bool = True,
    export_pickle: bool = False,
    export_crop: bool = False,
    coco_file_path: bool = None,
    project: str = "runs/predict",
    name: str = "exp",
    visual_bbox_thickness: int = 1,
    visual_text_size: float = 0.3,
    visual_text_thickness: int = 1,
    visual_export_format: str = "png",
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
        source: str
            Folder directory that contains images or path of the image to be predicted.
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
        postprocess_type: str
            Type of the postprocess to be used after sliced inference while merging/eliminating predictions.
            Options are 'UNIONMERGE' or 'NMS'. Default is 'UNIONMERGE'.
        postprocess_match_metric: str
            Metric to be used during object prediction matching after sliced prediction.
            'IOU' for intersection over union, 'IOS' for intersection over smaller area.
        postprocess_match_threshold: float
            Sliced predictions having higher iou than postprocess_match_threshold will be
            postprocessed after sliced prediction.
        postprocess_class_agnostic: bool
            If True, postprocess will ignore category ids.
        export_pickle: bool
            Export predictions as .pickle
        export_crop: bool
            Export predictions as cropped images.
        coco_file_path: str
            If coco file path is provided, detection results will be exported in coco json format.
        project: str
            Save results to project/name.
        name: str
            Save results to project/name.
        visual_bbox_thickness: int
        visual_text_size: float
        visual_text_thickness: int
        visual_export_format: str
            Can be specified as 'jpg' or 'png'
        verbose: int
            0: no print
            1: print slice/prediction durations, number of slices, model loading/file exporting durations
    """
    # for profiling
    durations_in_seconds = dict()

    # list image files in directory
    if coco_file_path:
        coco: Coco = Coco.from_coco_dict_or_path(coco_file_path)
        image_path_list = [str(Path(source) / Path(coco_image.file_name)) for coco_image in coco.images]
        coco_json = []
    elif os.path.isdir(source):
        time_start = time.time()
        image_path_list = list_files(
            directory=source,
            contains=[".jpg", ".jpeg", ".png"],
            verbose=verbose,
        )
        time_end = time.time() - time_start
        durations_in_seconds["list_files"] = time_end
    else:
        image_path_list = [source]
        durations_in_seconds["list_files"] = 0

    # init export directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
    crop_dir = save_dir / "crops"
    visual_dir = save_dir / "visuals"
    visual_with_gt_dir = save_dir / "visuals_with_gt"
    pickle_dir = save_dir / "pickles"
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # init model instance
    time_start = time.time()
    DetectionModel = import_class(model_name)
    detection_model = DetectionModel(
        model_path=model_parameters["model_path"],
        config_path=model_parameters.get("config_path", None),
        prediction_score_threshold=model_parameters.get("prediction_score_threshold", 0.25),
        device=model_parameters.get("device", None),
        category_mapping=model_parameters.get("category_mapping", None),
        category_remapping=model_parameters.get("category_remapping", None),
        load_at_init=False,
    )
    detection_model.load_model()
    time_end = time.time() - time_start
    durations_in_seconds["model_load"] = time_end

    # iterate over source images
    durations_in_seconds["prediction"] = 0
    durations_in_seconds["slice"] = 0
    for ind, image_path in enumerate(tqdm(image_path_list)):
        # get filename
        if os.path.isdir(source):  # preserve source folder structure in export
            relative_filepath = image_path.split(source)[-1]
            relative_filepath = relative_filepath[1:] if relative_filepath[0] == os.sep else relative_filepath
        else:  # no process if source is single file
            relative_filepath = image_path
        filename_without_extension = Path(relative_filepath).stem
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
                postprocess_type=postprocess_type,
                postprocess_match_metric=postprocess_match_metric,
                postprocess_match_threshold=postprocess_match_threshold,
                postprocess_class_agnostic=postprocess_class_agnostic,
                verbose=verbose,
            )
            object_prediction_list = prediction_result["object_prediction_list"]
            durations_in_seconds["slice"] += prediction_result["durations_in_seconds"]["slice"]
        else:
            # get full sized prediction
            prediction_result = get_prediction(
                image=image,
                detection_model=detection_model,
                shift_amount=[0, 0],
                full_shape=None,
                postprocess=None,
                verbose=0,
            )
            object_prediction_list = prediction_result["object_prediction_list"]

        durations_in_seconds["prediction"] += prediction_result["durations_in_seconds"]["prediction"]

        if coco_file_path:
            image_id = ind + 1
            # append predictions in coco format
            for object_prediction in object_prediction_list:
                coco_prediction = object_prediction.to_coco_prediction()
                coco_prediction.image_id = image_id
                coco_prediction_json = coco_prediction.json
                coco_json.append(coco_prediction_json)
            # convert ground truth annotations to object_prediction_list
            coco_image: CocoImage = coco.images[ind]
            object_prediction_gt_list: List[ObjectPrediction] = []
            for coco_annotation in coco_image.annotations:
                coco_annotation_dict = coco_annotation.json
                category_name = coco_annotation.category_name
                full_shape = [coco_image.height, coco_image.width]
                object_prediction_gt = ObjectPrediction.from_coco_annotation_dict(
                    annotation_dict=coco_annotation_dict, category_name=category_name, full_shape=full_shape
                )
                object_prediction_gt_list.append(object_prediction_gt)
            # export visualizations with ground truths
            output_dir = str(visual_with_gt_dir / Path(relative_filepath).parent)
            color = (0, 255, 0)  # original annotations in green
            result = visualize_object_predictions(
                image,
                object_prediction_list=object_prediction_gt_list,
                rect_th=visual_bbox_thickness,
                text_size=visual_text_size,
                text_th=visual_text_thickness,
                color=color,
                output_dir=None,
                file_name=None,
                export_format=None,
            )
            color = (255, 0, 0)  # model predictions in red
            _ = visualize_object_predictions(
                result["image"],
                object_prediction_list=object_prediction_list,
                rect_th=visual_bbox_thickness,
                text_size=visual_text_size,
                text_th=visual_text_thickness,
                color=color,
                output_dir=output_dir,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )

        time_start = time.time()
        # export prediction boxes
        if export_crop:
            output_dir = str(crop_dir / Path(relative_filepath).parent)
            crop_object_predictions(
                image=image,
                object_prediction_list=object_prediction_list,
                output_dir=output_dir,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )
        # export prediction list as pickle
        if export_pickle:
            save_path = str(pickle_dir / Path(relative_filepath).parent / (filename_without_extension + ".pickle"))
            save_pickle(data=object_prediction_list, save_path=save_path)
        # export visualization
        if export_visual:
            output_dir = str(visual_dir / Path(relative_filepath).parent)
            visualize_object_predictions(
                image,
                object_prediction_list=object_prediction_list,
                rect_th=visual_bbox_thickness,
                text_size=visual_text_size,
                text_th=visual_text_thickness,
                output_dir=output_dir,
                file_name=filename_without_extension,
                export_format=visual_export_format,
            )
        time_end = time.time() - time_start
        durations_in_seconds["export_files"] = time_end

    # export coco results
    if coco_file_path:
        save_path = str(save_dir / "result.json")
        save_json(coco_json, save_path)

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
        if export_visual:
            print(
                "Exporting performed in",
                durations_in_seconds["export_files"],
                "seconds.",
            )
