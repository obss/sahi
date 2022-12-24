# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2022.

import time
from collections import defaultdict
from multiprocessing import Process, Queue
from typing import List, Union

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from sahi.dataset import ImageDataset, collate_fn
from sahi.models.mmdet import MmdetDetectionModel  # todo: update to sahi.modelsv2
from sahi.modelsv2.base import DetectionModel
from sahi.pipelines.core import PredictionResult
from sahi.pipelines.merge import merge_prediction_results


def merge_prediction_results(merge_queue, result_queue, max_counter):
    image_id_to_predictions_results = defaultdict(list)
    prediction_results = []
    object_predictions = []
    counter = 0
    while counter < (max_counter - 1):
        print(counter, max_counter - 1)
        try:
            # Try to get next element from queue
            args = merge_queue.get()
            new_image_id = args["image_id"]
            sliced_object_predictions = args["object_predictions_per_image"]
            image_path = args["image_path"]
            confidence_threshold = args["confidence_threshold"]
            postprocess = args["postprocess"]

            if new_image_id != image_id:
                # postprocess matching predictions
                if postprocess is not None:
                    object_predictions = postprocess(object_predictions)

                prediction_result = PredictionResult(
                    image=image_path,
                    object_predictions=object_predictions,
                    durations_in_seconds=None,
                )
                prediction_results.append(prediction_result)
                object_predictions = []
            image_id = new_image_id

            # filter out predictions with lower score
            sliced_object_predictions = [
                object_prediction
                for object_prediction in sliced_object_predictions
                if object_prediction.score.value > confidence_threshold
            ]

            # append slice predictions
            object_predictions.extend(sliced_object_predictions)
            print("len_object_predictions: ", len(object_predictions))
            print("len_prediction_results: ", len(prediction_results))

            counter += 1
        except:
            # Wait if queue is empty
            time.sleep(0.01)  # queue is either empty or no update

    result_queue.put(prediction_results)


def worker_init_fn(_):
    """
    This function is used to initialize the worker processes in the DataLoader.
    """
    import torch

    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    data_slice = slice(worker_id, len(dataset._samples) + 1, worker_info.num_workers)

    dataset._samples = dataset._samples[data_slice]
    return None


def simple_detection_pipeline(
    detection_model: DetectionModel,
    image: Union[Image.Image, str, np.ndarray] = None,
    image_folder_dir: str = None,
    include_slices: bool = False,
    include_original: bool = True,
    slice_size: int = 512,
    slice_overlap_ratio: int = 0.2,
    postprocess_type: str = "batched_nms",
    postprocess_iou_threshold: float = 0.5,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: int = 0,
) -> List[PredictionResult]:
    target_image_format = "bgr" if isinstance(detection_model, MmdetDetectionModel) else "rgb"

    image_dataset = ImageDataset(
        image=image,
        image_folder_dir=image_folder_dir,
        slice_height=slice_size,
        slice_width=slice_size,
        slice_overlap_height_ratio=slice_overlap_ratio,
        slice_overlap_width_ratio=slice_overlap_ratio,
        include_slices=include_slices,
        include_original=include_original,
        verbose=verbose,
        target_image_format=target_image_format,
    )

    dataloader = DataLoader(
        image_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
    )

    merge_prediction_results_queue = Queue()
    result_queue = Queue()
    p = Process(
        target=merge_prediction_results, args=(merge_prediction_results_queue, result_queue, len(image_dataset))
    )
    p.start()

    prediction_results: List[PredictionResult] = []

    for batch in dataloader:
        batch_prediction_results: List[PredictionResult] = detection_model.predict(batch["images"])
        for ind, prediction_result in enumerate(batch_prediction_results):
            prediction_result.image_id = batch["image_ids"][ind]
            prediction_result.offset_amount = batch["offset_amounts"][ind]
            prediction_result.full_shape = batch["full_shapes"][ind]
            prediction_result.filepath = batch["filepaths"][ind]
            if include_slices:
                merge_prediction_results_queue.put(
                    {
                        "image_id": batch["image_ids"][ind],
                        "offset_amount": batch["offset_amounts"][ind],
                        "full_shape": batch["full_shapes"][ind],
                        "filepath": batch["filepaths"][ind],
                    }
                )

    if include_slices:
        prediction_results = merge_prediction_results(
            prediction_results=prediction_results,
            postprocess_type=postprocess_type,
            postprocess_iou_threshold=postprocess_iou_threshold,
        )

    return prediction_results


if __name__ == "__main__":
    # load model
    from sahi.modelsv2 import HuggingfaceDetectionModel
    from sahi.utils.huggingface import HuggingfaceTestConstants

    huggingface_detection_model = HuggingfaceDetectionModel(
        model_path=HuggingfaceTestConstants.YOLOS_TINY_MODEL_PATH,
        confidence_threshold=0.1,
        device="cuda:0",
        load_at_init=True,
        image_size=640,
    )

    results = simple_detection_pipeline(
        # image="demo/demo_data/small-vehicles1.jpeg",
        image_folder_dir="demo/demo_data",
        detection_model=huggingface_detection_model,
        include_slices=True,
        include_original=True,
        slice_size=512,
        slice_overlap_ratio=0.2,
        postprocess_type="batched_nms",
        postprocess_iou_threshold=0.4,
        batch_size=4,
        num_workers=2,
        verbose=1,
    )

    results[0].export_visuals("temp2", file_name="result0", export_format="png")
    results[1].export_visuals("temp2", file_name="result1", export_format="png")
