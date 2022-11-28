# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2022.

from typing import List, Union

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from sahi.dataset import ImageDataset, collate_fn
from sahi.modelsv2.base import DetectionModel
from sahi.modelsv2.mmdet import MmdetDetectionModel
from sahi.pipelines.core import PredictionResult
from sahi.pipelines.merge import merge_prediction_results


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
        image_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn
    )

    prediction_results: List[PredictionResult] = []

    for batch in dataloader:
        batch_prediction_results: List[PredictionResult] = detection_model.predict(batch["images"])
        for ind, prediction_result in enumerate(batch_prediction_results):
            prediction_result.register_batch_info(
                image_id=batch["image_ids"][ind],
                offset_amount=batch["offset_amounts"][ind],
                full_shape=batch["full_shapes"][ind],
                image=batch["images"][ind] if image else None,
                filepath=batch["filepaths"][ind],
            )
        prediction_results.extend(batch_prediction_results)

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
        num_workers=0,
        verbose=1,
    )

    results[0].export_visuals("temp2", file_name="result0", export_format="png")
    results[1].export_visuals("temp2", file_name="result1", export_format="png")
