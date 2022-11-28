# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2022.

from typing import List

from sahi.pipelines.core import PredictionResult


def merge_prediction_results(
    prediction_results: List[PredictionResult],
    postprocess_type: str = "batched_nms",
    postprocess_iou_threshold: float = 0.5,
    inplace: bool = False,
) -> List[PredictionResult]:
    merged_prediction_results: List[PredictionResult] = []
    base_prediction_result = None
    for prediction_result in prediction_results:
        # assign new base prediction result
        if base_prediction_result is None:
            base_prediction_result = prediction_result.remap()
        else:
            # combine predictions results having the same image id
            if base_prediction_result.image_id == prediction_result.image_id:
                base_prediction_result += prediction_result.remap()
            # perform postprocessing if new image id is encountered
            else:
                base_prediction_result = base_prediction_result.postprocess(
                    type=postprocess_type, inplace=inplace, iou_threshold=postprocess_iou_threshold
                )
                merged_prediction_results.append(base_prediction_result)
                # set base_prediction_result to none to indicate that a new image id has been encountered
                base_prediction_result = None
    # perform postprocessing on the last prediction result
    base_prediction_result = base_prediction_result.postprocess(
        type=postprocess_type, inplace=inplace, iou_threshold=postprocess_iou_threshold
    )
    merged_prediction_results.append(base_prediction_result)

    return merged_prediction_results
