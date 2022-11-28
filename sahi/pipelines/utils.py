# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2022.

import copy
import time
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from sahi.utils.cv import Colors, apply_color_mask


def visualize_predictions(
    image: np.ndarray,
    bboxes: "BoundingBoxes",
    masks: "Masks" = None,
    label_id_to_name: Dict[int, str] = None,
    output_dir: str = None,
    file_name: str = "prediction_visual",
    export_format: str = "png",
    color: tuple = None,
    rect_th: int = None,
    text_size: float = None,
    text_th: float = None,
):

    elapsed_time = time.time()
    # deepcopy image so that original is not altered
    image = copy.deepcopy(image)
    # select predefined classwise color palette if not specified
    if color is None:
        colors = Colors()
    else:
        colors = None
    # set rect_th for boxes
    rect_th = rect_th or max(round(sum(image.shape) / 2 * 0.001), 1)
    # set text_th for category names
    text_th = text_th or max(rect_th - 1, 1)
    # set text_size for category names
    text_size = text_size or rect_th / 3
    # add bbox and mask to image if present
    xyxysl = bboxes.to_xyxysl()
    for ind in range(len(bboxes)):
        bbox = xyxysl[ind, :4]
        score = xyxysl[ind, 4]
        label_id = xyxysl[ind, 5]
        label_name = label_id_to_name[label_id] if label_id_to_name else str(int(label_id))

        # set color
        if colors is not None:
            color = colors(label_id)
        # visualize masks if present
        if masks is not None:
            # deepcopy mask so that original is not altered
            mask = copy.deepcopy(np.array(masks[ind]))
            # draw mask
            rgb_mask = apply_color_mask(mask, color)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.4, 0)
        # set bbox points
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        # visualize boxes
        cv2.rectangle(
            image,
            p1,
            p2,
            color=color,
            thickness=rect_th,
        )
        # arange bounding box text location
        label = f"{label_name} {score:.2f}"
        w, h = cv2.getTextSize(label, 0, fontScale=text_size, thickness=text_th)[0]  # label width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # add bounding box text
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
            0,
            text_size,
            (255, 255, 255),
            thickness=text_th,
        )

    # export if output_dir is present
    if output_dir is not None:
        # export image with predictions
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        # save inference result
        save_path = str(Path(output_dir) / (file_name + "." + export_format))
        cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    elapsed_time = time.time() - elapsed_time
    return {"image": image, "elapsed_time": elapsed_time}
