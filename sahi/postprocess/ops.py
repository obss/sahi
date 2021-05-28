# OBSS SAHI Tool
# Code written by Cemil Cengiz, 2020.
# Modified by Fatih C Akyon, 2020.

import numpy as np
from sahi.prediction import ObjectPrediction
from typing import List

BoxArray = np.ndarray # x1 y1 x2 y2


def calculate_area(box: List[int]) -> float:
    """
    Args:
        box (List[int]): [x1, y1, x2, y2]
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection_area(box1: BoxArray, box2: BoxArray) -> float:
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    width_height = (right_bottom - left_top).clip(min=0)
    return width_height[0] * width_height[1]


def have_same_class(pred1: ObjectPrediction, pred2: ObjectPrediction) -> bool:
    return pred1.category.id == pred2.category.id


def box_iou(box1: List[int], box2: List[int]) -> float:
    """ Returns the ratio of intersection area to the union 
    Args:
        box1 (List[int]): [x1, y1, x2, y2]
        box2 (List[int]): [x1, y1, x2, y2]
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    intersect = intersection_area(box1, box2)
    return intersect / (area1 + area2 - intersect)


def box_ios(box1: List[int], box2: List[int]) -> float:
    """ Returns the ratio of intersection area to the smaller box's area
    Args:
        box1 (List[int]): [x1, y1, x2, y2]
        box2 (List[int]): [x1, y1, x2, y2]
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    area1 = calculate_area(box1)
    area2 = calculate_area(box2)
    intersect = intersection_area(box1, box2)
    smaller_area = np.minimum(area1, area2)
    return intersect / smaller_area


def box_union(box1: List[int], box2: List[int]) -> BoxArray:
    """
        Args:
        box1 (List[int]): [x1, y1, x2, y2]
        box2 (List[int]): [x1, y1, x2, y2]
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    left_top = np.minimum(box1[:2], box2[:2])
    right_bottom = np.maximum(box1[2:], box2[2:])
    return np.concatenate((left_top, right_bottom))


def box_intersection(box1: List[int], box2: List[int]) -> BoxArray:
    """
        Args:
        box1 (List[int]): [x1, y1, x2, y2]
        box2 (List[int]): [x1, y1, x2, y2]
    """
    box1 = np.array(box1)
    box2 = np.array(box2)
    left_top = np.maximum(box1[:2], box2[:2])
    right_bottom = np.minimum(box1[2:], box2[2:])
    return np.concatenate((left_top, right_bottom))
