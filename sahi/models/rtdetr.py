# OBSS SAHI Tool
# Code written by AnNT, 2023.

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements
from sahi.models.yolov8 import Yolov8DetectionModel

class RTDetrDetectionModel(Yolov8DetectionModel):
    def check_dependencies(self) -> None:
        check_requirements(["ultralytics"])

    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        from ultralytics import RTDETR

        try:
            model = RTDETR(self.model_path)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid rtdet model path: ", e)