# OBSS SAHI Tool
# Code written by AnNT, 2023.

import logging

from sahi.models.yolov8 import Yolov8DetectionModel
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)


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
            model.to(self.device)

            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid rtdet model path: ", e)
