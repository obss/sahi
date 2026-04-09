"""RT-DETR detection model wrapper for SAHI.

Provides integration with Ultralytics RT-DETR real-time detection transformer models.
"""

from __future__ import annotations

from sahi.models.ultralytics import UltralyticsDetectionModel


class RTDetrDetectionModel(UltralyticsDetectionModel):
    """RT-DETR object detection model.

    Wraps Ultralytics RT-DETR for real-time detection inference.
    """

    def load_model(self) -> None:
        """Detection model is initialized and set to self.model."""
        from ultralytics import RTDETR

        try:
            model_source = self.model_path or "rtdetr-l.pt"
            model = RTDETR(model_source)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid rtdetr model path: ", e)
