from __future__ import annotations

from sahi.models.ultralytics import UltralyticsDetectionModel


class YOLOWorldDetectionModel(UltralyticsDetectionModel):
    def load_model(self):
        """Detection model is initialized and set to self.model."""

        from ultralytics import YOLOWorld

        try:
            model_source = self.model_path or "yolov8s-worldv2.pt"
            model = YOLOWorld(model_source)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid yolo world model path: ", e)
