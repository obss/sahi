# OBSS SAHI Tool
# Code written by AnNT, 2023.

from sahi.models.ultralytics import UltralyticsDetectionModel


class RTDetrDetectionModel(UltralyticsDetectionModel):
    def __init__(self, *args, **kwargs):
        self.required_packages = list(getattr(self, "required_packages", [])) + ["ultralytics"]
        super().__init__(*args, **kwargs)

    def load_model(self):
        """Detection model is initialized and set to self.model."""
        from ultralytics import RTDETR

        try:
            model_source = self.model_path or "rtdetr-l.pt"
            model = RTDETR(model_source)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid rtdet model path: ", e)
