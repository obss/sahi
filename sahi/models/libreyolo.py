"""LibreYOLO detection model wrapper for SAHI.

LibreYOLO (https://github.com/LibreYOLO/libreyolo) is an MIT-licensed library with an Ultralytics compatible API.
"""

from __future__ import annotations

from sahi.models.ultralytics import UltralyticsDetectionModel


class LibreYoloDetectionModel(UltralyticsDetectionModel):
    """LibreYOLO object detection model, reusing the Ultralytics wrapper."""

    def check_dependencies(self, packages: list[str] | None = None) -> None:
        """Check for libreyolo instead of ultralytics."""
        super().check_dependencies(packages=["libreyolo"])

    def load_model(self) -> None:
        """Detection model is initialized and set to self.model."""
        from libreyolo import LibreYOLO

        try:
            model = LibreYOLO(self.model_path or "LibreYOLO9t.pt", device=self.device, task=self.task)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid LibreYOLO model path: ", e)
