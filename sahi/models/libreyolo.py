"""LibreYOLO detection model wrapper for SAHI.

Provides integration with LibreYOLO (MIT-licensed) object detection models.
Supports detection, segmentation, and oriented bounding box (OBB) tasks.

LibreYOLO GitHub: https://github.com/LibreYOLO/libreyolo
"""

from __future__ import annotations

import numpy as np

from sahi.models.ultralytics import UltralyticsDetectionModel


class LibreYoloDetectionModel(UltralyticsDetectionModel):
    """LibreYOLO object detection model.

    Wraps LibreYOLO models for sliced inference via SAHI.
    LibreYOLO provides an MIT-licensed alternative to Ultralytics
    with a compatible API (Results, Boxes, Masks, OBB).
    """

    def check_dependencies(self, packages: list[str] | None = None) -> None:
        """Check for libreyolo instead of ultralytics."""
        super().check_dependencies(packages=["libreyolo"])

    def load_model(self) -> None:
        """Initialize the LibreYOLO model and assign it to self.model."""
        from libreyolo import LibreYOLO

        try:
            model_source = self.model_path or "LibreYOLO9t.pt"
            model = LibreYOLO(model_source, device=self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError("model_path is not a valid LibreYOLO model path: ", e)

    def perform_batch_inference(self, images: list[np.ndarray]) -> None:
        """Perform batch inference, without the 'cfg' kwarg that LibreYOLO rejects."""
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        kwargs = {"verbose": False, "conf": self.confidence_threshold, "device": self.device}

        if self.image_size is not None:
            kwargs = {"imgsz": self.image_size, **kwargs}

        images_bgr = [img[:, :, ::-1] for img in images]
        prediction_result = self.model(images_bgr, **kwargs)

        self._original_predictions = self._extract_predictions(prediction_result)
        self._original_shapes = [img.shape for img in images]

    def _extract_predictions(self, prediction_result: list) -> list:
        """Extract predictions using libreyolo.Masks instead of ultralytics.Masks."""
        import torch

        if self.has_mask:
            from libreyolo import Masks

            for result in prediction_result:
                if not result.masks:
                    device = getattr(self.model, "device", "cpu")
                    result.masks = Masks(torch.tensor([], device=device), result.boxes.orig_shape)

            return [(result.boxes.data, result.masks.data) for result in prediction_result]
        elif self.is_obb:
            device = getattr(self.model, "device", "cpu")
            return [
                (
                    torch.cat(
                        [
                            result.obb.xyxy,
                            result.obb.conf.unsqueeze(-1),
                            result.obb.cls.unsqueeze(-1),
                        ],
                        dim=1,
                    )
                    if result.obb is not None
                    else torch.empty((0, 6), device=device),
                    result.obb.xyxyxyxy if result.obb is not None else torch.empty((0, 4, 2), device=device),
                )
                for result in prediction_result
            ]
        else:
            return [result.boxes.data for result in prediction_result]
