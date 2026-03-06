from __future__ import annotations

from sahi.models.ultralytics import UltralyticsDetectionModel


class YOLOEDetectionModel(UltralyticsDetectionModel):
    """YOLOE Detection Model for open-vocabulary detection and segmentation.

    YOLOE (Real-Time Seeing Anything) is a zero-shot, promptable YOLO model designed for
    open-vocabulary detection and segmentation. It supports text prompts, visual prompts,
    and prompt-free detection with internal vocabulary (1200+ categories).

    Key Features:
        - Open-vocabulary detection: Detect any object class via text prompts
        - Visual prompting: One-shot detection using reference images
        - Instance segmentation: Built-in segmentation for detected objects
        - Real-time performance: Maintains YOLO speed with no inference overhead
        - Prompt-free mode: Uses internal vocabulary for open-set recognition

    Available Models:
        Text/Visual Prompt models:
            - yoloe-11s-seg.pt, yoloe-11m-seg.pt, yoloe-11l-seg.pt
            - yoloe-v8s-seg.pt, yoloe-v8m-seg.pt, yoloe-v8l-seg.pt

        Prompt-free models:
            - yoloe-11s-seg-pf.pt, yoloe-11m-seg-pf.pt, yoloe-11l-seg-pf.pt
            - yoloe-v8s-seg-pf.pt, yoloe-v8m-seg-pf.pt, yoloe-v8l-seg-pf.pt

    !!! example "Usage Text Prompts"
        ```python
        from sahi import AutoDetectionModel

        # Load YOLOE model
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yoloe",
            model_path="yoloe-11l-seg.pt",
            confidence_threshold=0.3,
            device="cuda:0"
        )

        # Set text prompts for specific classes
        detection_model.model.set_classes(
            ["person", "car", "traffic light"],
            detection_model.model.get_text_pe(["person", "car", "traffic light"])
        )

        # Perform prediction
        from sahi.predict import get_prediction
        result = get_prediction("image.jpg", detection_model)
        ```

    !!! example "Usage for standard detection (no prompts)"
        ```python
        from sahi import AutoDetectionModel

        # Load YOLOE model (works like standard YOLO)
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yoloe",
            model_path="yoloe-11l-seg.pt",
            confidence_threshold=0.3,
            device="cuda:0"
        )

        # Perform prediction without prompts (uses internal vocabulary)
        from sahi.predict import get_sliced_prediction
        result = get_sliced_prediction(
            "image.jpg",
            detection_model,
            slice_height=512,
            slice_width=512,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        ```

    Note:
        - YOLOE models perform instance segmentation by default
        - When used without prompts, YOLOE performs like standard YOLO11 with identical speed
        - For visual prompting, see Ultralytics YOLOE documentation
        - YOLOE achieves +3.5 AP over YOLO-Worldv2 on LVIS with 1.4x faster inference

    References:
        - Paper: https://arxiv.org/abs/2503.07465
        - Docs: https://docs.ultralytics.com/models/yoloe/
        - GitHub: https://github.com/THU-MIG/yoloe
    """

    def load_model(self):
        """Loads the YOLOE detection model from the specified path.

        Initializes the YOLOE model with the given model path or uses the default
        'yoloe-11s-seg.pt' if no path is provided. The model is then moved to the
        specified device (CPU/GPU).

        By default, YOLOE works in prompt-free mode using its internal vocabulary
        of 1200+ categories. To use text prompts for specific classes, call
        model.set_classes() after loading:

            model.set_classes(["person", "car"], model.get_text_pe(["person", "car"]))

        Raises:
            TypeError: If the model_path is not a valid YOLOE model path or if
                      the ultralytics package with YOLOE support is not installed.
        """
        from ultralytics import YOLOE

        try:
            model_source = self.model_path or "yoloe-11s-seg.pt"
            model = YOLOE(model_source)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError(f"model_path is not a valid YOLOE model path: {e}") from e
