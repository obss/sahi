"""YOLOE detection model wrapper for SAHI.

Provides integration with YOLOE (Real-Time Seeing Anything) open-vocabulary
detection and segmentation models.
"""

from sahi.models.ultralytics import UltralyticsDetectionModel


class YOLOEDetectionModel(UltralyticsDetectionModel):
    """YOLOE Detection Model for open-vocabulary detection and segmentation.

    YOLOE (Real-Time Seeing Anything) is a zero-shot, promptable YOLO model designed for
    open-vocabulary detection and segmentation. It supports text prompts, visual prompts,
    and prompt-free detection with a built-in vocabulary (the -pf models).

    Key Features:
        - Open-vocabulary detection: Detect any object class via text prompts
        - Visual prompting: One-shot detection using reference images
        - Instance segmentation: Built-in segmentation for detected objects
        - Real-time performance: Maintains YOLO speed with no inference overhead
        - Prompt-free mode: the -pf models detect from a built-in vocabulary without prompts

    Available Models:
        Text/Visual Prompt models:
            - yoloe-26n-seg.pt, yoloe-26s-seg.pt, yoloe-26m-seg.pt, yoloe-26l-seg.pt, yoloe-26x-seg.pt
            - yoloe-11s-seg.pt, yoloe-11m-seg.pt, yoloe-11l-seg.pt
            - yoloe-v8s-seg.pt, yoloe-v8m-seg.pt, yoloe-v8l-seg.pt

        Prompt-free models:
            - yoloe-26n-seg-pf.pt, yoloe-26s-seg-pf.pt, yoloe-26m-seg-pf.pt, yoloe-26l-seg-pf.pt, yoloe-26x-seg-pf.pt
            - yoloe-11s-seg-pf.pt, yoloe-11m-seg-pf.pt, yoloe-11l-seg-pf.pt
            - yoloe-v8s-seg-pf.pt, yoloe-v8m-seg-pf.pt, yoloe-v8l-seg-pf.pt

    !!! example "Usage Text Prompts"
        ```python
        from sahi import AutoDetectionModel

        # Load YOLOE model
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yoloe",
            model_path="yoloe-26l-seg.pt",
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

    !!! example "Usage without prompts (prompt-free model)"
        ```python
        from sahi import AutoDetectionModel

        # Load a prompt-free YOLOE model, the -seg models need set_classes first
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="yoloe",
            model_path="yoloe-26l-seg-pf.pt",
            confidence_threshold=0.3,
            device="cuda:0"
        )

        # Perform prediction without prompts (uses the built-in vocabulary)
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
        - When used without prompts, YOLOE runs at the same speed as the YOLO model it is built on
        - For visual prompting, see Ultralytics YOLOE documentation
        - YOLOE achieves +3.5 AP over YOLO-Worldv2 on LVIS with 1.4x faster inference

    References:
        - Paper: https://arxiv.org/abs/2503.07465
        - Docs: https://docs.ultralytics.com/models/yoloe/
        - GitHub: https://github.com/THU-MIG/yoloe
    """

    def load_model(self) -> None:
        """Loads the YOLOE detection model from the specified path.

        Initializes the YOLOE model with the given model path or uses the default
        'yoloe-26s-seg.pt' if no path is provided. The model is then moved to the
        specified device (CPU/GPU).

        Text prompt models such as the default yoloe-26s-seg.pt detect nothing until
        classes are set, while the -pf models detect from their built-in vocabulary
        without prompts. To use text prompts for specific classes, call
        model.set_classes() after loading:

            model.set_classes(["person", "car"], model.get_text_pe(["person", "car"]))

        Raises:
            TypeError: If the model_path is not a valid YOLOE model path or if
                      the ultralytics package with YOLOE support is not installed.
        """
        from ultralytics import YOLOE

        try:
            model_source = self.model_path or "yoloe-26s-seg.pt"
            model = YOLOE(model_source)
            model.to(self.device)
            self.set_model(model)
        except Exception as e:
            raise TypeError(f"model_path is not a valid YOLOE model path: {e}") from e
