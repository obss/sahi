from typing import Dict, Optional

from sahi.utils.file import import_model_class
from sahi.utils.import_utils import check_requirements

MODEL_TYPE_TO_MODEL_CLASS_NAME = {
    "mmdet": "MmdetDetectionModel",
    "yolov5": "Yolov5DetectionModel",
    "detectron2": "Detectron2DetectionModel",
    "huggingface": "HuggingfaceDetectionModel",
    "torchvision": "TorchVisionDetectionModel",
    "yolov7": "Yolov7DetectionModel",
}


class AutoDetectionModel:
    @staticmethod
    def from_pretrained(
        model_type: str,
        model_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
        **kwargs,
    ):
        """
        Loads a DetectionModel from given path.

        Args:
            model_type: str
                Name of the detection framework (example: "yolov5", "mmdet", "detectron2")
            model_path: str
                Path of the Layer model (ex. '/sahi/yolo/models/yolov5')
            config_path: str
                Path of the config file (ex. 'mmdet/configs/cascade_rcnn_r50_fpn_1x.py')
            device: str
                Device, "cpu" or "cuda:0"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        Returns:
            Returns an instance of a DetectionModel
        Raises:
            ImportError: If given {model_type} framework is not installed
        """

        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
        DetectionModel = import_model_class(model_class_name)

        return DetectionModel(
            model_path=model_path,
            config_path=config_path,
            device=device,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            load_at_init=load_at_init,
            image_size=image_size,
            **kwargs,
        )

    @staticmethod
    def from_layer(
        model_path: str,
        no_cache: bool = False,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        image_size: int = None,
    ):
        """
        Loads a DetectionModel from Layer. You can pass additional parameters in the name to retrieve a specific version
        of the model with format: ``model_path:major_version.minor_version``
        By default, this function caches models locally when possible.
        Args:
            model_path: str
                Path of the Layer model (ex. '/sahi/yolo/models/yolov5')
            no_cache: bool
                If True, force model fetch from the remote location.
            device: str
                Device, "cpu" or "cuda:0"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            image_size: int
                Inference input size.
        Returns:
            Returns an instance of a DetectionModel
        Raises:
            ImportError: If Layer is not installed in your environment
            ValueError: If model path does not match expected pattern: organization_name/project_name/models/model_name
        """
        check_requirements(["layer"])

        import layer

        layer_model = layer.get_model(name=model_path, no_cache=no_cache).get_train()
        if layer_model.__class__.__module__ in ["yolov5.models.common", "models.common"]:
            model_type = "yolov5"
        else:
            raise Exception(f"Unsupported model: {type(layer_model)}. Only YOLOv5 models are supported.")

        model_class_name = MODEL_TYPE_TO_MODEL_CLASS_NAME[model_type]
        DetectionModel = import_model_class(model_class_name)

        return DetectionModel(
            model=layer_model,
            device=device,
            mask_threshold=mask_threshold,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            category_remapping=category_remapping,
            image_size=image_size,
        )
