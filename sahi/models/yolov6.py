# OBSS SAHI Tool YOLOv6 extension
# Code written by Ho Wing Yip, 2024.

from typing import Any, Dict, List, Optional, Union, Iterable

import numpy as np
import math
import torch

# Have yolov6 somewhere in sys.path first
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.core.inferer import Inferer

from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import is_available
from sahi.utils.torch import select_device as select_torch_device


def check_img_size(img_size, s=32, floor=0):
    def make_divisible( x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor
    """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
    if isinstance(img_size, int):  # integer i.e. img_size=640
        new_size = max(make_divisible(img_size, int(s)), floor)
    elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
        new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
    else:
        raise Exception(f"Unsupported type of img_size: {type(img_size)}")

    if new_size != img_size:
        print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
    return new_size if isinstance(img_size,list) else [new_size]*2

class Yolov6DetectionModel:
    def __init__(
        self,
        model_path: Optional[str] = None,
        sliced_model_path: Optional[str] = None,
        model: Optional[Any] = None,
        device: Optional[str] = None,
        nms_confidence_threshold: float = 0.01,
        iou_threshold: float = 0.3,
        filter_confidence_threshold: float = 0.25,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        half: bool = True,
        full_image_size: Union[Iterable, int] = None,
        sliced_image_size: Union[Iterable, int] = None,
    ):
        """
        Init object detection/instance segmentation model.
        Args:
            model_path: str
                Path for the instance segmentation model weight
            device: str
                Torch device, "cpu" or "cuda"
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
        """
        self.model_path = model_path
        self.sliced_model_path = sliced_model_path
        self.model = None
        self.device = device
        self.nms_confidence_threshold = nms_confidence_threshold
        self.iou_threshold = iou_threshold
        self.filter_confidence_threshold = filter_confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.full_image_size = full_image_size
        self.sliced_image_size = sliced_image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None

        self.half = half

        self.set_device()

        # automatically load model if load_at_init is True
        if load_at_init:
            if model:
                self.set_model(model)
            else:
                self.load_model()

    def check_dependencies(self) -> None:
        """
        This function can be implemented to ensure model dependencies are installed.
        """
        pass

    def fix_model_dtype(self):
        if self.half and self.device != "cpu":
            self.model.model.half()
            self.sliced_model.model.half()
        else:
            self.model.model.float()
            self.sliced_model.model.float()
            self.half = False

    def load_model(self):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.model_path, self.config_path, and self.device should be utilized)
        """
        self.model = DetectBackend(weights=self.model_path, device=self.device)
        self.sliced_model = DetectBackend(weights=self.sliced_model_path, device=self.device)
        self.fix_model_dtype()

    def set_model(self, model: Any, **kwargs):
        """
        This function should be implemented to instantiate a DetectionModel out of an already loaded model
        Args:
            model: Any
                Loaded model
        """
        # self.model = model
        # self.fix_model_dtype()
        raise Exception("Yolov6DetectionModel.set_model may not be called as two models must be supplied")

    def set_device(self):
        """
        Sets the device for the model.
        """
        if is_available("torch"):
            self.device = select_torch_device(self.device)
        else:
            raise NotImplementedError()

    def unload_model(self):
        """
        Unloads the model from CPU/GPU.
        """
        self.model = self.sliced_model = None
        if is_available("torch"):
            from sahi.utils.torch import empty_cuda_cache

            empty_cuda_cache()

    def perform_inference(self, images: Union[np.ndarray, List[np.ndarray]], num_batch: int = 1):
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction result should be set to self._original_predictions.
        Args:
            image: np.ndarray or List[np.ndarray]
                A numpy array or list of numpy arrays that contains the image(s) to be predicted, in HWC format.
        """
        if not isinstance(images, list):
            images = [images]

        # import copy
        # images_np = copy.deepcopy(images)

        orig_img_size = images[0].shape[:-1]

        for i in range(len(images)):
            images[i] = letterbox(
                images[i], check_img_size(list(images[i].shape[:-1])), stride=self.model.stride
            )[0].transpose((2, 0, 1)) # HWC to CHW

        images = torch.from_numpy(np.ascontiguousarray(images)).to(self.device)
        images = images.half() if self.half else images.float()
        images /= 255

        assert orig_img_size in (self.full_image_size, self.sliced_image_size), \
            f"Image size must be either the full image size {self.full_image_size} " \
            f"or the sliced image size {self.sliced_image_size}; got {orig_img_size}"

        if orig_img_size == self.full_image_size:
            print("Using full model for images of size", orig_img_size)
            pred_results = self.model(images)
        else:
            print("Using sliced model for images of size", orig_img_size)
            pred_results = self.sliced_model(images)
        
        dets = non_max_suppression(
            prediction=pred_results,
            conf_thres=self.nms_confidence_threshold,
            iou_thres=self.iou_threshold,
            classes=None,
            agnostic=False,
            max_det=1000,
        )

        for i in range(len(dets)):
            dets[i][:, :4] = Inferer.rescale(images[i].shape[1:], dets[i][:, :4], orig_img_size).round()
            dets[i] = dets[i][dets[i][:, 4] >= self.filter_confidence_threshold]

            # from PIL import Image, ImageDraw
            # import time
            # import os
            # im = Image.fromarray(images_np[i])
            # draw = ImageDraw.Draw(im)
            # for *xyxy, conf, cls in dets[i]:
            #     draw.rectangle(xyxy, fill=None, outline="red")

            # os.makedirs("sahi_output", exist_ok=True)
            # im.save(f"sahi_output/{int(time.time())}_slice{i+1}.png")

        self._original_predictions = dets

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        This function should be implemented in a way that self._original_predictions should
        be converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list. self.mask_threshold can also be utilized.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """

        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        full_shape_list *= len(shift_amount_list)  # fix zip later as full_shape_list only has 1 item
        self._object_prediction_list_per_image = [
            [ObjectPrediction(
                bbox=[x.item() for x in xyxy],
                category_id=int(cls),
                score=float(conf),
                segmentation=None,
                category_name=self.category_mapping[str(int(cls))],
                shift_amount=shift_amount,
                full_shape=full_shape,
            ) for *xyxy, conf, cls in single_img_preds]
            for single_img_preds, shift_amount, full_shape
            in zip(self._original_predictions, shift_amount_list, full_shape_list)
        ]

    def _apply_category_remapping(self):
        """
        Applies category remapping based on mapping given in self.category_remapping
        """
        # confirm self.category_remapping is not None
        if self.category_remapping is None:
            raise ValueError("self.category_remapping cannot be None")
        # remap categories
        for object_prediction_list in self._object_prediction_list_per_image:
            for object_prediction in object_prediction_list:
                old_category_id_str = str(object_prediction.category.id)
                new_category_id_int = self.category_remapping[old_category_id_str]
                object_prediction.category.id = new_category_id_int

    def convert_original_predictions(
        self,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Converts original predictions of the detection model to a list of
        prediction.ObjectPrediction object. Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        self._create_object_prediction_list_from_original_predictions(
            shift_amount_list=shift_amount,
            full_shape_list=full_shape,
        )
        if self.category_remapping:
            self._apply_category_remapping()

    @property
    def object_prediction_list(self):
        return self._object_prediction_list_per_image

    @property
    def object_prediction_list_per_image(self):
        return self._object_prediction_list_per_image

    @property
    def original_predictions(self):
        return self._original_predictions
