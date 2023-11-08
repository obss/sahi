import numpy as np
import cv2
import torch

import logging
from typing import Any, List, Optional, Tuple
import time
logger = logging.getLogger(__name__)


from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements
# from openvino.runtime import Core, AsyncInferQueue
# from ultralytics.utils.ops import non_max_suppression, scale_boxes


class Yolov8OpenvinoDetectionModel(DetectionModel):

    # def __init__(self):
    #     self.output = None

    def check_dependencies(self) -> None:
        check_requirements(["ultralytics"])

    def load_model(self):
        
        from openvino.runtime import Core, AsyncInferQueue
        from ultralytics.utils.ops import non_max_suppression, scale_boxes

        self.non_max_suppression = non_max_suppression
        self.scale_boxes = scale_boxes
        """
        OpenVino IR model is initialized and set to self.model.
        """
        try:
            core = Core()
            ov_model = core.read_model(self.model_path)
            self.cls = ov_model.rt_info
            model = core.compile_model(ov_model, "CPU")
            self.infer_queue = AsyncInferQueue(model, 2)
            self.infer_queue.set_callback(self.callback)
            self.set_model(model)
            self.output = None
        except Exception as e:
            raise TypeError("model_path is not a valid yolov8 OpenVino model path: ", e)

    def set_model(self, model: Any):
        """
        Sets the underlying YOLOv8 model.
        Args:
            model: Any
                A YOLOv8 OpenVino IR model
        """
        self.model = model
        self.input_layer_ir = self.model.input(0)
        self.output_layer = self.model.output(0)

        # set category_mapping
        if not self.category_mapping:
            category_mapping = {str(ind): category_name for ind, category_name in enumerate(self.category_names)}
            self.category_mapping = category_mapping
   
    def callback(self,infer_request, info) -> None:
        """
        Define the callback function for postprocessing
        
        :param: infer_request: the infer_request object
                info: a tuple includes original frame and starts time
        :returns:
                None
        """
        result = infer_request.get_output_tensor(0).data

        input_hw = self.input_tensor.shape[2:]

        prediction_result = self.non_max_suppression(
            torch.from_numpy(result),
            conf_thres = self.confidence_threshold
        )

        #Scale the detected bboxes
        for i, pred in enumerate(prediction_result):
            shape = self.orig_image.shape      
            pred[:, :4] = self.scale_boxes(input_hw, pred[:, :4], shape).round()

        if self.output == None:
            self.output = prediction_result

    def pad_resize_image(self,
        cv2_img: np.ndarray,
        new_size: Tuple[int, int] = (640, 640),
        color: Tuple[int, int, int] = (125, 125, 125)) -> np.ndarray:
        """Resize and pad image with color if necessary, maintaining orig scale

        args:
            cv2_img: numpy.ndarray = cv2 image
            new_size: tuple(int, int) = (width, height)
            color: tuple(int, int, int) = (B, G, R)
        """
        in_h, in_w = cv2_img.shape[:2]
        new_w, new_h = new_size
        # rescale down
        scale = min(new_w / in_w, new_h / in_h)
        # get new sacled widths and heights
        scale_new_w, scale_new_h = int(in_w * scale), int(in_h * scale)
        
        resized_img = cv2.resize(cv2_img, (scale_new_w, scale_new_h))

        # print(resized_img.shape)
        # calculate deltas for padding
        d_w = max(new_w - scale_new_w, 0)
        d_h = max(new_h - scale_new_h, 0)
        # center image with padding on top/bottom or left/right
        top, bottom = d_h // 2, d_h - (d_h // 2)
        left, right = d_w // 2, d_w - (d_w // 2)
        pad_resized_img = cv2.copyMakeBorder(resized_img,
                                            top, bottom, left, right,
                                            cv2.BORDER_CONSTANT,
                                            value=color)
        return pad_resized_img

    def perform_inference(self, image: np.ndarray):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """
        self.orig_image = image.copy()

        #Resize image and padding for detection.
        image = self.pad_resize_image(image)
        
        # Convert HWC to CHW
        image = image.transpose(2, 0, 1)

        #Image to Tensor
        image = np.ascontiguousarray(image)
        input_tensor = image.astype(np.float32)
        input_tensor /= 255.0
        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        self.input_tensor = input_tensor.astype(np.float32)

        self.infer_queue.start_async({self.input_layer_ir.any_name: self.input_tensor},(self.input_tensor, time.time()))
        self.infer_queue.wait_all()
        if self.output != None:
            self._original_predictions = self.output
        self.output = None

    @property
    def category_names(self):

        return eval(self.cls["framework"]["names"].value).values()

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []
        for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[image_ind]
            object_prediction_list = []

            # process predictions
            for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
                x1 = prediction[0]
                y1 = prediction[1]
                x2 = prediction[2]
                y2 = prediction[3]
                bbox = [x1, y1, x2, y2]
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
            object_prediction_list_per_image.append(object_prediction_list)
        # print(object_prediction_list_per_image)
        self._object_prediction_list_per_image = object_prediction_list_per_image

