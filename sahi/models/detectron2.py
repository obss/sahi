# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import logging
from typing import List, Optional

import numpy as np

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_bbox_from_bool_mask
from sahi.utils.import_utils import check_requirements

logger = logging.getLogger(__name__)


class Detectron2DetectionModel(DetectionModel):
    def check_dependencies(self):
        check_requirements(["torch", "detectron2"])

    def load_model(self):
        from detectron2.config import get_cfg
        from detectron2.data import MetadataCatalog
        from detectron2.engine import DefaultPredictor
        from detectron2.model_zoo import model_zoo

        cfg = get_cfg()

        try:  # try to load from model zoo
            config_file = model_zoo.get_config_file(self.config_path)
            cfg.merge_from_file(config_file)
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.config_path)
        except Exception as e:  # try to load from local
            print(e)
            if self.config_path is not None:
                cfg.merge_from_file(self.config_path)
            cfg.MODEL.WEIGHTS = self.model_path

        # set model device
        cfg.MODEL.DEVICE = self.device
        # set input image size
        if self.image_size is not None:
            cfg.INPUT.MIN_SIZE_TEST = self.image_size
            cfg.INPUT.MAX_SIZE_TEST = self.image_size
        # init predictor
        model = DefaultPredictor(cfg)

        self.model = model

        # detectron2 category mapping
        if self.category_mapping is None:
            try:  # try to parse category names from metadata
                metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
                category_names = metadata.thing_classes
                self.category_names = category_names
                self.category_mapping = {
                    str(ind): category_name for ind, category_name in enumerate(self.category_names)
                }
            except Exception as e:
                logger.warning(e)
                # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#update-the-config-for-new-datasets
                if cfg.MODEL.META_ARCHITECTURE == "RetinaNet":
                    num_categories = cfg.MODEL.RETINANET.NUM_CLASSES
                else:  # fasterrcnn/maskrcnn etc
                    num_categories = cfg.MODEL.ROI_HEADS.NUM_CLASSES
                self.category_names = [str(category_id) for category_id in range(num_categories)]
                self.category_mapping = {
                    str(ind): category_name for ind, category_name in enumerate(self.category_names)
                }
        else:
            self.category_names = list(self.category_mapping.values())

    def perform_inference(self, images: List):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            images: List[np.ndarray, PIL.Image.Image]
                A numpy array that contains one image to be predicted. 3 channel image should be in RGB order.
        """

        if not isinstance(images, list):
            images = [images]

        if len(images) > 1:
            raise NotImplementedError("Detectron2 does not support batch inference.")

        # Confirm model is loaded
        if self.model is None:
            raise RuntimeError("Model is not loaded, load it by calling .load_model()")

        if isinstance(images[0], np.ndarray) and self.model.input_format == "BGR":
            # convert RGB image to BGR format
            images[0] = images[0][:, :, ::-1]

        prediction_result = self.model(images[0])

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        num_categories = len(self.category_mapping)
        return num_categories

    def _create_object_predictions_from_original_predictions(
        self,
        shift_amounts: Optional[List[List[int]]] = [[0, 0]],
        full_shapes: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_predictions_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # parse boxes, masks, scores, category_ids from predictions
        boxes = original_predictions["instances"].pred_boxes.tensor.tolist()
        scores = original_predictions["instances"].scores.tolist()
        category_ids = original_predictions["instances"].pred_classes.tolist()

        # check if predictions contain mask
        try:
            masks = original_predictions["instances"].pred_masks.tolist()
        except AttributeError:
            masks = None

        # create object_predictions
        object_predictions_per_image = []
        object_predictions = []

        # detectron2 DefaultPredictor supports single image
        shift_amount = shift_amounts[0]
        full_shape = None if full_shapes is None else full_shapes[0]

        for ind in range(len(boxes)):
            score = scores[ind]
            if score < self.confidence_threshold:
                continue

            category_id = category_ids[ind]

            if masks is None:
                bbox = boxes[ind]
                mask = None
            else:
                mask = np.array(masks[ind])

                # check if mask is valid
                # https://github.com/obss/sahi/issues/389
                if get_bbox_from_bool_mask(mask) is None:
                    continue
                else:
                    bbox = None

            object_prediction = ObjectPrediction(
                bbox=bbox,
                bool_mask=mask,
                category_id=category_id,
                category_name=self.category_mapping[str(category_id)],
                shift_amount=shift_amount,
                score=score,
                full_shape=full_shape,
            )
            object_predictions.append(object_prediction)

        # detectron2 DefaultPredictor supports single image
        object_predictions_per_image = [object_predictions]

        self._object_predictions_per_image = object_predictions_per_image
