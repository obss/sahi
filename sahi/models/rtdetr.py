# OBSS SAHI Tool
# Code written by AnNT, 2023.

from typing import Any, List, Optional
import numpy as np
import torch

from .base import DetectionModel


class RTDetrDetectionModel(DetectionModel):
    def load_model(self):
        from ultralytics import RTDETR
        model = RTDETR(self.model_path)
        model.to(self.device)
        self.set_model(model)

    def set_model(self, model: Any, **kwargs):
        # model set + bayraklar + sınıf isimleri
        self.model = model
        self.has_mask = False
        self.is_obb = False
        names = getattr(getattr(model, "model", model), "names", None)
        self.category_names = names
        self.set_model_loaded(True)

    def perform_inference(self, image: np.ndarray):
        if self.model is None:
            self.load_model()
        with torch.no_grad():
            results = self.model(image, imgsz=getattr(self, "image_size", None), verbose=False)
        res = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(res, "boxes", None)
        if hasattr(boxes, "data"):
            boxes = boxes.data  # (N,6) -> [x1,y1,x2,y2,conf,cls]
        self._original_predictions = boxes

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = None,
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        from sahi.prediction import ObjectPrediction

        preds = getattr(self, "_original_predictions", None)
        if preds is None:
            self._object_prediction_list_per_image = [[]]
            return self._object_prediction_list_per_image

        batch = preds if isinstance(preds, (list, tuple)) else [preds]
        if shift_amount_list is None:
            shift_amount_list = [[0, 0] for _ in batch]
        if full_shape_list is None:
            full_shape_list = [[None, None] for _ in batch]

        out: List[List[ObjectPrediction]] = []
        conf_thres = float(getattr(self, "confidence_threshold", 0.0) or 0.0)

        for i, boxes in enumerate(batch):
            if boxes is None:
                out.append([])
                continue

            arr = boxes.detach().cpu().numpy() if torch.is_tensor(boxes) else np.asarray(boxes)
            if arr.ndim == 1:
                arr = arr[None, :]

            dx, dy = shift_amount_list[i]
            H, W = full_shape_list[i]
            lst: List[ObjectPrediction] = []

            for row in arr:
                x1, y1, x2, y2 = map(float, row[:4])
                conf = float(row[4]) if arr.shape[1] > 4 else 1.0
                cls_id = int(row[5]) if arr.shape[1] > 5 else 0
                if conf < conf_thres:
                    continue
                obj = ObjectPrediction(
                    bbox=[x1, y1, x2, y2],
                    segmentation=None,
                    category_id=cls_id,
                    category_name=(self.category_names[cls_id] if self.category_names else str(cls_id)),
                    shift_amount=[dx, dy],
                    full_shape=[H, W],
                )
                obj.score.value = conf
                lst.append(obj)

            out.append(lst)

        self._object_prediction_list_per_image = out
        return out
