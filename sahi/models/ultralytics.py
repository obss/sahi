from typing import List

from sahi.models.base import DetectionModel


class UltralyticsDetectionModel(DetectionModel):
    """
    Ultralytics YOLO (v8/v11) modelleri için DetectionModel.
    """

    def load_model(self):
        from ultralytics import YOLO

        # Modeli yükle
        self.model = YOLO(self.model_path)

        # Cihaz
        if hasattr(self.model, "to"):
            self.model.to(self.device)

        # Görev bayrakları (detect/segment/obb)
        task = getattr(getattr(self.model, "model", self.model), "task", None)
        self.has_mask = task == "segment"
        self.is_obb = task == "obb"

        # (Opsiyonel) sınıf isimleri
        names = None
        try:
            names = getattr(getattr(self.model, "model", self.model), "names", None)
        except Exception:
            pass
        self.category_names = names

        # set_model_loaded yok; gerek de yok. Model yüklendi.
        return self.model

    def _to_yolo_xyxy_conf_cls_tensor(self, boxes):
        """
        Herhangi bir boxes formatını YOLO [N,6] (x1,y1,x2,y2,conf,cls) tensörüne çevirir.
        """
        import torch

        if boxes is None:
            return None

        # Eğer Results nesnesi geldiyse -> .boxes'a in
        rb = getattr(boxes, "boxes", None)
        if rb is not None:
            boxes = rb

        # Ultralytics Boxes: doğrudan .data varsa onu kullan
        if hasattr(boxes, "data") and isinstance(boxes.data, torch.Tensor):
            return boxes.data

        # Bazı sürümlerde .xyxy/.conf/.cls alanları var
        if all(hasattr(boxes, a) for a in ("xyxy", "conf", "cls")):
            xyxy = boxes.xyxy
            conf = boxes.conf[:, None] if boxes.conf.ndim == 1 else boxes.conf
            cls_ = boxes.cls[:, None] if boxes.cls.ndim == 1 else boxes.cls

            if not isinstance(xyxy, torch.Tensor):
                xyxy = torch.as_tensor(xyxy)
            if not isinstance(conf, torch.Tensor):
                conf = torch.as_tensor(conf)
            if not isinstance(cls_, torch.Tensor):
                cls_ = torch.as_tensor(cls_)

            return torch.cat([xyxy, conf, cls_], dim=1)

        # Liste/tuple içinde tensörler ise birleştir
        if isinstance(boxes, (list, tuple)) and len(boxes) and isinstance(boxes[0], torch.Tensor):
            return torch.cat([b for b in boxes], dim=0)

        # Son çare: tensöre dök
        return torch.as_tensor(boxes)

    def _boxes_to_tensor(self, boxes):
        import torch

        # Eğer Results nesnesi geldiyse -> .boxes'a in
        rb = getattr(boxes, "boxes", None)
        if rb is not None:
            boxes = rb

        # Ultralytics Boxes: doğrudan .data varsa onu kullan
        if hasattr(boxes, "data") and isinstance(boxes.data, torch.Tensor):
            return boxes.data

        # Bazı sürümlerde .xyxy/.conf/.cls alanları var
        if all(hasattr(boxes, a) for a in ("xyxy", "conf", "cls")):
            xyxy = boxes.xyxy
            conf = boxes.conf[:, None] if boxes.conf.ndim == 1 else boxes.conf
            cls_ = boxes.cls[:, None] if boxes.cls.ndim == 1 else boxes.cls

            if not isinstance(xyxy, torch.Tensor):
                xyxy = torch.as_tensor(xyxy)
            if not isinstance(conf, torch.Tensor):
                conf = torch.as_tensor(conf)
            if not isinstance(cls_, torch.Tensor):
                cls_ = torch.as_tensor(cls_)

            return torch.cat([xyxy, conf, cls_], dim=1)

        # Liste/tuple içinde tensörler ise birleştir
        if isinstance(boxes, (list, tuple)) and len(boxes) and isinstance(boxes[0], torch.Tensor):
            return torch.cat([b for b in boxes], dim=0)

        # Son çare: tensöre dök
        return torch.as_tensor(boxes)

    def perform_inference_batch(self, images, **kwargs):
        import numpy as np
        import torch
        from PIL import Image

        if self.model is None:
            self.load_model()

        norm_imgs = []
        full_shape_list = []
        for im in images:
            if isinstance(im, Image.Image):
                arr = np.array(im)  # RGB
            elif isinstance(im, np.ndarray):
                arr = im
            elif isinstance(im, torch.Tensor):
                t = im.detach().cpu()
                if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                    t = t.permute(1, 2, 0)
                elif t.ndim == 4 and t.shape[0] == 1:
                    t = t.squeeze(0).permute(1, 2, 0)
                arr = t.numpy()
            else:
                raise TypeError(f"Unsupported image type: {type(im)}")

            norm_imgs.append(arr)
            full_shape_list.append([int(arr.shape[0]), int(arr.shape[1])])

        with torch.no_grad():
            results = self.model(norm_imgs, verbose=False)

        # --- BURASI ÖNEMLİ: her zaman listeye normalize et
        try:
            from ultralytics.engine.results import Results as UResults
        except Exception:
            UResults = None

        if UResults and isinstance(results, UResults):
            results_list = [results]
        elif isinstance(results, (list, tuple)):
            results_list = list(results)
        else:
            results_list = [results]
        # ---

        # prediction_result'u sadece tensör/ilkel yapılardan kur
        prediction_result = []
        for res in results_list:
            b = getattr(res, "boxes", None)
            m = getattr(res, "masks", None)
            obb = getattr(res, "obb", None)

            if self.is_obb:
                rboxes = getattr(getattr(obb, "rboxes", None), "data", None)
                boxes = getattr(b, "data", None)
                masks = getattr(m, "data", None) if m is not None else None
                prediction_result.append((boxes, rboxes, masks))
            elif self.has_mask:
                boxes = getattr(b, "data", None)
                masks = getattr(m, "data", None) if m is not None else None
                prediction_result.append((boxes, masks))
            else:
                boxes = getattr(b, "data", None)
                prediction_result.append(boxes)

        self._original_predictions = prediction_result

        # Get slice offsets and full shape from kwargs if provided
        slice_offsets = kwargs.get("slice_offsets", None)
        full_shape = kwargs.get("full_shape", None)

        if slice_offsets is not None and full_shape is not None:
            # Use provided slice offsets and full shape
            shift_amount_list = slice_offsets
            full_shape_list = [full_shape for _ in norm_imgs]
        else:
            # Fallback to default values
            shift_amount_list = [[0, 0] for _ in norm_imgs]

        self._create_object_prediction_list_from_original_predictions(
            shift_amount_list=shift_amount_list,
            full_shape_list=full_shape_list,
        )

        return self._object_prediction_list_per_image

    def perform_inference(self, image):
        """
        Tek görsel tahmin. self._original_predictions'ı testlerin beklediği formatta kurar.
        Detect: [boxes_tensor]
        Seg:    [(boxes_tensor, masks_tensor)]
        OBB:    [(boxes_tensor_xyxyccls, obb_xyxyxyxy_tensor, None)]
        """
        import numpy as np
        import torch
        from PIL import Image

        if self.model is None:
            self.load_model()

        # HWC numpy'a normalize et
        if isinstance(image, Image.Image):
            im = np.array(image)
        elif isinstance(image, torch.Tensor):
            t = image.detach().cpu()
            if t.ndim == 3 and t.shape[0] in (1, 3, 4):
                t = t.permute(1, 2, 0)
            elif t.ndim == 4 and t.shape[0] == 1:
                t = t.squeeze(0).permute(1, 2, 0)
            im = t.numpy()
        else:
            im = image

        # Giriş boyutunu kaydet
        h, w = int(im.shape[0]), int(im.shape[1])
        self._last_full_shape = [h, w]

        # Ultralytics tahmini (eşik ve boyutu geçir)
        predict_kwargs = {}
        if getattr(self, "confidence_threshold", None) is not None:
            predict_kwargs["conf"] = float(self.confidence_threshold)
        if getattr(self, "image_size", None) is not None:
            predict_kwargs["imgsz"] = int(self.image_size)

        with torch.no_grad():
            results = self.model([im], verbose=False, **predict_kwargs)
        res = results[0]

        th = float(getattr(self, "confidence_threshold", 0.0) or 0.0)

        # --- SEG (mask) değil, OBB değil: klasik detect
        if not getattr(self, "has_mask", False) and not getattr(self, "is_obb", False):
            data = res.boxes.data  # (N, 6) -> xyxy, conf, cls
            if th > 0:
                data = data[data[:, 4] >= th]
            # Liste olarak kaydet (testler böyle bekliyor)
            self._original_predictions = [data]
            return

        # --- SEGMENTATION
        if getattr(self, "has_mask", False) and not getattr(self, "is_obb", False):
            boxes = res.boxes.data if res.boxes is not None else None
            masks = res.masks.data if getattr(res, "masks", None) is not None else None
            if boxes is None:
                self._original_predictions = [(None, masks)]
                return
            if th > 0:
                keep = boxes[:, 4] >= th
                boxes = boxes[keep]
                if masks is not None:
                    masks = masks[keep]
            self._original_predictions = [(boxes, masks)]
            return

        # --- OBB
        # Beklenti: (boxes_xyxyccls, obb_xyxyxyxy, None)
        obb = getattr(res, "obb", None)
        xyxyxyxy = None
        if obb is not None:
            # Ultralytics OBB polygonları
            xyxyxyxy = getattr(obb, "xyxyxyxy", None)
            if hasattr(xyxyxyxy, "data"):
                xyxyxyxy = xyxyxyxy.data

        # Axis-aligned boxes'ı polygonlardan üret
        boxes_tensor = None
        if xyxyxyxy is not None:
            # (N, 8) -> (N, 4, 2)
            coords = xyxyxyxy.view(-1, 4, 2) if hasattr(xyxyxyxy, "view") else xyxyxyxy.reshape(-1, 4, 2)
            x = coords[..., 0]
            y = coords[..., 1]
            x1 = x.min(dim=1).values if hasattr(x, "min") else x.min(1)[0]
            y1 = y.min(dim=1).values if hasattr(y, "min") else y.min(1)[0]
            x2 = x.max(dim=1).values if hasattr(x, "max") else x.max(1)[0]
            y2 = y.max(dim=1).values if hasattr(y, "max") else y.max(1)[0]

            # OBB skor ve sınıf bilgisi varsa kullan
            conf = getattr(obb, "conf", None)
            cls_ = getattr(obb, "cls", None)
            if conf is None:
                conf = torch.ones_like(x1, dtype=x1.dtype)
            if cls_ is None:
                cls_ = torch.zeros_like(x1, dtype=x1.dtype)

            if conf.ndim == 1:
                conf = conf.unsqueeze(1)
            if cls_.ndim == 1:
                cls_ = cls_.unsqueeze(1)

            boxes_tensor = torch.stack([x1, y1, x2, y2], dim=1)
            boxes_tensor = torch.cat([boxes_tensor, conf, cls_], dim=1)  # (N, 6)

            # Eşik uygula
            if th > 0:
                keep = boxes_tensor[:, 4] >= th
                boxes_tensor = boxes_tensor[keep]
                xyxyxyxy = xyxyxyxy[keep]

        # Eğer Ultralytics klasik axis-aligned boxes doldurmuşsa, onu da tercih edebiliriz
        if boxes_tensor is None and res.boxes is not None:
            data = res.boxes.data
            if th > 0:
                data = data[data[:, 4] >= th]
            boxes_tensor = data

        self._original_predictions = [(boxes_tensor, xyxyxyxy, None)]

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list=None,
        full_shape_list=None,
    ):
        import numpy as np
        import torch

        from sahi.prediction import ObjectPrediction

        preds = getattr(self, "_original_predictions", None)
        if preds is None:
            self._object_prediction_list_per_image = [[]]
            return self._object_prediction_list_per_image

        # >>> FARK: her zaman listeye çevir
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        n = len(preds)

        # shift default
        if shift_amount_list is None or (
            isinstance(shift_amount_list, (list, tuple))
            and shift_amount_list
            and isinstance(shift_amount_list[0], (int, float))
        ):
            shift_amount_list = [[0, 0] for _ in range(n)]

        # FULL SHAPE fallback (en kritik kısım)
        if full_shape_list is None or (
            isinstance(full_shape_list, (list, tuple))
            and full_shape_list
            and (full_shape_list[0] is None or isinstance(full_shape_list[0], (int, float)))
        ):
            if hasattr(self, "_last_full_shape") and self._last_full_shape:
                full_shape_list = [self._last_full_shape for _ in range(n)]
            else:
                # son çare: 0,0 ver (clip içinde hatayı önlemek için)
                full_shape_list = [[0, 0] for _ in range(n)]
        # <<< FARK

        out: List[List[ObjectPrediction]] = []
        conf_thres = float(getattr(self, "confidence_threshold", 0.0) or 0.0)

        for i in range(n):
            p = preds[i]
            dx, dy = shift_amount_list[i]
            H, W = full_shape_list[i]

            # boxes + (opsiyonel) obb_points'u ayıkla
            boxes = p
            obb_points = None
            if isinstance(p, (list, tuple)):
                if len(p) >= 1:
                    boxes = p[0]
                if len(p) >= 2:
                    obb_points = p[1]

            # boxes -> torch.Tensor [N, >=6] (x1,y1,x2,y2,conf,cls)
            boxes = self._to_yolo_xyxy_conf_cls_tensor(
                boxes
            )  # kendi yardımcı dönüşümün; zaten detection tarafında kullanıyorsun
            if boxes is None or boxes.numel() == 0:
                out.append([])
                continue

            # obb_points -> numpy [N, 4, 2] (varsa)
            obb_np = None
            if obb_points is not None:
                if hasattr(obb_points, "data"):
                    obb_points = obb_points.data
                if isinstance(obb_points, torch.Tensor):
                    obb_np = obb_points.detach().cpu().numpy()
                else:
                    obb_np = np.asarray(obb_points)

            objs: List[ObjectPrediction] = []
            for j in range(boxes.shape[0]):
                x1, y1, x2, y2, conf, cls_id = (
                    boxes[j, 0].item(),
                    boxes[j, 1].item(),
                    boxes[j, 2].item(),
                    boxes[j, 3].item(),
                    boxes[j, 4].item(),
                    int(boxes[j, 5].item()),
                )
                if conf < conf_thres:
                    continue

                cat_name = None
                if getattr(self, "category_names", None) and 0 <= cls_id < len(self.category_names):
                    cat_name = self.category_names[cls_id]

                score = float(conf)

                if self.is_obb and obb_np is not None:
                    # 4 köşe -> COCO polygon (tek halka)
                    # obb_np[j] şekli (4, 2) -> [x1,y1,x2,y2,x3,y3,x4,y4]
                    seg = obb_np[j].reshape(-1).tolist()
                    obj = ObjectPrediction(
                        segmentation=[seg],
                        category_id=cls_id,
                        category_name=cat_name,
                        score=score,
                        shift_amount=[dx, dy],
                        full_shape=[H, W],
                    )
                else:
                    # klasik axis-aligned bbox
                    obj = ObjectPrediction(
                        bbox=[x1, y1, x2, y2],
                        category_id=cls_id,
                        category_name=cat_name,
                        score=score,
                        shift_amount=[dx, dy],
                        full_shape=[H, W],
                    )

                objs.append(obj)

            out.append(objs)

        self._object_prediction_list_per_image = out
        return out
