# OBSS SAHI Tool
# Code written by Karl-Joan Alesma, 2023

import unittest

import cv2
import numpy as np

from sahi.utils.yolov8onnx import Yolov8ONNXTestConstants, download_yolov8n_onnx_model

MODEL_DEVICE = "cpu"
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.7
IMAGE_SIZE = 640


class TestYolov8OnnxDetectionModel(unittest.TestCase):
    def test_load_model(self):
        from sahi.models.yolov8onnx import Yolov8OnnxDetectionModel

        download_yolov8n_onnx_model()

        yolov8_onnx_detection_model = Yolov8OnnxDetectionModel(
            model_path=Yolov8ONNXTestConstants.YOLOV8N_ONNX_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device=MODEL_DEVICE,
            category_mapping={"0": "something"},
            load_at_init=False,
        )

        # Test setting options for onnxruntime
        yolov8_onnx_detection_model.load_model({"enable_mem_pattern": False})

        self.assertNotEqual(yolov8_onnx_detection_model.model, None)

    def test_set_model(self):
        import onnxruntime

        from sahi.models.yolov8onnx import Yolov8OnnxDetectionModel

        download_yolov8n_onnx_model()

        yolo_model = onnxruntime.InferenceSession(Yolov8ONNXTestConstants.YOLOV8N_ONNX_MODEL_PATH)

        yolov8_onnx_detection_model = Yolov8OnnxDetectionModel(
            model=yolo_model,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device=MODEL_DEVICE,
            category_mapping={"0": "something"},
            load_at_init=True,
        )

        self.assertNotEqual(yolov8_onnx_detection_model.model, None)

    def test_perform_inference(self):
        from sahi.models.yolov8onnx import Yolov8OnnxDetectionModel

        # Init model
        download_yolov8n_onnx_model()

        yolov8_onnx_detection_model = Yolov8OnnxDetectionModel(
            model_path=Yolov8ONNXTestConstants.YOLOV8N_ONNX_MODEL_PATH,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            iou_threshold=IOU_THRESHOLD,
            device=MODEL_DEVICE,
            category_mapping={"0": "something"},
            load_at_init=True,
            image_size=IMAGE_SIZE,
        )

        # Prepare image
        image_path = "tests/data/small-vehicles1.jpeg"
        image = cv2.imread(image_path)

        # Perform inference
        yolov8_onnx_detection_model.perform_inference(image)
        original_predictions = yolov8_onnx_detection_model.original_predictions

        boxes = original_predictions[0]

        # Find most confident bbox for car
        best_box_index = np.argmax(boxes[boxes[:, 5] == 2][:, 4])
        best_bbox = boxes[best_box_index]

        # Compare
        desired_bbox = [603, 239, 629, 259]
        predicted_bbox = best_bbox.tolist()
        margin = 2

        for ind, point in enumerate(predicted_bbox[:4]):
            assert point < desired_bbox[ind] + margin and point > desired_bbox[ind] - margin

        for box in boxes[0]:
            self.assertGreaterEqual(predicted_bbox[4], CONFIDENCE_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
