# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import sys
import unittest
from unittest import mock

from sahi.auto_model import AutoDetectionModel

# layer is not available for python<3.7
if sys.version_info >= (3, 7):

    class TestLayerDetectionModel(unittest.TestCase):
        @mock.patch("layer.get_model")
        def test_load_layer_model(self, mock_layer_get_model):
            import yolov5
            from layer import Model
            from layer.flavors.base import ModelRuntimeObjects

            from sahi.model import Yolov5DetectionModel

            layer_model_path = "sahi/yolo/models/yolov5s"

            # Return a YOLO model once mocked `layer.get_model()` is called
            yolo_model = yolov5.load("tests/data/models/yolov5n.pt")
            layer_model = Model(layer_model_path, model_runtime_objects=ModelRuntimeObjects(yolo_model))
            mock_layer_get_model.return_value = layer_model

            # Make the call
            detection_model = AutoDetectionModel.from_layer(layer_model_path)

            # Run assertions
            mock_layer_get_model.assert_called_once_with(name=layer_model_path, no_cache=False)

            self.assertIsInstance(detection_model, type(Yolov5DetectionModel()))
            self.assertIsInstance(detection_model.model, type(yolo_model))

        @mock.patch("layer.get_model")
        def test_load_layer_fails(self, mock_layer_get_model):
            import torch
            from layer import Model
            from layer.flavors.base import ModelRuntimeObjects

            layer_model_path = "sahi/yolo/models/yolov5s"

            # Return a basic Torch model once mocked `layer.get_model()` is called
            torch_model = torch.nn.Sequential()
            layer_model = Model(layer_model_path, model_runtime_objects=ModelRuntimeObjects(torch_model))
            mock_layer_get_model.return_value = layer_model

            # Make the call expecting an exception
            with self.assertRaises(Exception):
                AutoDetectionModel.from_layer(layer_model_path)


if __name__ == "__main__":
    unittest.main()
