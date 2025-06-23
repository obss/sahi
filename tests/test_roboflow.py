import unittest

from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.util.coco_classes import COCO_CLASSES

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction
from sahi.utils.cv import read_image


class TestRoboflowDetectionModel(unittest.TestCase):
    def test_roboflow_universe(self):
        """Test the Roboflow Universe model for object detection."""
        model = AutoDetectionModel.from_pretrained(
            model_type="roboflow",
            model="rfdetr-base",
            confidence_threshold=0.5,
            device="cpu",
        )

        image_path = "tests/data/small-vehicles1.jpeg"
        image = read_image(image_path)

        result = get_prediction(image, model)
        predictions = result.object_prediction_list

        self.assertGreater(len(predictions), 0)

        sliced_results = get_sliced_prediction(
            image,
            model,
            slice_height=224,
            slice_width=224,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        sliced_predictions = sliced_results.object_prediction_list
        self.assertGreater(len(sliced_predictions), len(predictions))

    def test_rfdetr(self):
        """Test the RFDETR model classes and instances for object detection."""
        models = [
            RFDETRBase,
            RFDETRBase(),
            RFDETRLarge,
            RFDETRLarge(),
        ]
        for model_variant in models:
            with self.subTest(model=model_variant):
                model = AutoDetectionModel.from_pretrained(
                    model_type="roboflow",
                    model=model_variant,
                    confidence_threshold=0.5,
                    category_mapping=COCO_CLASSES,
                    device="cpu",
                )

                image_path = "tests/data/small-vehicles1.jpeg"
                image = read_image(image_path)

                result = get_prediction(image, model)
                predictions = result.object_prediction_list

                self.assertGreater(len(predictions), 0)

                sliced_results = get_sliced_prediction(
                    image,
                    model,
                    slice_height=224,
                    slice_width=224,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2,
                )

                sliced_predictions = sliced_results.object_prediction_list
                self.assertGreater(len(sliced_predictions), len(predictions))


if __name__ == "__main__":
    unittest.main()
