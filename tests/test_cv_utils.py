from unittest.mock import patch

import numpy as np

from sahi.utils.cv import (
    Colors,
    apply_color_mask,
    get_bbox_from_bool_mask,
    get_coco_segmentation_from_bool_mask,
    read_image,
)


class TestCvUtils:
    def test_hex_to_rgb(self):
        colors = Colors()
        assert colors.hex_to_rgb("#FF3838") == (255, 56, 56)

    def test_hex_to_rgb_retrieve(self):
        colors = Colors()
        assert colors(0) == (255, 56, 56)

    @patch("sahi.utils.cv.cv2.cvtColor")
    @patch("sahi.utils.cv.cv2.imread")
    def test_read_image(self, mock_imread, mock_cvtColor):
        fake_image = "test.jpg"
        fake_image_val = np.array([[[10, 20, 30]]], dtype=np.uint8)
        fake_image_rbg_val = np.array([[[10, 20, 30]]], dtype=np.uint8)
        mock_imread.return_value = fake_image_val
        mock_cvtColor.return_value = fake_image_rbg_val

        result = read_image(fake_image)

        # mock_cv2.assert_called_once_with(fake_image)
        mock_imread.assert_called_once_with(fake_image)
        np.testing.assert_array_equal(result, fake_image_rbg_val)

    def test_apply_color_mask(self):
        image = np.array([[0, 1]], dtype=np.uint8)
        color = (255, 0, 0)

        expected_output = np.array([[[0, 0, 0], [255, 0, 0]]], dtype=np.uint8)

        result = apply_color_mask(image, color)

        np.testing.assert_array_equal(result, expected_output)

    def test_get_coco_segmentation_from_bool_mask_simple(self):
        mask = np.zeros((10, 10), dtype=bool)
        result = get_coco_segmentation_from_bool_mask(mask)
        assert result == []

    def test_get_coco_segmentation_from_bool_mask_polygon(self):
        mask = np.zeros((10, 20), dtype=bool)
        mask[1:4, 1:4] = True
        mask[5:8, 5:8] = True
        result = get_coco_segmentation_from_bool_mask(mask)
        assert len(result) == 2

    def test_get_bbox_from_bool_mask(self):
        mask = np.array(
            [
                [False, False, False],
                [False, True, True],
                [False, True, True],
                [False, False, False],
            ]
        )
        expected_result = [1, 1, 2, 2]
        result = get_bbox_from_bool_mask(mask)
        assert result == expected_result


class TestEnhancedImageProcessing:
    """Test cases for enhanced image processing utilities."""

    def test_apply_clahe_grayscale(self):
        from sahi.utils.cv import apply_clahe

        # Create low contrast grayscale image
        image = np.random.randint(100, 150, (100, 100), dtype=np.uint8)
        enhanced = apply_clahe(image)

        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8
        # Enhanced image should have better contrast
        assert np.std(enhanced) >= np.std(image)

    def test_apply_clahe_color(self):
        from sahi.utils.cv import apply_clahe

        # Create low contrast color image
        image = np.random.randint(100, 150, (100, 100, 3), dtype=np.uint8)
        enhanced = apply_clahe(image)

        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8
        assert len(enhanced.shape) == 3

    def test_apply_gaussian_blur(self):
        from sahi.utils.cv import apply_gaussian_blur

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        blurred = apply_gaussian_blur(image, kernel_size=5)

        assert blurred.shape == image.shape
        # Blurred image should be smoother (lower gradient variance)
        assert np.var(np.gradient(blurred)[0]) <= np.var(np.gradient(image)[0])

    def test_apply_bilateral_filter(self):
        from sahi.utils.cv import apply_bilateral_filter

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        filtered = apply_bilateral_filter(image, d=9)

        assert filtered.shape == image.shape
        assert filtered.dtype == np.uint8

    def test_apply_unsharp_mask(self):
        from sahi.utils.cv import apply_unsharp_mask

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        sharpened = apply_unsharp_mask(image, kernel_size=5, amount=1.0)

        assert sharpened.shape == image.shape
        assert sharpened.dtype == np.uint8

    def test_auto_gamma_correction(self):
        from sahi.utils.cv import auto_gamma_correction

        # Test that gamma correction works without errors
        image = np.random.randint(50, 150, (100, 100, 3), dtype=np.uint8)
        corrected = auto_gamma_correction(image, target_mean=128.0)

        assert corrected.shape == image.shape
        assert corrected.dtype == np.uint8
        # Verify gamma correction produces valid pixel values
        assert corrected.min() >= 0
        assert corrected.max() <= 255

    def test_adaptive_threshold_image(self):
        from sahi.utils.cv import adaptive_threshold_image

        # Create grayscale image
        image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        binary = adaptive_threshold_image(image, method="gaussian")

        assert binary.shape == image.shape
        # Binary image should only have 0 or 255
        assert set(np.unique(binary)).issubset({0, 255})

    def test_calculate_image_quality_score(self):
        from sahi.utils.cv import calculate_image_quality_score

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        metrics = calculate_image_quality_score(image)

        assert "brightness" in metrics
        assert "contrast" in metrics
        assert "sharpness" in metrics
        assert "saturation" in metrics
        assert "noise_estimate" in metrics
        assert 0 <= metrics["brightness"] <= 255
        assert metrics["contrast"] >= 0
        assert metrics["sharpness"] >= 0

    def test_is_blurry(self):
        from sahi.utils.cv import apply_gaussian_blur, is_blurry

        # Create sharp image
        sharp_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        # Create blurry image
        blurry_image = apply_gaussian_blur(sharp_image, kernel_size=15)

        assert is_blurry(blurry_image, threshold=100.0)
        # Sharp image may or may not be detected as blurry depending on content

    def test_is_overexposed(self):
        from sahi.utils.cv import is_overexposed

        # Create overexposed image (mostly white)
        overexposed = np.ones((100, 100, 3), dtype=np.uint8) * 255
        assert is_overexposed(overexposed, threshold=0.5)

        # Create normal image
        normal = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        assert not is_overexposed(normal, threshold=0.05)

    def test_is_underexposed(self):
        from sahi.utils.cv import is_underexposed

        # Create underexposed image (mostly black)
        underexposed = np.zeros((100, 100, 3), dtype=np.uint8)
        assert is_underexposed(underexposed, threshold=0.5)

        # Create normal image
        normal = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        assert not is_underexposed(normal, threshold=0.05)

    def test_auto_enhance_image(self):
        from sahi.utils.cv import auto_enhance_image

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        enhanced = auto_enhance_image(image, enhance_contrast=True, enhance_sharpness=True, denoise=False)

        assert enhanced.shape == image.shape
        assert enhanced.dtype == np.uint8

    def test_apply_color_jitter(self):
        from sahi.utils.cv import apply_color_jitter

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        jittered = apply_color_jitter(image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)

        assert jittered.shape == image.shape
        assert jittered.dtype == np.uint8
        # Jittered image should be different from original
        assert not np.array_equal(jittered, image)

    def test_create_image_pyramid(self):
        from sahi.utils.cv import create_image_pyramid

        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        pyramid = create_image_pyramid(image, scales=[1.0, 0.5, 0.25], min_size=32)

        assert len(pyramid) == 3
        # First scale should be original size
        assert pyramid[0][0] == 1.0
        assert pyramid[0][1].shape[:2] == (640, 640)
        # Second scale should be half size
        assert pyramid[1][0] == 0.5
        assert pyramid[1][1].shape[:2] == (320, 320)

    def test_apply_mixup(self):
        from sahi.utils.cv import apply_mixup

        image1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        image2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mixed = apply_mixup(image1, image2, alpha=0.5)

        assert mixed.shape == image1.shape
        assert mixed.dtype == np.uint8

    def test_calculate_histogram(self):
        from sahi.utils.cv import calculate_histogram

        # Test color image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        hist = calculate_histogram(image, bins=256)

        assert "b" in hist
        assert "g" in hist
        assert "r" in hist
        assert len(hist["b"]) == 256

        # Test grayscale image
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        hist_gray = calculate_histogram(gray, bins=256)

        assert "gray" in hist_gray
        assert len(hist_gray["gray"]) == 256

    def test_match_histogram(self):
        from sahi.utils.cv import match_histogram

        source = np.random.randint(0, 100, (100, 100, 3), dtype=np.uint8)
        reference = np.random.randint(150, 255, (100, 100, 3), dtype=np.uint8)
        matched = match_histogram(source, reference)

        assert matched.shape == source.shape
        assert matched.dtype == np.uint8
        # Matched image should be brighter than source
        assert np.mean(matched) > np.mean(source)


class TestAdvancedSlicingUtilities:
    """Test cases for advanced slicing strategies."""

    def test_calculate_optimal_slice_size(self):
        from sahi.slicing import calculate_optimal_slice_size

        slice_h, slice_w, overlap_h, overlap_w = calculate_optimal_slice_size(
            image_height=2000,
            image_width=3000,
            model_input_size=640,
            min_overlap_ratio=0.2,
            max_slices=100
        )

        assert slice_h > 0
        assert slice_w > 0
        assert 0 <= overlap_h <= 1
        assert 0 <= overlap_w <= 1

    def test_calculate_optimal_slice_size_with_target_object(self):
        from sahi.slicing import calculate_optimal_slice_size

        slice_h, slice_w, overlap_h, overlap_w = calculate_optimal_slice_size(
            image_height=2000,
            image_width=3000,
            model_input_size=640,
            target_object_size=100
        )

        # Slices should be at least 3x target object size
        assert slice_h >= 300
        assert slice_w >= 300

    def test_generate_adaptive_grid_without_density(self):
        from sahi.slicing import generate_adaptive_grid

        slices = generate_adaptive_grid(
            image_height=1000,
            image_width=1500,
            density_map=None,
            base_slice_size=640
        )

        assert len(slices) > 0
        # Each slice should have (x, y, width, height)
        for slice in slices:
            assert len(slice) == 4
            assert all(v >= 0 for v in slice)

    def test_calculate_slice_overlap_iou(self):
        from sahi.slicing import calculate_slice_overlap_iou

        # Non-overlapping slices
        slice1 = (0, 0, 100, 100)
        slice2 = (200, 200, 100, 100)
        iou = calculate_slice_overlap_iou(slice1, slice2)
        assert iou == 0.0

        # Partially overlapping slices
        slice3 = (50, 50, 100, 100)
        iou2 = calculate_slice_overlap_iou(slice1, slice3)
        assert 0 < iou2 < 1

        # Identical slices
        iou3 = calculate_slice_overlap_iou(slice1, slice1)
        assert iou3 == 1.0

    def test_visualize_slicing_grid(self):
        from sahi.slicing import visualize_slicing_grid

        slices = [(0, 0, 640, 640), (320, 320, 640, 640), (640, 0, 640, 640)]
        vis = visualize_slicing_grid(
            image_height=1280,
            image_width=1280,
            slices=slices,
            output_path=None,
            show_overlap=True
        )

        assert vis.shape == (1280, 1280, 3)
        assert vis.dtype == np.uint8

    def test_estimate_memory_usage(self):
        from sahi.slicing import estimate_memory_usage

        memory_est = estimate_memory_usage(
            image_height=4000,
            image_width=6000,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )

        assert "num_slices" in memory_est
        assert "full_image_mb" in memory_est
        assert "peak_memory_mb" in memory_est
        assert memory_est["num_slices"] > 0
        assert memory_est["full_image_mb"] > 0

    def test_optimize_slicing_parameters(self):
        from sahi.slicing import optimize_slicing_parameters

        config = optimize_slicing_parameters(
            image_height=4000,
            image_width=6000,
            model_input_size=640,
            max_memory_mb=8000,
            min_overlap_ratio=0.1,
            max_overlap_ratio=0.3
        )

        assert "slice_height" in config
        assert "slice_width" in config
        assert "overlap_height_ratio" in config
        assert "overlap_width_ratio" in config
        assert config["slice_height"] > 0
        assert config["slice_width"] > 0


class TestBatchProcessing:
    """Test cases for batch processing and performance utilities."""

    def test_analyze_prediction_distribution(self):
        from sahi.predict import analyze_prediction_distribution
        from sahi.prediction import ObjectPrediction, PredictionResult

        # Create mock predictions
        predictions = []
        for i in range(5):
            obj_preds = []
            for j in range(3):
                obj_pred = ObjectPrediction(
                    bbox=[10, 10, 50, 50],
                    category_id=0,
                    category_name="cat",
                    score=0.8,
                )
                obj_preds.append(obj_pred)
            
            pred_result = PredictionResult(
                object_prediction_list=obj_preds,
                image=np.zeros((100, 100, 3), dtype=np.uint8)
            )
            predictions.append(pred_result)

        analysis = analyze_prediction_distribution(predictions)

        assert analysis["num_images"] == 5
        assert analysis["total_detections"] == 15
        assert "mean_detections_per_image" in analysis
        assert "class_distribution" in analysis
        assert analysis["class_distribution"]["cat"] == 15

    def test_calculate_prediction_metrics(self):
        from sahi.predict import calculate_prediction_metrics
        from sahi.prediction import ObjectPrediction

        # Create mock predictions
        predictions = [
            ObjectPrediction(
                bbox=[10, 10, 50, 50],
                category_id=0,
                category_name="cat",
                score=0.9,
            )
        ]

        # Create mock ground truth
        ground_truth = [
            {"bbox": [10, 10, 50, 50], "category_id": 0, "category_name": "cat"}
        ]

        metrics = calculate_prediction_metrics(predictions, ground_truth, iou_threshold=0.5)

        assert "true_positives" in metrics
        assert "false_positives" in metrics
        assert "false_negatives" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        # Should be a perfect match
        assert metrics["true_positives"] == 1
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
