"""
Simple example demonstrating SAHI video segmentation usage.

This example shows how to use the SAHI video segmentation functionality
programmatically (without CLI).
"""

from pathlib import Path
import sys

# Add parent directory to path to import the script
sys.path.insert(0, str(Path(__file__).parent.parent))

from sahi.auto_model import AutoDetectionModel
from sahi_video_segmentation import process_video, process_image


def example_video_processing():
    """Example: Process a video with SAHI segmentation."""

    # Initialize SAHI model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-seg.pt",  # or "yolov8n-seg.pt"
        device="cuda:0",  # or "cpu"
        confidence_threshold=0.25,
    )

    # Process video
    process_video(
        video_path="input_video.mp4",
        detection_model=detection_model,
        slice_height=1024,  # Tile height
        slice_width=1024,   # Tile width
        overlap_ratio=0.25, # 25% overlap
        output_dir=Path("output/video_results"),
        frame_skip_interval=0,  # Process all frames
        save_output_video=True,
        save_isolated=True,
        save_visualization=False,  # Set True to save individual frame visualizations
        view_video=False,  # Set True to display video in real-time
    )


def example_image_processing():
    """Example: Process a single image with SAHI segmentation."""

    # Initialize SAHI model
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        confidence_threshold=0.25,
    )

    # Process image
    process_image(
        image_path="input_image.jpg",
        detection_model=detection_model,
        slice_height=1024,
        slice_width=1024,
        overlap_ratio=0.25,
        output_dir=Path("output/image_results"),
        save_isolated=True,
        save_visualization=True,
    )


def example_batch_processing():
    """Example: Process multiple images in a directory."""

    # Initialize model once
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        confidence_threshold=0.25,
    )

    # Process all images in a directory
    image_dir = Path("input_images")
    output_dir = Path("output/batch_results")

    for img_path in image_dir.glob("*.jpg"):
        print(f"\nProcessing: {img_path.name}")
        process_image(
            image_path=str(img_path),
            detection_model=detection_model,
            slice_height=1024,
            slice_width=1024,
            overlap_ratio=0.25,
            output_dir=output_dir / img_path.stem,
            save_isolated=True,
            save_visualization=True,
        )


def example_custom_tile_sizes():
    """Example: Using different tile sizes for different scenarios."""

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        confidence_threshold=0.25,
    )

    # Small objects: Use smaller tiles
    process_image(
        image_path="small_objects.jpg",
        detection_model=detection_model,
        slice_height=640,
        slice_width=640,
        overlap_ratio=0.3,  # More overlap for small objects
        output_dir=Path("output/small_objects"),
    )

    # Large objects: Use larger tiles
    process_image(
        image_path="large_objects.jpg",
        detection_model=detection_model,
        slice_height=1280,
        slice_width=1280,
        overlap_ratio=0.2,  # Less overlap for large objects
        output_dir=Path("output/large_objects"),
    )


if __name__ == "__main__":
    # Uncomment the example you want to run

    # Example 1: Process a video
    # example_video_processing()

    # Example 2: Process a single image
    # example_image_processing()

    # Example 3: Batch process multiple images
    # example_batch_processing()

    # Example 4: Custom tile sizes
    # example_custom_tile_sizes()

    print("Please uncomment one of the examples above to run it.")
