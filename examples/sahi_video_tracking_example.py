"""
SAHI Video Tracking Segmentation Examples

This example demonstrates how to use SAHI with Ultralytics tracking
and batched inference for video and image segmentation.

Key features demonstrated:
- 1024x1024 tiles with 25% overlap
- Batched tile processing for faster inference
- Inter-frame object tracking
- Isolated object extraction with tracking IDs
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sahi_video_tracking_segmentation import SAHITrackedSegmentation


def example_video_tracking():
    """
    Example: Process a video with SAHI tiling, batched inference, and tracking.

    This will:
    1. Split each frame into 1024x1024 tiles with 25% overlap
    2. Process multiple tiles simultaneously (batched inference)
    3. Track objects across frames with unique IDs
    4. Save isolated objects with tracking information
    """

    # Initialize SAHI Tracked Segmentation
    sahi_tracker = SAHITrackedSegmentation(
        model_path="yolo11n-seg.pt",  # or "yolov8x-seg.pt" for better accuracy
        device="cuda:0",  # or "cpu" if no GPU
        conf_threshold=0.25,
        iou_threshold=0.7,
        slice_height=1024,  # Tile height
        slice_width=1024,   # Tile width
        overlap_ratio=0.25,  # 25% overlap
        batch_size=4,  # Process 4 tiles at once (adjust based on GPU memory)
        tracker="bytetrack.yaml",  # or "botsort.yaml"
    )

    # Process video
    sahi_tracker.process_video(
        video_path="input_video.mp4",
        output_dir=Path("output/tracked_video"),
        frame_skip_interval=0,  # Process all frames
        save_output_video=True,  # Save video with tracking visualization
        save_isolated=True,  # Save isolated objects with track IDs
        save_visualization=True,  # Save individual frame visualizations
        view_video=False,  # Set to True to display in real-time
    )

    print("\nOutput structure:")
    print("output/tracked_video/")
    print("  ├── video_name.mp4                    # Processed video with tracking")
    print("  ├── isolated_objects/                 # Isolated objects per frame")
    print("  │   └── video_name_frame_000000/")
    print("  │       ├── id1_person_score0.85.png  # Object with track ID 1")
    print("  │       └── id2_car_score0.92.png     # Object with track ID 2")
    print("  └── visualizations/                   # Individual frame visualizations")
    print("      └── video_name_frame_000000.jpg")


def example_image_batched():
    """
    Example: Process an image with SAHI tiling and batched inference.

    For images, tracking is not used, but batched inference speeds up
    processing by running multiple tiles simultaneously.
    """

    sahi_tracker = SAHITrackedSegmentation(
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        conf_threshold=0.25,
        iou_threshold=0.7,
        slice_height=1024,
        slice_width=1024,
        overlap_ratio=0.25,
        batch_size=8,  # Process 8 tiles at once for images
    )

    # Process image
    sahi_tracker.process_image(
        image_path="large_image.jpg",
        output_dir=Path("output/batched_image"),
        save_isolated=True,
        save_visualization=True,
    )


def example_custom_settings():
    """
    Example: Custom settings for different scenarios.
    """

    # High-accuracy tracking for dense scenes
    sahi_tracker_accurate = SAHITrackedSegmentation(
        model_path="yolo11x-seg.pt",  # Larger model
        device="cuda:0",
        conf_threshold=0.3,  # Higher confidence
        iou_threshold=0.5,  # Stricter NMS
        slice_height=1024,
        slice_width=1024,
        overlap_ratio=0.3,  # More overlap for better detection
        batch_size=2,  # Smaller batch for larger model
        tracker="botsort.yaml",  # More robust tracker
    )

    # Fast processing for real-time applications
    sahi_tracker_fast = SAHITrackedSegmentation(
        model_path="yolo11n-seg.pt",  # Smallest/fastest model
        device="cuda:0",
        conf_threshold=0.2,
        iou_threshold=0.7,
        slice_height=640,  # Smaller tiles
        slice_width=640,
        overlap_ratio=0.2,  # Less overlap
        batch_size=8,  # Larger batch
        tracker="bytetrack.yaml",  # Faster tracker
    )

    # Process with accurate settings
    sahi_tracker_accurate.process_video(
        video_path="dense_crowd.mp4",
        output_dir=Path("output/accurate_tracking"),
        frame_skip_interval=0,
    )

    # Process with fast settings
    sahi_tracker_fast.process_video(
        video_path="simple_scene.mp4",
        output_dir=Path("output/fast_tracking"),
        frame_skip_interval=2,  # Skip every 2 frames for faster processing
    )


def example_programmatic_usage():
    """
    Example: Programmatic access to tracking data.

    This shows how to get tracking information for custom processing.
    """

    import cv2
    import numpy as np

    sahi_tracker = SAHITrackedSegmentation(
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        slice_height=1024,
        slice_width=1024,
        overlap_ratio=0.25,
        batch_size=4,
    )

    # Load video
    cap = cv2.VideoCapture("input_video.mp4")

    frame_idx = 0
    tracking_data = {}  # Store tracking information

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        result = sahi_tracker.process_frame_with_tracking(
            frame=frame,
            frame_idx=frame_idx,
            output_dir=None,  # Don't save to disk
            save_isolated=False,
            save_visualization=False,
        )

        # Access tracking information
        for obj in result["tracked_objects"]:
            track_id = obj["track_id"]

            # Initialize tracking data for new objects
            if track_id not in tracking_data:
                tracking_data[track_id] = {
                    "category": obj["category"],
                    "first_seen": frame_idx,
                    "positions": [],
                    "scores": [],
                }

            # Store position and score
            tracking_data[track_id]["positions"].append(obj["centroid"])
            tracking_data[track_id]["scores"].append(obj["score"])

        print(f"Frame {frame_idx}: {result['num_objects']} objects, {result['tiles_processed']} tiles")

        frame_idx += 1

    cap.release()

    # Analyze tracking data
    print("\n=== Tracking Summary ===")
    for track_id, data in tracking_data.items():
        duration = len(data["positions"])
        avg_score = np.mean(data["scores"])
        print(f"Track ID {track_id}:")
        print(f"  Category: {data['category']}")
        print(f"  Duration: {duration} frames")
        print(f"  Avg Score: {avg_score:.2f}")
        print(f"  First seen: Frame {data['first_seen']}")


def example_comparing_your_script():
    """
    Example: Using this script similar to your original code.

    Your original code:
    ```python
    from ultralytics import YOLO
    m = YOLO("yolo11n-seg.pt")
    res = m.predict()

    for r in res:
        for ci, c in enumerate(r):
            label = c.names[c.boxes.cls.tolist().pop()]
            contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
            # ... isolate object
    ```

    Enhanced with SAHI + Tracking:
    """

    # Initialize (similar to YOLO("model.pt"))
    sahi_tracker = SAHITrackedSegmentation(
        model_path="yolo11n-seg.pt",
        device="cuda:0",
        slice_height=1024,
        slice_width=1024,
        overlap_ratio=0.25,
        batch_size=4,
    )

    # Process video (instead of m.predict())
    sahi_tracker.process_video(
        video_path="input_video.mp4",
        output_dir=Path("output/enhanced"),
        save_isolated=True,  # Automatically saves isolated objects
        save_visualization=True,
    )

    # The script now:
    # 1. Handles videos natively
    # 2. Uses SAHI tiling for better detection
    # 3. Processes tiles in batches
    # 4. Tracks objects across frames
    # 5. Automatically extracts and saves isolated objects


if __name__ == "__main__":
    print("SAHI Video Tracking Segmentation Examples")
    print("=" * 50)
    print("\nAvailable examples:")
    print("1. example_video_tracking()       - Basic video tracking")
    print("2. example_image_batched()        - Image with batched inference")
    print("3. example_custom_settings()      - Custom accuracy/speed settings")
    print("4. example_programmatic_usage()   - Access tracking data programmatically")
    print("5. example_comparing_your_script() - Compare with original script")
    print("\nUncomment the example you want to run.")

    # Uncomment to run examples:
    # example_video_tracking()
    # example_image_batched()
    # example_custom_settings()
    # example_programmatic_usage()
    # example_comparing_your_script()
