"""
SAHI Video Segmentation Script

This script combines SAHI (Slicing Aided Hyper Inference) with YOLO segmentation
to process videos and images with tiled inference for accurate segmentation.

Features:
- 1024x1024 tiles with 25% overlap
- Support for both images and videos
- Individual object mask extraction
- Saves isolated objects with black background
"""

from pathlib import Path
from typing import Optional, Union
import argparse

import cv2
import numpy as np
from PIL import Image
from sahi.auto_model import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import get_video_reader, VIDEO_EXTENSIONS


def process_frame_with_sahi(
    frame: Union[np.ndarray, Image.Image],
    detection_model: AutoDetectionModel,
    slice_height: int = 1024,
    slice_width: int = 1024,
    overlap_ratio: float = 0.25,
    output_dir: Optional[Path] = None,
    frame_name: str = "frame",
    save_isolated: bool = True,
    save_visualization: bool = True,
) -> dict:
    """
    Process a single frame with SAHI for segmentation.

    Args:
        frame: Input frame (numpy array or PIL Image)
        detection_model: SAHI AutoDetectionModel instance
        slice_height: Height of each tile (default: 1024)
        slice_width: Width of each tile (default: 1024)
        overlap_ratio: Overlap ratio for tiles (default: 0.25 = 25%)
        output_dir: Directory to save outputs
        frame_name: Name prefix for saved files
        save_isolated: Whether to save isolated objects
        save_visualization: Whether to save visualization

    Returns:
        Dictionary with prediction results and statistics
    """
    # Convert PIL Image to numpy array if needed
    if isinstance(frame, Image.Image):
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

    # Get sliced predictions
    prediction_result = get_sliced_prediction(
        image=frame,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_ratio,
        overlap_width_ratio=overlap_ratio,
        perform_standard_pred=True,  # Also run on full image
        postprocess_type="GREEDYNMM",  # Greedy Non-Maximum Merging
        postprocess_match_metric="IOS",  # Intersection Over Smaller
        postprocess_match_threshold=0.5,
    )

    img_copy = np.copy(frame)
    num_objects = len(prediction_result.object_prediction_list)

    # Process each detected object
    isolated_objects = []
    for obj_idx, obj_pred in enumerate(prediction_result.object_prediction_list):
        category_name = obj_pred.category.name
        score = obj_pred.score.value
        bbox = obj_pred.bbox.to_xyxy()

        # Check if segmentation mask exists
        if obj_pred.mask is not None:
            # Get binary mask from SAHI prediction
            binary_mask = obj_pred.mask.bool_mask.astype(np.uint8) * 255

            # Create contour mask
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Create a fresh mask for this object
            object_mask = np.zeros(frame.shape[:2], np.uint8)
            cv2.drawContours(object_mask, contours, -1, (255, 255, 255), cv2.FILLED)

            # Isolate object with black background
            mask_3ch = cv2.cvtColor(object_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask_3ch, img_copy)

            isolated_objects.append({
                "image": isolated,
                "mask": object_mask,
                "category": category_name,
                "score": score,
                "bbox": bbox,
            })

            # Save isolated object
            if save_isolated and output_dir:
                isolated_dir = output_dir / "isolated_objects" / frame_name
                isolated_dir.mkdir(parents=True, exist_ok=True)

                isolated_path = isolated_dir / f"{category_name}_{obj_idx}_score{score:.2f}.png"
                cv2.imwrite(str(isolated_path), isolated)

        elif obj_pred.segmentation:
            # Fallback: use polygon segmentation if mask not available
            b_mask = np.zeros(frame.shape[:2], np.uint8)

            # Convert segmentation polygon to contour
            for segment in obj_pred.segmentation:
                polygon = np.array(segment).reshape(-1, 1, 2).astype(np.int32)
                cv2.drawContours(b_mask, [polygon], -1, (255, 255, 255), cv2.FILLED)

            # Isolate object
            mask_3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            isolated = cv2.bitwise_and(mask_3ch, img_copy)

            isolated_objects.append({
                "image": isolated,
                "mask": b_mask,
                "category": category_name,
                "score": score,
                "bbox": bbox,
            })

            # Save isolated object
            if save_isolated and output_dir:
                isolated_dir = output_dir / "isolated_objects" / frame_name
                isolated_dir.mkdir(parents=True, exist_ok=True)

                isolated_path = isolated_dir / f"{category_name}_{obj_idx}_score{score:.2f}.png"
                cv2.imwrite(str(isolated_path), isolated)

    # Save visualization
    if save_visualization and output_dir:
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        prediction_result.export_visuals(
            export_dir=str(vis_dir),
            file_name=frame_name,
            text_size=0.7,
            hide_conf=False,
        )

    return {
        "prediction_result": prediction_result,
        "isolated_objects": isolated_objects,
        "num_objects": num_objects,
    }


def process_video(
    video_path: str,
    detection_model: AutoDetectionModel,
    slice_height: int = 1024,
    slice_width: int = 1024,
    overlap_ratio: float = 0.25,
    output_dir: Path = Path("output"),
    frame_skip_interval: int = 0,
    save_output_video: bool = True,
    save_isolated: bool = True,
    save_visualization: bool = False,
    view_video: bool = False,
) -> None:
    """
    Process video with SAHI for segmentation.

    Args:
        video_path: Path to input video
        detection_model: SAHI AutoDetectionModel instance
        slice_height: Height of each tile (default: 1024)
        slice_width: Width of each tile (default: 1024)
        overlap_ratio: Overlap ratio for tiles (default: 0.25)
        output_dir: Directory to save outputs
        frame_skip_interval: Number of frames to skip (0 = process all)
        save_output_video: Whether to save processed video
        save_isolated: Whether to save isolated objects
        save_visualization: Whether to save frame visualizations
        view_video: Whether to display video in real-time
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get video reader
    read_video_frame, output_video_writer, video_name, num_frames = get_video_reader(
        source=video_path,
        save_dir=str(output_dir),
        frame_skip_interval=frame_skip_interval,
        export_visual=save_output_video,
        view_visual=view_video,
    )

    print(f"Processing video: {video_name}")
    print(f"Total frames: {num_frames}")
    print(f"Tile size: {slice_width}x{slice_height} with {overlap_ratio*100}% overlap")

    frame_idx = 0
    total_objects = 0

    try:
        for frame_pil in read_video_frame:
            # Convert PIL to numpy array (BGR for OpenCV)
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            # Process frame
            frame_name = f"{video_name}_frame_{frame_idx:06d}"
            result = process_frame_with_sahi(
                frame=frame,
                detection_model=detection_model,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_ratio=overlap_ratio,
                output_dir=output_dir,
                frame_name=frame_name,
                save_isolated=save_isolated,
                save_visualization=save_visualization,
            )

            total_objects += result["num_objects"]

            # Write frame to output video with visualizations
            if save_output_video and output_video_writer is not None:
                # Get visualization image
                vis_img = result["prediction_result"].image
                if isinstance(vis_img, Image.Image):
                    vis_img = cv2.cvtColor(np.array(vis_img), cv2.COLOR_RGB2BGR)
                output_video_writer.write(vis_img)

            if (frame_idx + 1) % 10 == 0:
                print(f"Processed {frame_idx + 1}/{num_frames} frames, detected {result['num_objects']} objects")

            frame_idx += 1

    finally:
        # Release video writer
        if output_video_writer is not None:
            output_video_writer.release()

    print(f"\nProcessing complete!")
    print(f"Total frames processed: {frame_idx}")
    print(f"Total objects detected: {total_objects}")
    print(f"Output directory: {output_dir}")


def process_image(
    image_path: str,
    detection_model: AutoDetectionModel,
    slice_height: int = 1024,
    slice_width: int = 1024,
    overlap_ratio: float = 0.25,
    output_dir: Path = Path("output"),
    save_isolated: bool = True,
    save_visualization: bool = True,
) -> None:
    """
    Process single image with SAHI for segmentation.

    Args:
        image_path: Path to input image
        detection_model: SAHI AutoDetectionModel instance
        slice_height: Height of each tile (default: 1024)
        slice_width: Width of each tile (default: 1024)
        overlap_ratio: Overlap ratio for tiles (default: 0.25)
        output_dir: Directory to save outputs
        save_isolated: Whether to save isolated objects
        save_visualization: Whether to save visualization
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image_name = Path(image_path).stem

    print(f"Processing image: {image_name}")
    print(f"Image size: {image.shape[1]}x{image.shape[0]}")
    print(f"Tile size: {slice_width}x{slice_height} with {overlap_ratio*100}% overlap")

    # Process image
    result = process_frame_with_sahi(
        frame=image,
        detection_model=detection_model,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_ratio=overlap_ratio,
        output_dir=output_dir,
        frame_name=image_name,
        save_isolated=save_isolated,
        save_visualization=save_visualization,
    )

    print(f"\nProcessing complete!")
    print(f"Objects detected: {result['num_objects']}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="SAHI Video/Image Segmentation with YOLO"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to input video or image",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n-seg.pt",
        help="Path to YOLO segmentation model (default: yolo11n-seg.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--slice-size",
        type=int,
        default=1024,
        help="Size of tiles (default: 1024)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap ratio (default: 0.25 = 25%%)",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use (default: cuda:0)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Frame skip interval for videos (default: 0 = process all frames)",
    )
    parser.add_argument(
        "--no-isolated",
        action="store_true",
        help="Don't save isolated objects",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="Don't save visualizations",
    )
    parser.add_argument(
        "--view-video",
        action="store_true",
        help="Display video in real-time during processing",
    )

    args = parser.parse_args()

    # Load SAHI model
    print(f"Loading model: {args.model}")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=args.model,
        device=args.device,
        confidence_threshold=args.conf_threshold,
    )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Check if input is video or image
    if input_path.suffix.lower() in VIDEO_EXTENSIONS:
        # Process video
        process_video(
            video_path=str(input_path),
            detection_model=detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_ratio=args.overlap,
            output_dir=output_dir,
            frame_skip_interval=args.frame_skip,
            save_output_video=True,
            save_isolated=not args.no_isolated,
            save_visualization=not args.no_visualization,
            view_video=args.view_video,
        )
    else:
        # Process image
        process_image(
            image_path=str(input_path),
            detection_model=detection_model,
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_ratio=args.overlap,
            output_dir=output_dir,
            save_isolated=not args.no_isolated,
            save_visualization=not args.no_visualization,
        )


if __name__ == "__main__":
    main()
