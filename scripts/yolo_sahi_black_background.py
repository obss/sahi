#!/usr/bin/env python3
"""
YOLO Segmentation Inference with SAHI on Black Background

This script performs YOLO segmentation inference on a folder of images using SAHI
(Slicing Aided Hyper Inference) and overlays the results on a black background.

Example usage:
    python scripts/yolo_sahi_black_background.py \
        --model-path yolov8m-seg.pt \
        --source /path/to/images \
        --output runs/black_background_results

Requirements:
    - SAHI installed with ultralytics support
    - YOLO segmentation model weights
    - Source folder containing images
"""

import argparse
from pathlib import Path

from sahi.predict import predict


def main():
    parser = argparse.ArgumentParser(
        description="Run YOLO segmentation inference with SAHI on black background"
    )

    # Model parameters
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to YOLO segmentation model weights (e.g., yolov8m-seg.pt)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="ultralytics",
        help="Model type (default: ultralytics)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Model confidence threshold (default: 0.4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use for inference (default: 0 for cuda:0)",
    )

    # Source and output
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source folder containing images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/black_background_results",
        help="Output directory (default: runs/black_background_results)",
    )

    # Slicing parameters
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for both width and height (default: 1024)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio for both width and height (default: 0.5)",
    )

    # Postprocessing parameters
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=1.0,
        help="IOU threshold for postprocessing (default: 1.0)",
    )
    parser.add_argument(
        "--postprocess-type",
        type=str,
        default="GREEDYNMM",
        choices=["NMS", "GREEDYNMM", "NMM", "LSNMS"],
        help="Postprocessing type (default: GREEDYNMM)",
    )
    parser.add_argument(
        "--postprocess-metric",
        type=str,
        default="IOS",
        choices=["IOU", "IOS"],
        help="Postprocessing match metric (default: IOS)",
    )

    # Visualization parameters
    parser.add_argument(
        "--no-black-background",
        action="store_true",
        help="Disable black background and use original image instead",
    )
    parser.add_argument(
        "--bbox-thickness",
        type=int,
        default=None,
        help="Bounding box thickness (default: auto)",
    )
    parser.add_argument(
        "--text-size",
        type=float,
        default=None,
        help="Label text size (default: auto)",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="Hide class labels on visualizations",
    )
    parser.add_argument(
        "--hide-conf",
        action="store_true",
        help="Hide confidence scores on visualizations",
    )
    parser.add_argument(
        "--export-format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Export format for visualizations (default: png)",
    )

    args = parser.parse_args()

    # Validate source
    if not Path(args.source).exists():
        raise FileNotFoundError(f"Source path does not exist: {args.source}")

    # Validate model path
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")

    print("=" * 80)
    print("YOLO Segmentation Inference with SAHI - Black Background Mode")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Source: {args.source}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"Confidence: {args.confidence}")
    print(f"Tile Size: {args.tile_size}x{args.tile_size}")
    print(f"Overlap: {args.overlap * 100}%")
    print(f"IOU Threshold: {args.iou_threshold}")
    print(f"Black Background: {not args.no_black_background}")
    print("=" * 80)
    print()

    # Run prediction
    result = predict(
        # Model parameters
        model_type=args.model_type,
        model_path=args.model_path,
        model_confidence_threshold=args.confidence,
        model_device=args.device,

        # Source and output
        source=args.source,
        project=args.output,
        name="exp",

        # Slicing parameters
        slice_height=args.tile_size,
        slice_width=args.tile_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,

        # Postprocessing parameters
        postprocess_type=args.postprocess_type,
        postprocess_match_metric=args.postprocess_metric,
        postprocess_match_threshold=args.iou_threshold,

        # Visualization parameters
        visual_black_background=not args.no_black_background,
        visual_bbox_thickness=args.bbox_thickness,
        visual_text_size=args.text_size,
        visual_hide_labels=args.hide_labels,
        visual_hide_conf=args.hide_conf,
        visual_export_format=args.export_format,

        # Other parameters
        verbose=1,
        return_dict=True,
    )

    print()
    print("=" * 80)
    print("Inference Complete!")
    print("=" * 80)
    if result and "export_dir" in result:
        print(f"Results saved to: {result['export_dir']}")
        print(f"  - Visualizations: {result['export_dir']}/visuals")
    print("=" * 80)


if __name__ == "__main__":
    main()
