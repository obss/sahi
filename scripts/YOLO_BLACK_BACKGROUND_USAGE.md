# YOLO Segmentation Inference with Black Background

This guide shows how to run YOLO segmentation inference using SAHI with results overlayed on a black background.

## Quick Start

### Using the Python Script

The easiest way to run inference with black background is using the provided script:

```bash
python scripts/yolo_sahi_black_background.py \
  --model-path yolov8m-seg.pt \
  --source /path/to/images \
  --output runs/black_background_results \
  --confidence 0.4 \
  --device 0 \
  --tile-size 1024 \
  --overlap 0.5 \
  --iou-threshold 1.0
```

### Your Specific Configuration

Based on your requirements (50% overlap, 1024x1024 tiles, 0.4 confidence, IOU 1.0, device 0):

```bash
python scripts/yolo_sahi_black_background.py \
  --model-path /path/to/your/yolo-seg-model.pt \
  --source /path/to/your/images \
  --output runs/my_results \
  --confidence 0.4 \
  --device 0 \
  --tile-size 1024 \
  --overlap 0.5 \
  --iou-threshold 1.0
```

## Using SAHI CLI Directly

You can also use the SAHI CLI with the new `--visual_black_background` flag:

```bash
sahi predict \
  --model_type ultralytics \
  --model_path yolov8m-seg.pt \
  --model_confidence_threshold 0.4 \
  --model_device 0 \
  --source /path/to/images \
  --slice_height 1024 \
  --slice_width 1024 \
  --overlap_height_ratio 0.5 \
  --overlap_width_ratio 0.5 \
  --postprocess_match_threshold 1.0 \
  --visual_black_background True \
  --project runs/black_background \
  --name exp1
```

## Using Python API

You can also use the Python API directly:

```python
from sahi.predict import predict

result = predict(
    # Model configuration
    model_type="ultralytics",
    model_path="yolov8m-seg.pt",
    model_confidence_threshold=0.4,
    model_device="0",

    # Input/Output
    source="/path/to/images",
    project="runs/black_background",
    name="exp",

    # Slicing configuration
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.5,
    overlap_width_ratio=0.5,

    # Postprocessing
    postprocess_match_threshold=1.0,
    postprocess_type="GREEDYNMM",
    postprocess_match_metric="IOS",

    # Visualization with black background
    visual_black_background=True,

    verbose=1,
    return_dict=True,
)

print(f"Results saved to: {result['export_dir']}")
```

## Advanced Options

### Script Parameters

- `--model-path`: Path to YOLO segmentation model (required)
- `--source`: Folder containing images (required)
- `--output`: Output directory (default: runs/black_background_results)
- `--confidence`: Model confidence threshold (default: 0.4)
- `--device`: Device for inference, e.g., "0" for cuda:0 (default: 0)
- `--tile-size`: Tile size for both width and height (default: 1024)
- `--overlap`: Overlap ratio for both dimensions (default: 0.5)
- `--iou-threshold`: IOU threshold for postprocessing (default: 1.0)
- `--postprocess-type`: Type of postprocessing [NMS, GREEDYNMM, NMM, LSNMS] (default: GREEDYNMM)
- `--postprocess-metric`: Match metric [IOU, IOS] (default: IOS)
- `--no-black-background`: Use original image instead of black background
- `--bbox-thickness`: Bounding box line thickness
- `--text-size`: Label text size
- `--hide-labels`: Don't show class labels
- `--hide-conf`: Don't show confidence scores
- `--export-format`: Export format [png, jpg] (default: png)

## Output Structure

After running inference, your output directory will contain:

```
runs/black_background_results/exp/
├── visuals/           # Visualization images with black background
├── pickles/          # (if --export_pickle) Serialized predictions
└── crops/            # (if --export_crop) Cropped detections
```

## Notes

1. **Black Background**: When enabled, predictions are overlayed on a completely black image, showing only the segmentation masks, bounding boxes, and labels without the original image content.

2. **Model Requirements**: Ensure you're using a YOLO segmentation model (e.g., yolov8m-seg.pt, yolov11m-seg.pt). Regular detection models won't have segmentation masks.

3. **Device**: Use "0" for cuda:0, "1" for cuda:1, "cpu" for CPU inference, or "mps" for Apple Silicon.

4. **IOU Threshold**: Setting to 1.0 means only exact overlaps will be merged. Lower values (e.g., 0.5) are more aggressive in merging overlapping predictions.

5. **Overlap**: 0.5 means 50% overlap between tiles. Higher overlap can improve detection at tile boundaries but increases computation time.

## Examples

### Basic Example (Black Background)
```bash
python scripts/yolo_sahi_black_background.py \
  --model-path yolov8m-seg.pt \
  --source ./demo/images \
  --output ./results
```

### High Resolution with Large Tiles
```bash
python scripts/yolo_sahi_black_background.py \
  --model-path yolov11l-seg.pt \
  --source ./high_res_images \
  --output ./results_hires \
  --tile-size 2048 \
  --overlap 0.3 \
  --confidence 0.5
```

### Use Original Image (Not Black Background)
```bash
python scripts/yolo_sahi_black_background.py \
  --model-path yolov8m-seg.pt \
  --source ./images \
  --output ./results_normal \
  --no-black-background
```

## Troubleshooting

**Issue**: Model not found
- Solution: Ensure the model path is correct and the model file exists

**Issue**: CUDA out of memory
- Solution: Reduce tile size (e.g., --tile-size 512) or use CPU (--device cpu)

**Issue**: No segmentation masks visible
- Solution: Ensure you're using a segmentation model (e.g., yolov8m-seg.pt, not yolov8m.pt)

**Issue**: Too many/too few detections
- Solution: Adjust confidence threshold (--confidence) and IOU threshold (--iou-threshold)
