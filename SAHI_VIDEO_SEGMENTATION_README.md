# SAHI Video Segmentation

This script combines SAHI (Slicing Aided Hyper Inference) with YOLO segmentation models to process videos and images with tiled inference for accurate object segmentation.

## Features

- **Tiled Inference**: Process large images/videos using 1024x1024 tiles with 25% overlap by default
- **Video Support**: Process videos frame-by-frame with segmentation
- **Individual Object Extraction**: Save each detected object as an isolated image with black background
- **Flexible Configuration**: Customize tile size, overlap, confidence threshold, and more
- **Batch Processing**: Process multiple images or video frames efficiently

## Installation

Make sure you have SAHI and dependencies installed:

```bash
pip install sahi ultralytics opencv-python numpy pillow
```

## Quick Start

### Command Line Interface

#### Process a Video

```bash
python sahi_video_segmentation.py input_video.mp4 \
    --model yolo11n-seg.pt \
    --output-dir output/video \
    --slice-size 1024 \
    --overlap 0.25 \
    --conf-threshold 0.25
```

#### Process an Image

```bash
python sahi_video_segmentation.py input_image.jpg \
    --model yolo11n-seg.pt \
    --output-dir output/image \
    --slice-size 1024 \
    --overlap 0.25
```

### Python API

```python
from pathlib import Path
from sahi.auto_model import AutoDetectionModel
from sahi_video_segmentation import process_video, process_image

# Initialize model
detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo11n-seg.pt",
    device="cuda:0",
    confidence_threshold=0.25,
)

# Process video
process_video(
    video_path="input.mp4",
    detection_model=detection_model,
    slice_height=1024,
    slice_width=1024,
    overlap_ratio=0.25,
    output_dir=Path("output"),
)

# Process image
process_image(
    image_path="input.jpg",
    detection_model=detection_model,
    slice_height=1024,
    slice_width=1024,
    overlap_ratio=0.25,
    output_dir=Path("output"),
)
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `input` | str | - | Path to input video or image (required) |
| `--model` | str | `yolo11n-seg.pt` | Path to YOLO segmentation model |
| `--output-dir` | str | `output` | Output directory |
| `--slice-size` | int | `1024` | Size of tiles (width and height) |
| `--overlap` | float | `0.25` | Overlap ratio (0.25 = 25%) |
| `--conf-threshold` | float | `0.25` | Confidence threshold for detections |
| `--device` | str | `cuda:0` | Device to use (cuda:0, cpu, etc.) |
| `--frame-skip` | int | `0` | Frame skip interval (0 = process all) |
| `--no-isolated` | flag | False | Don't save isolated objects |
| `--no-visualization` | flag | False | Don't save visualizations |
| `--view-video` | flag | False | Display video during processing |

## Output Structure

```
output/
├── visualizations/              # Visualization images with bboxes and masks
│   └── image_name.png
├── isolated_objects/            # Individual objects with black background
│   └── image_name/
│       ├── person_0_score0.95.png
│       ├── car_1_score0.87.png
│       └── ...
└── output_video.mp4            # Processed video (for video inputs)
```

## Examples

### 1. Process Video with Default Settings

```bash
python sahi_video_segmentation.py my_video.mp4
```

### 2. Process Video with Custom Tile Size

```bash
python sahi_video_segmentation.py my_video.mp4 \
    --slice-size 640 \
    --overlap 0.3 \
    --output-dir results/640x640
```

### 3. Process Every 5th Frame

```bash
python sahi_video_segmentation.py my_video.mp4 \
    --frame-skip 5 \
    --output-dir results/skip5
```

### 4. Process Image with High Confidence

```bash
python sahi_video_segmentation.py image.jpg \
    --conf-threshold 0.5 \
    --model yolo11m-seg.pt
```

### 5. View Video in Real-Time

```bash
python sahi_video_segmentation.py my_video.mp4 \
    --view-video \
    --no-isolated
```

## Tile Size Recommendations

| Image Resolution | Recommended Tile Size | Overlap |
|-----------------|----------------------|---------|
| < 1920x1080 | 640x640 | 0.2-0.3 |
| 1920x1080 - 4K | 1024x1024 | 0.25 |
| 4K - 8K | 1280x1280 or 1536x1536 | 0.2-0.25 |
| > 8K | 2048x2048 | 0.15-0.2 |

**Note**: Smaller objects benefit from smaller tiles and higher overlap (0.3-0.4), while larger objects work better with larger tiles and less overlap (0.15-0.25).

## Supported Models

Any YOLO segmentation model from Ultralytics:

- `yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`, `yolo11l-seg.pt`, `yolo11x-seg.pt`
- `yolov8n-seg.pt`, `yolov8s-seg.pt`, `yolov8m-seg.pt`, `yolov8l-seg.pt`, `yolov8x-seg.pt`
- Custom trained segmentation models

## Supported Video Formats

`.mp4`, `.mkv`, `.flv`, `.avi`, `.ts`, `.mpg`, `.mov`, `.wmv`

## How It Works

1. **Slicing**: The input frame/image is divided into overlapping tiles (e.g., 1024x1024 with 25% overlap)
2. **Inference**: YOLO segmentation model runs on each tile
3. **Merging**: Predictions from overlapping tiles are merged using Greedy Non-Maximum Merging (GREEDYNMM)
4. **Full Image Prediction**: Optionally runs prediction on the full image for better context
5. **Extraction**: Individual objects are isolated using their segmentation masks
6. **Output**: Saves visualizations, isolated objects, and processed video

## Performance Tips

1. **GPU Acceleration**: Use `--device cuda:0` for faster processing
2. **Frame Skipping**: For long videos, use `--frame-skip N` to process every Nth frame
3. **Disable Outputs**: Use `--no-isolated` and `--no-visualization` to speed up processing
4. **Batch Processing**: Process multiple files by calling the Python API in a loop
5. **Model Selection**: Smaller models (n, s) are faster; larger models (l, x) are more accurate

## Troubleshooting

### Out of Memory Error
- Reduce `--slice-size` (try 640 or 512)
- Use a smaller model (e.g., `yolo11n-seg.pt`)
- Process on CPU with `--device cpu`

### Low Detection Rate
- Increase overlap: `--overlap 0.3` or `0.4`
- Lower confidence threshold: `--conf-threshold 0.15`
- Use a larger/better model

### Slow Processing
- Increase tile size: `--slice-size 1280`
- Skip frames: `--frame-skip 2`
- Use smaller model or GPU acceleration

## Original Script Comparison

This SAHI-enhanced version improves upon the original script by:

1. ✅ **Tiled Inference**: Handles large images/videos that would otherwise miss small objects
2. ✅ **Video Support**: Processes videos frame-by-frame with proper video I/O
3. ✅ **Better Merging**: Uses GREEDYNMM to handle overlapping predictions
4. ✅ **Flexible Configuration**: Command-line and programmatic API
5. ✅ **Maintained Functionality**: Still extracts individual objects with black background like the original

## Credits

- **SAHI**: https://github.com/obss/sahi
- **Ultralytics YOLO**: https://github.com/ultralytics/ultralytics

## License

This script follows the same license as the SAHI project.
