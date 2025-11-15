# SAHI Video Tracking Segmentation

A comprehensive solution for accurate video segmentation using SAHI (Slicing Aided Hyper Inference) combined with Ultralytics tracking capabilities.

## Features

✅ **SAHI Overlapping Tiling**
- 1024x1024 tiles with 33% overlap (configurable)
- Accurate detection of small and large objects
- Better performance on high-resolution videos

✅ **Batched Inference**
- Process multiple tiles simultaneously
- Configurable batch size (default: 4)
- Significantly faster than sequential processing

✅ **Inter-frame Tracking**
- Track objects across frames with unique IDs
- ByteTrack or BoTSORT tracker support
- Persistent object identification

✅ **Binary Mask Generation**
- Creates binary masks for each detected object
- Isolates objects with black background
- Saves both RGB isolated images and binary masks

✅ **Video Support**
- Native video processing
- Configurable frame skipping
- Real-time visualization option

## Installation

```bash
pip install ultralytics sahi opencv-python numpy pillow
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from sahi_video_tracking_segmentation import SAHITrackedSegmentation

# Initialize
sahi_tracker = SAHITrackedSegmentation(
    model_path="yolo11n-seg.pt",
    device="cuda:0",
    slice_height=1024,
    slice_width=1024,
    overlap_ratio=0.33,  # 33% overlap
    batch_size=4,        # Process 4 tiles at once
    tracker="bytetrack.yaml",
)

# Process video
sahi_tracker.process_video(
    video_path="input_video.mp4",
    output_dir=Path("output/tracked"),
    save_isolated=True,
    save_visualization=True,
)
```

### Command-Line Usage

```bash
# Basic video processing
python sahi_video_tracking_segmentation.py input_video.mp4

# Custom settings
python sahi_video_tracking_segmentation.py input_video.mp4 \
    --model yolo11x-seg.pt \
    --slice-size 1024 \
    --overlap 0.33 \
    --batch-size 4 \
    --tracker bytetrack.yaml \
    --output-dir output/tracked

# Process image (no tracking, just batched SAHI)
python sahi_video_tracking_segmentation.py large_image.jpg \
    --batch-size 8
```

## Configuration Options

### Model Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | `"yolo11n-seg.pt"` | Path to YOLO segmentation model |
| `device` | `"cuda:0"` | Device for inference (`cuda:0`, `cpu`) |
| `conf_threshold` | `0.25` | Confidence threshold for detections |
| `iou_threshold` | `0.7` | IoU threshold for NMS |

### SAHI Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slice_height` | `1024` | Height of each tile in pixels |
| `slice_width` | `1024` | Width of each tile in pixels |
| `overlap_ratio` | `0.33` | Overlap ratio between tiles (33%) |

### Performance Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | `4` | Number of tiles to process simultaneously |

### Tracking Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tracker` | `"bytetrack.yaml"` | Tracker type (`bytetrack.yaml` or `botsort.yaml`) |

## Advanced Usage

### High-Accuracy Settings (Dense Scenes)

```python
sahi_tracker = SAHITrackedSegmentation(
    model_path="yolo11x-seg.pt",  # Larger, more accurate model
    device="cuda:0",
    conf_threshold=0.3,           # Higher confidence
    iou_threshold=0.5,            # Stricter NMS
    slice_height=1024,
    slice_width=1024,
    overlap_ratio=0.4,            # More overlap for better detection
    batch_size=2,                 # Smaller batch for larger model
    tracker="botsort.yaml",       # More robust tracker
)
```

### Fast Processing (Real-time Applications)

```python
sahi_tracker = SAHITrackedSegmentation(
    model_path="yolo11n-seg.pt",  # Smallest/fastest model
    device="cuda:0",
    conf_threshold=0.2,
    iou_threshold=0.7,
    slice_height=640,             # Smaller tiles
    slice_width=640,
    overlap_ratio=0.2,            # Less overlap
    batch_size=8,                 # Larger batch
    tracker="bytetrack.yaml",     # Faster tracker
)

sahi_tracker.process_video(
    video_path="simple_scene.mp4",
    output_dir=Path("output/fast"),
    frame_skip_interval=2,        # Skip every 2 frames
)
```

### Programmatic Access to Tracking Data

```python
import cv2
import numpy as np

sahi_tracker = SAHITrackedSegmentation(
    model_path="yolo11n-seg.pt",
    device="cuda:0",
    slice_height=1024,
    slice_width=1024,
    overlap_ratio=0.33,
    batch_size=4,
)

cap = cv2.VideoCapture("input_video.mp4")
frame_idx = 0
tracking_data = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    result = sahi_tracker.process_frame_with_tracking(
        frame=frame,
        frame_idx=frame_idx,
        output_dir=None,
        save_isolated=False,
        save_visualization=False,
    )

    # Access tracking information
    for obj in result["tracked_objects"]:
        track_id = obj["track_id"]

        if track_id not in tracking_data:
            tracking_data[track_id] = {
                "category": obj["category"],
                "first_seen": frame_idx,
                "positions": [],
                "scores": [],
            }

        tracking_data[track_id]["positions"].append(obj["centroid"])
        tracking_data[track_id]["scores"].append(obj["score"])

    print(f"Frame {frame_idx}: {result['num_objects']} objects, "
          f"{result['tiles_processed']} tiles")

    frame_idx += 1

cap.release()

# Analyze tracking data
for track_id, data in tracking_data.items():
    duration = len(data["positions"])
    avg_score = np.mean(data["scores"])
    print(f"Track ID {track_id}: {data['category']}, "
          f"Duration: {duration} frames, Avg Score: {avg_score:.2f}")
```

## Output Structure

```
output/tracked/
├── video_name_tracked.mp4              # Processed video with tracking visualization
├── isolated_objects/                   # Isolated objects per frame
│   └── video_name_frame_000000/
│       ├── id1_person_score0.85.png    # RGB isolated object with track ID
│       ├── id1_person_mask.png         # Binary mask
│       ├── id2_car_score0.92.png
│       └── id2_car_mask.png
└── visualizations/                     # Individual frame visualizations
    └── video_name_frame_000000.jpg
```

## Key Features Explained

### 1. SAHI Overlapping Tiling

SAHI splits each frame into overlapping tiles to improve detection accuracy:
- **Small objects**: Detected more accurately within tiles
- **Large objects**: Captured across multiple overlapping tiles
- **Edge cases**: Overlap ensures objects at tile boundaries are detected

### 2. Batched Inference

Instead of processing tiles one-by-one, batched inference processes multiple tiles simultaneously:
- **Performance**: Significantly faster than sequential processing
- **GPU Utilization**: Better GPU usage with parallel processing
- **Configurable**: Adjust batch size based on available GPU memory

### 3. Inter-frame Tracking

Ultralytics tracking maintains object identities across frames:
- **Unique IDs**: Each object gets a persistent tracking ID
- **Trajectory Analysis**: Track object movement over time
- **Object Counting**: Count objects entering/exiting scene

### 4. Binary Mask Generation

For each detected object:
- **Binary Mask**: White foreground, black background
- **Isolated Object**: Original RGB image with black background
- **Contour-based**: Uses precise segmentation masks

## Performance Tips

1. **Batch Size**: Increase for more powerful GPUs (e.g., 8 or 16 for A100)
2. **Overlap Ratio**: Higher overlap (0.4-0.5) for small objects, lower (0.2-0.25) for speed
3. **Tile Size**: Smaller tiles (640x640) for faster processing, larger (1024x1024) for accuracy
4. **Frame Skip**: Skip frames for faster processing if real-time not required
5. **Model Size**: Use smaller models (yolo11n) for speed, larger (yolo11x) for accuracy

## Comparison with Standard YOLO

| Feature | Standard YOLO | SAHI Tracking |
|---------|--------------|---------------|
| Small object detection | Limited | Excellent (tiling) |
| Large image support | Memory issues | Handles any size |
| Processing speed | Fast | Fast (batched) |
| Object tracking | Manual | Built-in |
| Batch processing | Single image | Multiple tiles |

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size`
- Use smaller `slice_height` and `slice_width`
- Use a smaller model (e.g., `yolo11n-seg.pt`)

### Slow Processing
- Increase `batch_size` if you have GPU memory
- Reduce `overlap_ratio`
- Use smaller tiles
- Skip frames with `frame_skip_interval`

### Poor Detection
- Increase `overlap_ratio` (e.g., 0.4 or 0.5)
- Use larger model (e.g., `yolo11x-seg.pt`)
- Increase `slice_height` and `slice_width`
- Lower `conf_threshold`

### Tracking Issues
- Switch tracker (`bytetrack.yaml` ↔ `botsort.yaml`)
- Adjust `iou_threshold` for NMS
- Process more frames (reduce `frame_skip_interval`)

## Examples

See `examples/sahi_video_tracking_example.py` for comprehensive examples including:
- Basic video tracking
- Image processing with batched inference
- Custom settings for different scenarios
- Programmatic access to tracking data

## License

This code is provided as-is for use with the SAHI and Ultralytics libraries.

## References

- [SAHI: Slicing Aided Hyper Inference](https://github.com/obss/sahi)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
