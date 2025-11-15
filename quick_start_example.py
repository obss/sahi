"""
Quick Start Example for SAHI Video Tracking Segmentation

This example shows how to quickly get started with SAHI video tracking,
similar to the original YOLO code but with enhanced capabilities.
"""

from pathlib import Path
from sahi_video_tracking_segmentation import SAHITrackedSegmentation

# Original approach (without SAHI, tracking, or batching):
"""
from ultralytics import YOLO
import cv2
import numpy as np

m = YOLO("yolo11n-seg.pt")
res = m.predict()

# Iterate detection results
for r in res:
    img = np.copy(r.orig_img)
    img_name = Path(r.path).stem

    # Iterate each object contour
    for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]
        b_mask = np.zeros(img.shape[:2], np.uint8)

        # Create contour mask
        contour = c.masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
        _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Isolate object with black background
        mask3ch = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
        isolated = cv2.bitwise_and(mask3ch, img)
"""

# Enhanced approach with SAHI + Tracking + Batching:

# Initialize with all the features you requested
sahi_tracker = SAHITrackedSegmentation(
    model_path="yolo11n-seg.pt",  # Same model
    device="cuda:0",               # Use GPU
    conf_threshold=0.25,
    iou_threshold=0.7,
    slice_height=1024,             # 1024x1024 tiles
    slice_width=1024,
    overlap_ratio=0.33,            # 33% overlap (as requested)
    batch_size=4,                  # Process 4 tiles simultaneously
    tracker="bytetrack.yaml",      # Inter-frame tracking
)

# Process a video (handles everything automatically)
sahi_tracker.process_video(
    video_path="input_video.mp4",
    output_dir=Path("output/tracked"),
    save_isolated=True,            # Saves binary masks with black background
    save_visualization=True,       # Saves visualization frames
    view_video=False,              # Set to True to watch in real-time
)

# The script now automatically:
# 1. Splits each frame into 1024x1024 tiles with 33% overlap ✓
# 2. Processes multiple tiles at once (batched inference) ✓
# 3. Tracks objects across frames with unique IDs ✓
# 4. Creates binary masks with black background ✓
# 5. Saves isolated objects as: id{track_id}_{class}_score{score}.png
# 6. Saves binary masks as: id{track_id}_{class}_mask.png

# Output structure:
# output/tracked/
#   ├── video_name_tracked.mp4              # Processed video
#   ├── isolated_objects/                   # Isolated objects per frame
#   │   └── video_name_frame_000000/
#   │       ├── id1_person_score0.85.png    # Object image
#   │       ├── id1_person_mask.png         # Binary mask
#   │       ├── id2_car_score0.92.png
#   │       └── id2_car_mask.png
#   └── visualizations/                     # Frame visualizations
#       └── video_name_frame_000000.jpg

print("Processing complete! Check the output directory for results.")
