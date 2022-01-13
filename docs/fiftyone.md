# Fiftyone Utilities

## <div align="center">FiftyOne Utilities</div>

<details open>
<summary>
<big><b>Explore COCO dataset via FiftyOne app:</b></big>
</summary>

For supported version: `pip install fiftyone>=0.14.2<0.15.0`

```python
from sahi.utils.fiftyone import launch_fiftyone_app

# launch fiftyone app:
session = launch_fiftyone_app(coco_image_dir, coco_json_path)

# close fiftyone app:
session.close()
```

</details>

<details closed>
<summary>
<big><b>Convert predictions to FiftyOne detection:</b></big>
</summary>

```python
from sahi import get_sliced_prediction

# perform sliced prediction
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

# convert detections into fiftyone detection format
fiftyone_detections = result.to_fiftyone_detections()
```

</details>

<details closed>
<summary>
<big><b>Explore detection results in Fiftyone UI:</b></big>
</summary>

```bash
sahi coco fifityone --image_dir dir/to/images --dataset_json_path dataset.json cocoresult1.json cocoresult2.json
```

will open a FiftyOne app that visualizes the given dataset and 2 detection results.

Specify IOU threshold for FP/TP by `--iou_threshold 0.5` argument

</details>