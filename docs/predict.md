---
tags:
  - inference
  - slicing
  - batch-inference
  - visualization
  - object-detection
  - small-object-detection
---

# Prediction Utilities

## Sliced inference

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init any model
detection_model = AutoDetectionModel.from_pretrained(model_type='mmdet',...) # for MMDetection models
detection_model = AutoDetectionModel.from_pretrained(model_type='ultralytics',...) # for YOLOv8/YOLO11/YOLO26 models
detection_model = AutoDetectionModel.from_pretrained(model_type='huggingface',...) # for HuggingFace detection models
detection_model = AutoDetectionModel.from_pretrained(model_type='torchvision',...) # for Torchvision detection models
detection_model = AutoDetectionModel.from_pretrained(model_type='rtdetr',...) # for RT-DETR models
detection_model = AutoDetectionModel.from_pretrained(model_type='yoloe',...) # for YOLOE models
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov5',...) # for YOLOv5 models
detection_model = AutoDetectionModel.from_pretrained(model_type='yolo-world',...) # for YOLOWorld models
detection_model = AutoDetectionModel.from_pretrained(model_type='roboflow',...) # for Roboflow RFDETR detection/segmentation models

# get sliced prediction result
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

```

### Predicting on very large images

When you call `get_sliced_prediction` with a file path, slices are read from disk one
row-band at a time instead of decoding the whole image up front, so peak memory stays a
small fraction of the scan rather than tracking its full decoded size. Two things unlock
the full saving:

- Install the optional streaming backend: `pip install sahi[bigimage]` (bundles
  libvips). Without it the image is decoded whole, matching the old behaviour.
- Pass `perform_standard_pred=False`. The default also runs inference on the full
  image, which decodes it in one piece and cancels out the saving. sahi logs a warning
  when you leave it on while slices are being streamed.

A Pillow image, a numpy array or a URL is already decoded (or has to be fetched whole),
so those inputs keep the old behaviour. The same goes for the `sahi predict` CLI, which
decodes the image up front for the visualisations it exports: use the Python API to
stream.

Images whose EXIF orientation tag rotates or mirrors them cannot be streamed and fall
back to a whole-image decode (a warning says so). `SliceImageStream.is_streaming` reports
whether a given input will actually be streamed.

#### Measured effect

A 39266x29140 (1144 MP) JPEG scan, sliced 640x640 at 0.2 overlap into 4389 slices,
peak RSS of the whole process:

| path | peak RSS | wall time |
| --- | --- | --- |
| streaming, libvips present | 671 MB | 10.9 s |
| whole-image decode alone (no slicing) | 6613 MB | 9.0 s |
| `slice_image`, all slices materialised | 3.2 GB image + 5.0 GB slices, does not complete in 6 GB | - |

Slicing on its own, with nothing consuming the batches, streaming is about 10-15% slower
than the eager path, plus a flat ~0.1 s for the one-time `import pyvips`:

| image | eager | streaming |
| --- | --- | --- |
| 1 MP | 0.27 s / 156 MB | 0.44 s / 156 MB |
| 16 MP | 0.62 s / 163 MB | 0.91 s / 189 MB |
| 64 MP | 1.83 s / 438 MB | 2.07 s / 327 MB |
| 144 MP | 3.71 s / 896 MB | 4.13 s / 508 MB |
| 256 MP | 6.56 s / 1537 MB | 6.69 s / 733 MB |

Below ~64 MP streaming costs memory rather than saving it, since the fixed overhead
outweighs the band saving.

That is the pessimistic case, though: it measures slicing with no inference. In a real
`get_sliced_prediction` the model runs on each batch, and the next band is decoded on a
worker thread while it does (libvips releases the GIL). At 30 ms of work per batch,
roughly what a small detector costs on 8 slices:

| image | eager | streaming, no prefetch | streaming + prefetch |
| --- | --- | --- | --- |
| 64 MP | 5.77 s / 438 MB | 5.83 s / 327 MB | **4.34 s / 321 MB** |
| 256 MP | 20.73 s / 1537 MB | 19.45 s / 733 MB | **16.62 s / 722 MB** |

So with anything real consuming the slices, streaming ends up both faster than the eager
path and half its peak memory. Prefetch is on by default and can be turned off with
`SliceImageStream.iter_batches(batch_size, prefetch=False)`.

Reproduce any of this with:

```console
python scripts/benchmark_streaming_slices.py                  # slicing alone
python scripts/benchmark_streaming_slices.py --consume-ms 30  # with simulated inference
```

Peak memory grows with image height, at roughly a tenth the rate of a whole-image
decode. libvips retains about 0.45 bytes for every pixel it decodes — some 360 MB across
the 1144 MP scan. This is the loader's own bookkeeping: the same climb reproduces with a
bare `pyvips.Region` fetch loop and no sahi in the process. Much of the rest of the
resident set is memory glibc has freed but not returned to the OS, which `malloc_trim(0)`
reclaims. Budget for that slope on scans far taller than the ones measured here.

#### Caveat: import order

`sahi.utils.cv` raises `OPENCV_IO_MAX_IMAGE_PIXELS` at import time, because OpenCV reads
that variable once when *it* is imported and refuses to decode anything above 2^30
pixels otherwise. If something imports `cv2` before any `sahi` module, the limit stays
at its default and images above 1073 MP quietly fall back to the slower, more allocating
PIL path instead of failing. Import `sahi` first, or set the variable in the environment,
if you work at that size.

#### Installing libvips

`pip install sahi[bigimage]` pulls `pyvips[binary]`, which ships libvips itself and
needs nothing from the system.

That extra only exists from pyvips 3.0 on, so it is unavailable when something else in
the environment caps pyvips below 3.0. Roboflow's `inference` is the common case, since
it requires `pyvips<3.0` transitively; `pip` then warns `does not have an extra named
binary` and installs the bindings with no libvips behind them, and slicing falls back to
decoding whole images — correct results, but not the low memory.

Install libvips through the system instead and streaming works on any pyvips, 2.x
included:

| OS | command |
| --- | --- |
| Debian/Ubuntu | `apt install libvips42t64` (Ubuntu 24.04+; `libvips42` before that, `libvips` on older releases) |
| Fedora/RHEL | `dnf install vips` |
| macOS | `brew install vips` |
| Windows | download the libvips zip and add its `bin` to `PATH`, or use `conda install -c conda-forge libvips` |

then `pip install pyvips`. Verify with:

```console
python -c "import pyvips; print(pyvips.version(0), pyvips.version(1), pyvips.version(2))"
```

An importable pyvips with no libvips behind it raises `OSError` rather than
`ImportError`. sahi treats that as a failed backend and decodes the image whole. This
is logged at **debug** level, not as a warning — most installs have no libvips and its
absence is not an error — so enable debug logging to see it:

```python
import logging

logging.getLogger("sahi").setLevel(logging.DEBUG)
```

## Standard inference

```python
from sahi.predict import get_prediction
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get standard prediction result
result = get_prediction(
    image,
    detection_model,
)

```

## Batch inference

### Batch prediction over a folder or file list

Use the high-level `predict` function to run sliced inference over many images
at once and export results automatically:

```python
from sahi.predict import predict
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get batch predict result
result = predict(
    model_type=..., # one of 'ultralytics', 'mmdet', 'huggingface'
    model_path=..., # path to model weight file
    model_config_path=..., # for mmdet models
    model_confidence_threshold=0.5,
    model_device='cpu', # or 'cuda:0'
    source=..., # image or folder path
    no_standard_prediction=True,
    no_sliced_prediction=False,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    export_pickle=False,
    export_crop=False,
    progress_bar=False,
)
```

### Low-level batch inference API

`perform_batch_inference` lets you run a model over multiple images in a single
call and retrieve per-image prediction lists. Ultralytics YOLO models use native
GPU batching; all other models fall back to sequential single-image inference
with the same API.

```python
import cv2
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)

# Load a batch of images as numpy arrays (H, W, C) in RGB
images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]

# Run batch inference (native GPU batching for Ultralytics)
detection_model.perform_batch_inference(images)

# Provide per-image shift amounts and full image sizes (use [[0, 0]] defaults
# when images are not slices)
shift_amount_list = [[0, 0]] * len(images)
full_shape_list   = [[img.shape[0], img.shape[1]] for img in images]

detection_model.convert_original_predictions(
    shift_amount=shift_amount_list,
    full_shape=full_shape_list,
)

# Access predictions per image
for i, preds in enumerate(detection_model.object_prediction_list_per_image):
    print(f"Image {i}: {len(preds)} detections")
    for pred in preds:
        print(pred.category.name, pred.score.value, pred.bbox.to_xyxy())
```

!!! note "Single-image compatibility" The existing `object_prediction_list`

property is unchanged and returns predictions for the first image, so code that
uses `perform_inference` + `convert_original_predictions` +
`object_prediction_list` continues to work without modification.

## Progress-Bar

Two options were added to control and receive progress updates when running
sliced inference over many slices:

- `progress_bar` (bool): When True, shows a tqdm progress bar during slice
  processing. Useful for visual feedback in terminals and notebooks. Default is
  False.
- `progress_callback` (callable): A callback function that will be called after
  each slice (or slice group) is processed. The callback receives two integer
  arguments: `(current_slice_index, total_slices)`. Use this to integrate custom
  progress reporting (for example, update a GUI element or log progress to a
  file).

Example using the callback:

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init model
detection_model = AutoDetectionModel.from_pretrained(...)

def my_progress_callback(current, total):
    print(f"Processed {current}/{total} slices")

result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    progress_bar=False,           # disable tqdm bar
    progress_callback=my_progress_callback,  # use callback to receive updates
)
```

!!! tip "Notes" - `progress_bar` and `progress_callback` can be used together.

When both are provided, the tqdm bar will display and the callback will be
called after each slice group is processed. - The `progress_callback` is called
with 1-based indices (i.e. first call will be `(1, total)`).

## Exclude custom classes on inference

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# define the class names to exclude from custom model inference
exclude_classes_by_name = ["car"]

# or exclude classes by its custom id
exclude_classes_by_id = [0]

result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
    exclude_classes_by_name = exclude_classes_by_name
    # exclude_classes_by_id = exclude_classes_by_id
)

```

## Visualization parameters and export formats

```python
from sahi.predict import get_prediction
from sahi import AutoDetectionModel
from PIL import Image

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get prediction result
result = get_prediction(
    image,
    detection_model,
)

# Export with custom visualization parameters
result.export_visuals(
    export_dir="outputs/",
    text_size=1.0,  # Size of the class label text
    rect_th=2,      # Thickness of bounding box lines
    text_th=2,      # Thickness of the text
    hide_labels=False,  # Set True to hide class labels
    hide_conf=False,    # Set True to hide confidence scores
    color=(255, 0, 0),  # Custom color in RGB format (red in this example)
    file_name="custom_visualization",
    export_format="jpg"  # Supports 'jpg' and 'png'
)

# Export as COCO format annotations
coco_annotations = result.to_coco_annotations()
# Example output: [{'image_id': None, 'bbox': [x, y, width, height], 'category_id': 0, 'area': width*height, ...}]

# Export as COCO predictions (includes confidence scores)
coco_predictions = result.to_coco_predictions(image_id=1)
# Example output: [{'image_id': 1, 'bbox': [x, y, width, height], 'score': 0.98, 'category_id': 0, ...}]

# Export as imantics format
imantics_annotations = result.to_imantics_annotations()
# For use with imantics library: https://github.com/jsbroks/imantics

# Export for FiftyOne visualization
fiftyone_detections = result.to_fiftyone_detections()
# For use with FiftyOne: https://github.com/voxel51/fiftyone
```

!!! tip "Interactive Demos"
    Want to see these prediction utilities in action? Check out our
    [interactive notebooks](notebooks.md) with hands-on examples for every
    supported framework.
