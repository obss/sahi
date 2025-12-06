# CLI Commands

SAHI provides a comprehensive command-line interface for object detection tasks. This guide covers all available commands with detailed examples and options.

## `predict` command usage

Perform object detection inference on images or videos using sliced inference for better small object detection.

### Basic Usage

```bash
sahi predict --source image/file/or/folder --model_path path/to/model --model_config_path path/to/config
```

This will perform sliced inference with default parameters and export prediction visuals to `runs/predict/exp` folder.

### Video Input Support

SAHI supports video inference with the same command structure:

```bash
sahi predict --model_path yolo11s.pt --model_type ultralytics --source video.mp4
```

#### Real-time Video Visualization

View video rendering during inference with the `--view_video` flag:

```bash
sahi predict --model_path yolo11s.pt --model_type ultralytics --source video.mp4 --view_video
```

**Keyboard Controls:**

- **`D`** - Forward 100 frames
- **`A`** - Rewind 100 frames
- **`G`** - Forward 20 frames
- **`F`** - Rewind 20 frames
- **`Esc`** - Exit viewer

> **Tip:** If `--view_video` is slow, add `--frame_skip_interval=20` to skip intervals of 20 frames.

### Advanced Slicing Parameters

Customize the slicing behavior for optimal detection:

```bash
sahi predict --slice_width 512 --slice_height 512 \
  --overlap_height_ratio 0.1 --overlap_width_ratio 0.1 \
  --model_confidence_threshold 0.25 \
  --source image/file/or/folder \
  --model_path path/to/model \
  --model_config_path path/to/config
```

#### Model Configuration

**Detection Framework:**

- `--model_type mmdet` - For MMDetection models
- `--model_type ultralytics` - For Ultralytics/YOLOv5/YOLO11 models
- `--model_type huggingface` - For HuggingFace models
- `--model_type torchvision` - For Torchvision models

**Confidence Threshold:**

- `--model_confidence_threshold 0.25` - Set minimum confidence for detections

#### Postprocessing Options

**Postprocess Type:**

- `--postprocess_type GREEDYNMM` - Greedy non-maximum merging (default)
- `--postprocess_type NMS` - Standard non-maximum suppression

**Match Metrics:**

- `--postprocess_match_metric IOS` - Intersection over smaller area
- `--postprocess_match_metric IOU` - Intersection over union (default)

**Additional Options:**

- `--postprocess_match_threshold 0.5` - Set matching threshold
- `--postprocess_class_agnostic` - Ignore category IDs during postprocessing

#### Export Options

**Visual Exports:**

- `--novisual` - Disable prediction visualization exports
- `--visual_export_format JPG` - Set export format (JPG, PNG, etc.)

**Data Exports:**

- `--export_pickle` - Export prediction pickles
- `--export_crop` - Export cropped detections

#### Inference Modes

By default, SAHI performs multi-stage inference (both standard and sliced prediction):

- `--no_sliced_prediction` - Disable sliced inference (standard only)
- `--no_standard_prediction` - Disable standard inference (sliced only)

### COCO Dataset Evaluation

Perform prediction using a COCO annotation file for evaluation:

```bash
sahi predict --dataset_json_path dataset.json \
  --source path/to/coco/image/folder \
  --model_path path/to/model
```

Predictions will be exported as a COCO JSON file to `runs/predict/exp/results.json`. You can then use:

- `sahi coco evaluate` - Calculate COCO evaluation metrics
- `sahi coco analyse` - Generate detailed error analysis plots

### Progress Reporting

Enable a progress bar to track inference progress:

```bash
sahi predict --model_path path/to/model --source images/ \
  --slice_width 512 --slice_height 512 --progress_bar
```

> **Note:** The `--progress_bar` flag controls CLI visual progress (tqdm). The `progress_callback` parameter is available in the Python API but not exposed as a CLI option.

---

## `predict-fiftyone` command usage

Perform sliced inference and visualize results interactively using the FiftyOne App.

### Basic Usage

```bash
sahi predict-fiftyone --image_dir image/file/or/folder \
  --dataset_json_path dataset.json \
  --model_path path/to/model \
  --model_config_path path/to/config
```

This will perform sliced inference with default parameters and launch the FiftyOne App for interactive exploration.

### Additional Parameters

All parameters from the [`sahi predict`](#predict-command-usage) command are supported.

---

## `coco fiftyone` command usage

Visualize and compare multiple detection results on your COCO dataset using FiftyOne UI.

### Basic Usage

You need to convert your predictions to [COCO result JSON format](https://cocodataset.org/#format-results). Use [`sahi predict`](#predict-command-usage) to generate this format.

```bash
sahi coco fiftyone --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json cocoresult2.json
```

This opens a FiftyOne app that visualizes the dataset and compares 2 detection results ordered by misdetections.

### Options

- `--iou_threshold 0.5` - Set IOU threshold for FP/TP classification

---

## `coco slice` command usage

Slice large images and their COCO format annotations into smaller tiles.

### Basic Usage

```bash
sahi coco slice --image_dir dir/to/images \
  --dataset_json_path dataset.json
```

Slices images and COCO annotations, exporting them to the output folder.

### Parameters

**Slice Dimensions:**

- `--slice_size 512` - Set slice height and width (default: 512)

**Overlap:**

- `--overlap_ratio 0.2` - Set overlap ratio for height/width (default: 0.2)

**Filtering:**

- `--ignore_negative_samples` - Exclude images without annotations

**Output:**

- `--out_dir output/folder` - Specify output directory

---

## `coco yolo` command usage

Convert COCO format datasets to YOLO format for training with Ultralytics.

> **Windows Users:** Open Anaconda prompt or Windows CMD **as administrator** to create symlinks properly.

### Basic Usage

```bash
sahi coco yolo --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  --train_split 0.9
```

Converts the COCO dataset to YOLO format and exports to `runs/coco2yolo/exp` folder.

### Parameters

- `--train_split 0.9` - Set training split ratio (default: 0.9)
- `--out_dir output/folder` - Specify output directory

---

## `coco evaluate` command usage

Calculate COCO evaluation metrics (mAP, mAR) for your predictions.

### Basic Usage

You need to convert your predictions to [COCO result JSON format](https://cocodataset.org/#format-results). Use [`sahi predict`](#predict-command-usage) to generate this format.

```bash
sahi coco evaluate --dataset_json_path dataset.json \
  --result_json_path result.json
```

Calculates COCO evaluation metrics and exports results to the output folder.

### Parameters

**Metric Type:**

- `--type bbox` - Evaluate bounding box detections (default)
- `--type mask` - Evaluate instance segmentation masks

**Scoring Options:**

- `--classwise` - Calculate per-class scores in addition to overall metrics

**Detection Limits:**

- `--proposal_nums "[10 100 500]"` - Set max detections per image (default: [100, 300, 1000])

**IOU Thresholds:**

- `--iou_thrs 0.5` - Specify IOU threshold (default: 0.50:0.95 and 0.5)

**Output:**

- `--out_dir output/folder` - Specify output directory

---

## `coco analyse` command usage

Generate detailed error analysis plots for COCO predictions.

### Basic Usage

You need to convert your predictions to [COCO result JSON format](https://cocodataset.org/#format-results). Use [`sahi predict`](#predict-command-usage) to generate this format.

```bash
sahi coco analyse --dataset_json_path dataset.json \
  --result_json_path result.json \
  --out_dir output/directory
```

Generates comprehensive error analysis plots and exports them to the specified folder.

### Parameters

**Analysis Type:**

- `--type bbox` - Analyze bounding box detections (default)
- `--type segm` - Analyze instance segmentation masks

**Additional Plots:**

- `--extraplots` - Generate extra mAP bar plots and annotation area statistics

**Area Regions:**

- `--areas "[1024 9216 10000000000]"` - Define area regions for analysis (default: small/medium/large COCO areas)

---

## `env` command usage

Display installed package versions related to SAHI.

### Usage

```bash
sahi env
```

### Example Output

```text
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   torch version 2.1.2 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   torchvision version 0.16.2 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   ultralytics version 8.3.86 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   transformers version 4.49.0 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   timm version 0.9.1 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   fiftyone version 0.14.2 is available.
```

---

## `version` command usage

Display your currently installed SAHI version.

### Usage

```bash
sahi version
0.11.22
```

---

## Custom Scripts

All scripts can be downloaded from the [scripts directory](https://github.com/obss/sahi/tree/main/scripts) and modified for your specific needs.

After installing SAHI via pip, all scripts can be called from any directory:

```bash
python script_name.py
```

---

## Additional Resources

Looking to dive deeper? Here are some helpful resources:

- **[Prediction Utilities Documentation](predict.md)** - Detailed walkthrough of prediction parameters and visualization
- **[Slicing Utilities Guide](slicing.md)** - In-depth exploration of slicing operations
- **[COCO Utilities Documentation](coco.md)** - Comprehensive examples with COCO format operations
- **[Interactive Demo Notebooks](../demo/)** - Hands-on examples of CLI commands in action
- **[Model Documentation](models/)** - Framework-specific model integration guides

These resources provide comprehensive examples and explanations to help you make the most of SAHI's command-line interface.
