# CHANGELOG

## 0.11.31

## What's Changed

* Make Category immutable and add tests by @gboeer in <https://github.com/obss/sahi/pull/1206>
* Update docstring for greedy_nmm by @kikefdezl in <https://github.com/obss/sahi/pull/1205>
* update version by @fcakyon in <https://github.com/obss/sahi/pull/1208>

## New Contributors

* @kikefdezl made their first contribution in <https://github.com/obss/sahi/pull/1205>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.30...0.11.31\n>

## 0.11.30

# SAHI v0.11.30 Release Notes

  We're excited to announce SAHI v0.11.30 with improved performance tracking, enhanced testing
  infrastructure, and better developer experience!

## 📈 Milestones

* **Academic papers citing SAHI reached 400!** ([#1168](https://github.com/obss/sahi/pull/1168))

## 🚀 Key Updates

### Performance & Monitoring

* **Fixed postprocess duration tracking** in `get_sliced_prediction` - now properly separates slice,
  prediction, and postprocess timings for accurate performance monitoring
  ([#1201](https://github.com/obss/sahi/pull/1201)) - Thanks @Toprak2!

### Framework Updates

* **Refactored Ultralytics support** with ONNX model support and better compatibility
  ([#1184](https://github.com/obss/sahi/pull/1184))
* **Updated TorchVision support** to latest API ([#1182](https://github.com/obss/sahi/pull/1182))
* **Improved Detectron2 support** with better config handling to prevent KeyError issues
  ([#1116](https://github.com/obss/sahi/pull/1116)) - Thanks @Arnesh1411!
* **Added Roboflow framework support** for RF-DETR models from the Roboflow Universe
  ([#1161](https://github.com/obss/sahi/pull/1161)) - Thanks @nok!
* **Removed deepsparse integration** as the framework is no longer maintained
  ([#1164](https://github.com/obss/sahi/pull/1164))

### Testing Infrastructure

* **Migrated test suite to pytest** ([#1187](https://github.com/obss/sahi/pull/1187))
  * Tests now run faster with better parallel execution
  * Extended Python version coverage (3.8, 3.9, 3.10, 3.11, 3.12)
  * Updated to more recent PyTorch versions for better compatibility testing
  * Improved test organization and maintainability
* **Refactored MMDetection tests** for better reliability ([#1185](https://github.com/obss/sahi/pull/1185))

### Developer Experience

* **Added Context7 MCP integration** for AI-assisted development
  ([#1198](https://github.com/obss/sahi/pull/1198))
  * SAHI's documentation is now [indexed in Context7 MCP](https://context7.com/obss/sahi)
  * Provides AI coding assistants with up-to-date, version-specific code examples
  * Includes [llms.txt](https://context7.com/obss/sahi/llms.txt) file for AI-readable documentation
  * Check out the [Context7 MCP installation
  guide](https://github.com/upstash/context7#%EF%B8%8F-installation) to integrate SAHI docs with your AI
  workflow

## 🛠️ Improvements

### Code Quality & Safety

* **Immutable bounding boxes** for thread-safe operations ([#1194](https://github.com/obss/sahi/pull/1194),
   [#1191](https://github.com/obss/sahi/pull/1191)) - Thanks @gboeer!
* **Enhanced type hints and docstrings** throughout the codebase
  ([#1195](https://github.com/obss/sahi/pull/1195)) - Thanks @gboeer!
* **Overloaded operators for prediction scores** enabling intuitive score comparisons
  ([#1190](https://github.com/obss/sahi/pull/1190)) - Thanks @gboeer!
* **PyTorch is now a soft dependency** improving flexibility
  ([#1162](https://github.com/obss/sahi/pull/1162)) - Thanks @ducviet00!

### Infrastructure & Stability

* **Improved dependency management** and documentation ([#1183](https://github.com/obss/sahi/pull/1183))
* **Enhanced pyproject.toml configuration** for better package management
  ([#1181](https://github.com/obss/sahi/pull/1181))
* **Optimized CI/CD workflows** for MMDetection tests ([#1186](https://github.com/obss/sahi/pull/1186))

## 🐛 Bug Fixes

* Fixed CUDA device selection to support devices other than cuda:0
  ([#1158](https://github.com/obss/sahi/pull/1158)) - Thanks @0xf21!
* Corrected parameter naming from 'confidence' to 'threshold' for consistency
  ([#1180](https://github.com/obss/sahi/pull/1180)) - Thanks @nok!
* Fixed regex string formatting in device selection function
  ([#1165](https://github.com/obss/sahi/pull/1165))
* Resolved torch import errors when PyTorch is not installed
  ([#1172](https://github.com/obss/sahi/pull/1172)) - Thanks @ducviet00!
* Fixed model instantiation issues with `AutoDetectionModel.from_pretrained`
  ([#1158](https://github.com/obss/sahi/pull/1158))

## 📦 Dependencies

* Updated OpenCV packages from 4.10.0.84 to 4.11.0.86 ([#1171](https://github.com/obss/sahi/pull/1171)) -
  Thanks @ducviet00-h2!
* Removed unmaintained matplotlib-stubs dependency ([#1169](https://github.com/obss/sahi/pull/1169))
* Cleaned up unused configuration files ([#1199](https://github.com/obss/sahi/pull/1199))

## 📚 Documentation

* Added context7.json for better AI tool integration ([#1200](https://github.com/obss/sahi/pull/1200))
* Updated README with new contributors ([#1175](https://github.com/obss/sahi/pull/1175),
  [#1179](https://github.com/obss/sahi/pull/1179))
* Added Roboflow+SAHI Colab tutorial link ([#1177](https://github.com/obss/sahi/pull/1177))

## Acknowledgments

  Special thanks to all contributors who made this release possible: @nok, @gboeer, @Toprak2, @Arnesh1411,
  @0xf21, @ducviet00, @ducviet00-h2, @p-constant, and @fcakyon!

  ---

  **Full Changelog**: <https://github.com/obss/sahi/compare/0.11.24...0.11.30\n>

## 0.11.29

## What's Changed

* Make bounding box immutable by @gboeer in <https://github.com/obss/sahi/pull/1194>
* Improve type hints and docstrings by @gboeer in <https://github.com/obss/sahi/pull/1195>
* update version by @fcakyon in <https://github.com/obss/sahi/pull/1196>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.28...0.11.29\n>

## 0.11.28

## What's Changed

* Add overloaded operators for prediction score by @gboeer in <https://github.com/obss/sahi/pull/1190>
* Improve detectron2 support by @Arnesh1411 in <https://github.com/obss/sahi/pull/1116>
* Use immutable arguments for bounding boxes by @gboeer in <https://github.com/obss/sahi/pull/1191>
* update version by @fcakyon in <https://github.com/obss/sahi/pull/1192>

## New Contributors

* @Arnesh1411 made their first contribution in <https://github.com/obss/sahi/pull/1116>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.27...0.11.28\n>

## 0.11.27

## What's Changed

* fix: Update inference method to use 'threshold' instead of 'confidence' by @nok in <https://github.com/obss/sahi/pull/1180>
* Update README.md by @nok in <https://github.com/obss/sahi/pull/1179>
* improve pyproject.toml by @fcakyon in <https://github.com/obss/sahi/pull/1181>
* Refactor dependency management and some docs by @fcakyon in <https://github.com/obss/sahi/pull/1183>
* update: refactor ultralytics support by @fcakyon in <https://github.com/obss/sahi/pull/1184>
* Refactor mmdet tests by @fcakyon in <https://github.com/obss/sahi/pull/1185>
* update torchvision support to latest api by @fcakyon in <https://github.com/obss/sahi/pull/1182>
* optimize mmdet workflow trigger condition by @fcakyon in <https://github.com/obss/sahi/pull/1186>
* Migrate tests to pytest by @fcakyon in <https://github.com/obss/sahi/pull/1187>
* update version by @fcakyon in <https://github.com/obss/sahi/pull/1188>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.26...0.11.27\n>

## 0.11.26

## What's Changed

* Bump opencv packages from `4.10.0.84` to `4.11.0.86` by @ducviet00-h2 in <https://github.com/obss/sahi/pull/1171>
* Add new framework Roboflow (RFDETR models) by @nok in <https://github.com/obss/sahi/pull/1161>
* add new contributors to readme by @fcakyon in <https://github.com/obss/sahi/pull/1175>
* add roboflow+sahi colab url to readme by @fcakyon in <https://github.com/obss/sahi/pull/1177>
* update version by @fcakyon in <https://github.com/obss/sahi/pull/1176>

## New Contributors

* @ducviet00-h2 made their first contribution in <https://github.com/obss/sahi/pull/1171>
* @nok made their first contribution in <https://github.com/obss/sahi/pull/1161>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.25...0.11.26\n>

## 0.11.25

## What's Changed

* update sahi citation in readme by @fcakyon in <https://github.com/obss/sahi/pull/1168>
* remove matplotlib-stubs as its not maintained by @fcakyon in <https://github.com/obss/sahi/pull/1169>
* Fix torch import errors by @ducviet00 in <https://github.com/obss/sahi/pull/1172>
* update version by @fcakyon in <https://github.com/obss/sahi/pull/1173>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.24...0.11.25\n>

## 0.11.24

## What's Changed

* Fix typo and scripts URL by @gboeer in <https://github.com/obss/sahi/pull/1155>
* fix ci workflow bug by @Dronakurl in <https://github.com/obss/sahi/pull/1156>
* [DOC] Fix typos by @gboeer in <https://github.com/obss/sahi/pull/1157>
* Remove deepsparse integration by @fcakyon in <https://github.com/obss/sahi/pull/1164>
* Fix: Make pytorch is not a hard dependency by @ducviet00 in <https://github.com/obss/sahi/pull/1162>
* fix: specify a device other than cuda:0 by @0xf21 in <https://github.com/obss/sahi/pull/1158>
* fix: correct regex string formatting in select_device function by @fcakyon in <https://github.com/obss/sahi/pull/1165>
* add TensorrtExecutionProvider to yolov8onnx by @p-constant in <https://github.com/obss/sahi/pull/1091>
* update version by @fcakyon in <https://github.com/obss/sahi/pull/1166>

## New Contributors

* @gboeer made their first contribution in <https://github.com/obss/sahi/pull/1155>
* @ducviet00 made their first contribution in <https://github.com/obss/sahi/pull/1162>
* @0xf21 made their first contribution in <https://github.com/obss/sahi/pull/1158>
* @p-constant made their first contribution in <https://github.com/obss/sahi/pull/1091>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.23...0.11.24\n>

## 0.11.23

## What's Changed

* fix(CI): numpy dependency fixes #1119 by @Dronakurl in <https://github.com/obss/sahi/pull/1144>
* Fix: Predict cannot find TIF files in source directory by @dibunker in <https://github.com/obss/sahi/pull/1142>
* Fixed typos in demo Notebooks by @picjul in <https://github.com/obss/sahi/pull/1150>
* fix: Fix Polygon Repair and Empty Polygon Issues, see #1118 by @mario-dg in <https://github.com/obss/sahi/pull/1138>
* improve package ci logging by @fcakyon in <https://github.com/obss/sahi/pull/1151>

## New Contributors

* @dibunker made their first contribution in <https://github.com/obss/sahi/pull/1142>
* @picjul made their first contribution in <https://github.com/obss/sahi/pull/1150>
* @mario-dg made their first contribution in <https://github.com/obss/sahi/pull/1138>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.22...0.11.23\n>

## 0.11.22

## What's Changed

* Improve suppot for latest mmdet (v3.3.0) by @fcakyon in <https://github.com/obss/sahi/pull/1129>
* Improve support for latest yolov5-pip and ultralytics versions by @fcakyon in <https://github.com/obss/sahi/pull/1130>
* support latest huggingface/transformers models by @fcakyon in <https://github.com/obss/sahi/pull/1131>
* refctor coco to yolo conversion, update docs by @fcakyon in <https://github.com/obss/sahi/pull/1132>
* bump version by @fcakyon in <https://github.com/obss/sahi/pull/1134>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.21...0.11.22>

## Core Documentation Files

### [Prediction Utilities](predict.md)

- Detailed guide for performing object detection inference
* Standard and sliced inference examples
* Batch prediction usage
* Class exclusion during inference
* Visualization parameters and export formats
* Interactive examples with various model integrations (YOLOv8, MMDetection, etc.)

### [Slicing Utilities](slicing.md)

- Guide for slicing large images and datasets
* Image slicing examples
* COCO dataset slicing examples
* Interactive demo notebook reference

### [COCO Utilities](coco.md)

- Comprehensive guide for working with COCO format datasets
* Dataset creation and manipulation
* Slicing COCO datasets
* Dataset splitting (train/val)
* Category filtering and updates
* Area-based filtering
* Dataset merging
* Format conversion (COCO ↔ YOLO)
* Dataset sampling utilities
* Statistics calculation
* Result validation

### [CLI Commands](cli.md)

- Complete reference for SAHI command-line interface
* Prediction commands
* FiftyOne integration
* COCO dataset operations
* Environment information
* Version checking
* Custom script usage

### [FiftyOne Integration](fiftyone.md)

- Guide for visualizing and analyzing predictions with FiftyOne
* Dataset visualization
* Result exploration
* Interactive analysis

## Interactive Examples

All documentation files are complemented by interactive Jupyter notebooks in the [demo directory](../demo/):
* `slicing.ipynb` - Slicing operations demonstration
* `inference_for_ultralytics.ipynb` - YOLOv8/YOLO11/YOLO12 integration
* `inference_for_yolov5.ipynb` - YOLOv5 integration
* `inference_for_mmdetection.ipynb` - MMDetection integration
* `inference_for_huggingface.ipynb` - HuggingFace models integration
* `inference_for_torchvision.ipynb` - TorchVision models integration
* `inference_for_rtdetr.ipynb` - RT-DETR integration
* `inference_for_sparse_yolov5.ipynb` - DeepSparse optimized inference

## Getting Started

If you're new to SAHI:

1. Start with the [prediction utilities](predict.md) to understand basic inference
2. Explore the [slicing utilities](slicing.md) to learn about processing large images
3. Check out the [CLI commands](cli.md) for command-line usage
4. Dive into [COCO utilities](coco.md) for dataset operations
5. Try the interactive notebooks in the [demo directory](../demo/) for hands-on experience
\n

## 0.11.21

## What's Changed

* Exclude classes from inference using pretrained or custom models by @gguzzy in <https://github.com/obss/sahi/pull/1104>
* pyproject.toml, pre-commit, ruff, uv and typing issues, fixes #1119 by @Dronakurl in <https://github.com/obss/sahi/pull/1120>
* add class exclusion example into predict docs by @gguzzy in <https://github.com/obss/sahi/pull/1125>
* Add OBB demo by @fcakyon in <https://github.com/obss/sahi/pull/1126>
* fix a type hint typo in predict func by @fcakyon in <https://github.com/obss/sahi/pull/1111>
* Remove numpy<2 upper pin by @weiji14 in <https://github.com/obss/sahi/pull/1112>
* fix ci badge on readme by @fcakyon in <https://github.com/obss/sahi/pull/1124>
* fix version in pyproject.toml by @fcakyon in <https://github.com/obss/sahi/pull/1127>

## New Contributors

* @Dronakurl made their first contribution in <https://github.com/obss/sahi/pull/1120>
* @gguzzy made their first contribution in <https://github.com/obss/sahi/pull/1104>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.20...0.11.21\n>

## 0.11.20

## What's Changed

* add yolo11 and ultralytics obb task support by @fcakyon in <https://github.com/obss/sahi/pull/1109>
* support latest opencv version by @fcakyon in <https://github.com/obss/sahi/pull/1106>
* simplify yolo detection model code by @fcakyon in <https://github.com/obss/sahi/pull/1107>
* Pin shapely>2.0.0 by @weiji14 in <https://github.com/obss/sahi/pull/1101>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.19...0.11.20\n>

## 0.11.19

## What's Changed

* fix ci actions by @fcakyon in <https://github.com/obss/sahi/pull/1073>
* Update has_mask method for mmdet models (handle an edge case) by @ccomkhj in <https://github.com/obss/sahi/pull/1066>
* Another self-intersection corner case handling by @sergiev in <https://github.com/obss/sahi/pull/982>
* Update README.md by @fcakyon in <https://github.com/obss/sahi/pull/1077>
* drop non-working yolonas support by @fcakyon in <https://github.com/obss/sahi/pull/1097>
* drop yolonas support part2 by @fcakyon in <https://github.com/obss/sahi/pull/1098>
* Update has_mask method for mmdet models (handle ConcatDataset) by @ccomkhj in <https://github.com/obss/sahi/pull/1092>

## New Contributors

* @ccomkhj made their first contribution in <https://github.com/obss/sahi/pull/1066>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.18...0.11.19\n>

## 0.11.18

## What's Changed

* add yolov8 mask support, improve mask processing speed by 4-5x by @mayrajeo in <https://github.com/obss/sahi/pull/1039>
* fix has_mask method for mmdet models by @Alias-z in <https://github.com/obss/sahi/pull/1054>
* Fix `TypeError: 'GeometryCollection' object is not subscriptable` when slicing COCO by @Alias-z in <https://github.com/obss/sahi/pull/1047>
* support opencv-python version 4.9 by @iokarkan in <https://github.com/obss/sahi/pull/1041>
* add upperlimit to numpy dep by @fcakyon in <https://github.com/obss/sahi/pull/1057>
* add more unit tests by @MMerling in <https://github.com/obss/sahi/pull/1048>
* upgrade ci actions by @fcakyon in <https://github.com/obss/sahi/pull/1049>

## New Contributors

* @iokarkan made their first contribution in <https://github.com/obss/sahi/pull/1041>
* @MMerling made their first contribution in <https://github.com/obss/sahi/pull/1048>
* @Alias-z made their first contribution in <https://github.com/obss/sahi/pull/1047>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.16...0.11.18\n>

## v0.11.16

## v0.11.15

## 0.11.14

## What's Changed

* support Deci-AI YOLO-NAS models by @ssahinnkadir in <https://github.com/obss/sahi/pull/874>
* Significant speed improvement for Detectron2 models by @MyosQ in <https://github.com/obss/sahi/pull/865>
* support ultralytics>=8.0.99 by @eVen-gits in <https://github.com/obss/sahi/pull/873>
* Documentation typo, and missing value by @Hamzalopode in <https://github.com/obss/sahi/pull/859>
* update version by @fcakyon in <https://github.com/obss/sahi/pull/876>
* update black version by @fcakyon in <https://github.com/obss/sahi/pull/877>

## New Contributors

* @Hamzalopode made their first contribution in <https://github.com/obss/sahi/pull/859>
* @eVen-gits made their first contribution in <https://github.com/obss/sahi/pull/873>
* @MyosQ made their first contribution in <https://github.com/obss/sahi/pull/865>

**Full Changelog**: <https://github.com/obss/sahi/compare/0.11.13...0.11.14\n>

## v0.11.13

## v0.11.12

## v0.11.11

## v0.11.10

## v0.11.9

## v0.11.8

## v0.11.7

## v0.11.6

## v0.11.5

## v0.11.4

## v0.11.3

## v0.11.2

## v0.11.1
