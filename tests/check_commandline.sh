#!/bin/bash

##################################################################################
# Checks, if the CLI works
##################################################################################

source .venv/bin/activate

set -e

# predict mmdet
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')

echo "mmcv"
if (($(echo "$PYTHON_VERSION < 3.11" | bc -l))); then
  echo "mmcv: 1"
  sahi predict --model_type mmdet --source tests/data/ --novisual --model_path tests/data/models/mmdet/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth --model_config_path tests/data/models/mmdet/yolox/yolox_tiny_8xb8-300e_coco.py --image_size 320
  echo "mmcv: 2"
  sahi predict --model_type mmdet --source tests/data/coco_utils/terrain1.jpg --export_pickle --export_crop --model_path tests/data/models/mmdet/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth --model_config_path tests/data/models/mmdet/yolox/yolox_tiny_8xb8-300e_coco.py --image_size 320
  echo "mmcv: 3"
  sahi predict --model_type mmdet --source tests/data/coco_utils/ --novisual --dataset_json_path tests/data/coco_utils/combined_coco.json --model_path tests/data/models/mmdet/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth --model_config_path tests/data/models/mmdet/yolox/yolox_tiny_8xb8-300e_coco.py --image_size 320
else
  echo "mmcv not available for Python version $PYTHON_VERSION"
fi

sahi predict --no_sliced_prediction --model_type yolov5 --source tests/data/coco_utils/terrain1.jpg --novisual --model_path tests/data/models/yolov5/yolov5s6.pt --image_size 320
sahi predict --model_type yolov5 --source tests/data/ --novisual --model_path tests/data/models/yolov5/yolov5s6.pt --image_size 320
sahi predict --model_type yolov5 --source tests/data/coco_utils/terrain1.jpg --export_pickle --export_crop --model_path tests/data/models/yolov5/yolov5s6.pt --image_size 320
sahi predict --model_type yolov5 --source tests/data/coco_utils/ --novisual --dataset_json_path tests/data/coco_utils/combined_coco.json --model_path tests/data/models/yolov5/yolov5s6.pt --image_size 320
# coco yolov5
sahi coco yolov5 --image_dir tests/data/coco_utils/ --dataset_json_path tests/data/coco_utils/combined_coco.json --train_split 0.9
# coco evaluate
sahi coco evaluate --dataset_json_path tests/data/coco_evaluate/dataset.json --result_json_path tests/data/coco_evaluate/result.json
# coco analyse
sahi coco analyse --dataset_json_path tests/data/coco_evaluate/dataset.json --result_json_path tests/data/coco_evaluate/result.json --out_dir tests/data/coco_evaluate/
