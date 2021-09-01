# SCRIPTS

## `predict.py` script usage:

```bash
python scripts/predict.py --source image/file/or/folder --model_path path/to/model --config_path path/to/config
```

will perform sliced inference on default parameters and export the prediction visuals to runs/predict/exp folder.

You can specify additional sliced prediction parameters as:

```bash
python scripts/predict.py --slice_width 256 --slice_height 256 --overlap_height_ratio 0.1 --overlap_width_ratio 0.1 --conf_thresh 0.25 --source image/file/or/folder --model_path path/to/model --config_path path/to/config
```

- Specify detection framework as `--model_type mmdet` for MMDetection or `--model_type yolov5` for YOLOv5, to match with your model weight

- Specify postprocess type as `--postprocess_type UNIONMERGE` or `--postprocess_type NMS` to be applied over sliced predictions

- Specify postprocess match metric as `--match_metric IOS` for intersection over smaller area or `--match_metric IOU` for intersection over union

- Specify postprocess match threshold as `--match_thresh 0.5`

- Add `--class_agnostic` argument to ignore category ids of the predictions during postprocess (merging/nms)

- If you want to export prediction pickles and cropped predictions add `--pickle` and `--crop` arguments. If you want to change crop extension type, set it as `--visual_export_format JPG`.

- If you don't want to export prediction visuals, add `--novisual` argument.

- By default, scripts apply both standard and sliced prediction (multi-stage inference). If you don't want to perform sliced prediction add `--no_sliced_pred` argument. If you don't want to perform standard prediction add `--no_standard_pred` argument.

- If you want to perform prediction using a COCO annotation file, provide COCO json path as add `--coco_file path/to/coco/file` and coco image folder as `--source path/to/coco/image/folder`, predictions will be exported as a coco json file to runs/predict/exp/results.json. Then you can use coco_evaluation.py script to calculate COCO evaluation results or coco_error_analysis.py script to calculate detailed COCO error plots.

## `cocoresult2fiftyone.py` script usage:

```bash
python scripts/cocoresult2fiftyone.py --coco_image_dir dir/to/images --coco_json_path path/to/json --coco_result_path path/to/cocoresult
```

will open a FiftyOne app that visualizes the given dataset and detections.

Specify IOU threshold for FP/TP by `--iou_threshold 0.5` argument

Specify FiftyOne result name by `--coco_result_name yolov5-detections` argument

## `slice_coco.py` script usage:

```bash
python scripts/slice_coco.py path/to/coco/json/file coco/images/directory
```

will slice the given images and COCO formatted annotations and export them to given output folder directory.

Specify slice height/width size as `--slice_size 512`.

Specify slice overlap ratio for height/width size as `--overlap_ratio 0.2`.

If you want to ignore images with annotations set it add `--ignore_negative_samples` argument.

## `coco2yolov5.py` script usage:

(In Windows be sure to open anaconda cmd prompt/windows cmd `as admin` to be able to create symlinks properly.)

```bash
python scripts/coco2yolov5.py --coco_file path/to/coco/file --source coco/images/directory --train_split 0.9
```

will convert given coco dataset to yolov5 format and export to runs/coco2yolov5/exp folder.

## `coco_evaluation.py` script usage:

```bash
python scripts/coco_evaluation.py dataset.json results.json
```

will calculate coco evaluation and export them to given output folder directory.

If you want to specify mAP metric type, set it as `--metric bbox mask`.

If you want to also calculate classwise scores add `--classwise` argument.

If you want to specify max detections, set it as `--proposal_nums 10 100 500`.

If you want to specify a psecific IOU threshold, set it as `--iou_thrs 0.5`. Default includes `0.50:0.95` and `0.5` scores.

If you want to specify export directory, set it as `--out_dir output/folder/directory `.

## `coco_error_analysis.py` script usage:

```bash
python scripts/coco_error_analysis.py dataset.json results.json
```

will calculate coco error plots and export them to given output folder directory.

If you want to specify mAP result type, set it as `--types bbox mask`.

If you want to export extra mAP bar plots and annotation area stats add `--extraplots` argument.

If you want to specify area regions, set it as `--areas 1024 9216 10000000000`.

If you want to specify export directory, set it as `--out_dir output/folder/directory `.
