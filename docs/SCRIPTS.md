# SCRIPTS

## `predict.py` script usage:

```bash
python scripts/predict.py --source image/file/or/folder --model_path path/to/model --config_path path/to/config
```

will perform sliced inference on default parameters and export the prediction visuals to runs/predict/exp folder.

You can specify additional prediction parameters as:

```bash
python scripts/predict.py --slice_width 256 --slice_height 256 --overlap_height_ratio 0.1 --overlap_width_ratio 0.1 --source image/file/or/folder --model_path path/to/model --config_path path/to/config
```

- Specify detection framework as `--model_type mmdet` for MMDetection or `--model_type yolov5` for YOLOv5, to match with your model weight

- Specify postprocess type as `--postprocess_type UNIONMERGE` or `--postprocess_type NMS` to be applied over sliced predictions

- Specify postprocess match metric as `--match_metric IOS` for intersection over smaller area or `--match_metric IOU` for intersection over union

- Specify postprocess match threshold as `--match_thresh 0.5`

- Add `--class_agnostic` argument to ignore category ids of the predictions during postprocess (merging/nms)

- If you want to export prediction pickles and cropped predictions add `--pickle` and `--crop` arguments. If you want to change crop extension type, set it as `--visual_export_format JPG`.

- If you don't want to export prediction visuals, add `--novisual` argument.

- If you want to perform standard prediction instead of sliced prediction, add `--standard_pred` argument.

- If you want to perform prediction using a COCO annotation file, provide COCO json path as add `--coco_file path/to/coco/file` and coco image folder as `--source path/to/coco/image/folder`, predictions will be exported as a coco json file to runs/predict/exp/results.json. Then you can use coco_error_analysis.py script to calculate COCO evaluation results.


## `coco2yolov5.py` script usage:

(In Windows be sure to open anaconda cmd prompt/windows cmd `as admin` to be able to create symlinks properly.)

```bash
python scripts/coco2yolov5.py --coco_file path/to/coco/file --source coco/images/directory --train_split 0.9
```

will convert given coco dataset to yolov5 format and export to runs/coco2yolov5/exp folder.

## `coco_error_analysis.py` script usage:

```bash
python scripts/coco_error_analysis.py results.json output/folder/directory --ann coco/annotation/path
```

will calculate coco error plots and export them to given output folder directory.

If you want to specify mAP result type, set it as `--types bbox mask`.

If you want to export extra mAP bar plots and annotation area stats add `--extraplots` argument.

If you want to specify area regions, set it as `--areas 1024 9216 10000000000`.
