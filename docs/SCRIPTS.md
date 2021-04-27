# SCRIPTS

## `predict.py` script usage:

```bash
python scripts/predict.py --source image/file/or/folder --model_path path/to/model --config_path path/to/config
```

will perform sliced inference on default parameters and export the prediction visuals to runs/predict/exp folder.

You can specify sliced inference parameters as:

```bash
python scripts/predict.py --slice_width 256 --slice_height 256 --overlap_height_ratio 0.1 --overlap_width_ratio 0.1 --iou_thresh 0.25 --source image/file/or/folder --model_path path/to/model --config_path path/to/config
```

If you want to export prediction pickles and cropped predictions add `--pickle` and `--crop` arguments. If you want to change crop extension type, set it as `--visual_export_format JPG`.

If you want to perform standard prediction instead of sliced prediction, add `--standard_pred` argument.

```bash
python scripts/predict.py --coco_file path/to/coco/file --source coco/images/directory --model_path path/to/model --config_path path/to/config
```

will perform inference using provided coco file, then export results as a coco json file to runs/predict/exp/results.json

If you don't want to export prediction visuals, add `--novisual` argument.

## `coco2yolov5.py` script usage:

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
