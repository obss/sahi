# CLI Commands

## `predict` command usage:

```bash
>>sahi predict --source image/file/or/folder --model_path path/to/model --model_config_path path/to/config
```

will perform sliced inference on default parameters and export the prediction visuals to runs/predict/exp folder.

- It also supports video input:

```bash
>>sahi predict --model_path yolo11s.pt --model_type ultralytics --source video.mp4
```

You can also view video render during video inference with `--view_video`:

```bash
>>sahi predict --model_path yolo11s.pt --model_type ultralytics --source video.mp4 --view_video
```

- To `forward 100 frames`, on opened window press key `D`
- To `revert 100 frames`, on opened window press key `A`
- To `forward 20 frames`, on opened window press key `G`
- To `revert 20 frames`, on opened window press key `F`
- To `exit`, on opened window press key `Esc`

Note: If `--view_video` is slow, you can add `--frame_skip_interval=20` argument to skip interval of 20 frames each time.

You can specify additional sliced prediction parameters as:

```bash
>>sahi predict --slice_width 512 --slice_height 512 --overlap_height_ratio 0.1 --overlap_width_ratio 0.1 --model_confidence_threshold 0.25 --source image/file/or/folder --model_path path/to/model --model_config_path path/to/config
```

- Specify detection framework as `--model_type mmdet` for MMDetection or `--model_type ultralytics` for Ultralytics, to match with your model weight file

- Specify postprocess type as `--postprocess_type GREEDYNMM` or `--postprocess_type NMS` to be applied over sliced predictions

- Specify postprocess match metric as `--postprocess_match_metric IOS` for intersection over smaller area or `--postprocess_match_metric IOU` for intersection over union

- Specify postprocess match threshold as `--postprocess_match_threshold 0.5`

- Add `--postprocess_class_agnostic` argument to ignore category ids of the predictions during postprocess (merging/nms)

- If you want to export prediction pickles and cropped predictions add `--export_pickle` and `--export_crop` arguments. If you want to change crop extension type, set it as `--visual_export_format JPG`.

- If you don't want to export prediction visuals, add `--novisual` argument.

- By default, scripts apply both standard and sliced prediction (multi-stage inference). If you don't want to perform sliced prediction add `--no_sliced_prediction` argument. If you don't want to perform standard prediction add `--no_standard_prediction` argument.

- If you want to perform prediction using a COCO annotation file, provide COCO json path as `--dataset_json_path dataset.json` and coco image folder as `--source path/to/coco/image/folder`, predictions will be exported as a coco json file to runs/predict/exp/results.json. Then you can use coco_evaluation command to calculate COCO evaluation results or coco_error_analysis command to calculate detailed COCO error plots.

## `predict-fiftyone` command usage:

```bash
>>sahi predict-fiftyone --image_dir image/file/or/folder --dataset_json_path dataset.json --model_path path/to/model --model_config_path path/to/config
```

will perform sliced inference on default parameters and show the inference result on FiftyOne App.

You can specify additional all extra parameters of the [sahi predict](https://github.com/obss/sahi/blob/main/docs/CLI.md#predict-command-usage) command.

## `coco fiftyone` command usage:

You need to convert your predictions into [COCO result json](https://cocodataset.org/#format-results), [sahi predict](https://github.com/obss/sahi/blob/main/docs/CLI.md#predict-command-usage) command can be used to create that.

```bash
>>sahi coco fiftyone --image_dir dir/to/images --dataset_json_path dataset.json cocoresult1.json cocoresult2.json
```

will open a FiftyOne app that visualizes the given dataset and 2 detection results.

Specify IOU threshold for FP/TP by `--iou_threshold 0.5` argument

## `coco slice` command usage:

```bash
>>sahi coco slice --image_dir dir/to/images --dataset_json_path dataset.json
```

will slice the given images and COCO formatted annotations and export them to given output folder directory.

Specify slice height/width size as `--slice_size 512`.

Specify slice overlap ratio for height/width size as `--overlap_ratio 0.2`.

If you want to ignore images with annotations set it add `--ignore_negative_samples` argument.

## `coco yolo` command usage:

(In Windows be sure to open anaconda cmd prompt/windows cmd `as admin` to be able to create symlinks properly.)

```bash
>>sahi coco yolo --image_dir dir/to/images --dataset_json_path dataset.json  --train_split 0.9
```

will convert given coco dataset to yolo format and export to runs/coco2yolo/exp folder.

## `coco evaluate` command usage:

You need to convert your predictions into [COCO result json](https://cocodataset.org/#format-results), [sahi predict](https://github.com/obss/sahi/blob/main/docs/CLI.md#predict-command-usage) command can be used to create that.

```bash
>>sahi coco evaluate --dataset_json_path dataset.json --result_json_path result.json
```

will calculate coco evaluation and export them to given output folder directory.

If you want to specify mAP metric type, set it as `--type bbox` or `--type mask`.

If you want to also calculate classwise scores add `--classwise` argument.

If you want to specify max detections, set it as `--proposal_nums "[10 100 500]"`.

If you want to specify a specific IOU threshold, set it as `--iou_thrs 0.5`. Default includes `0.50:0.95` and `0.5` scores.

If you want to specify an export directory, set it as `--out_dir output/folder/directory`.

## `coco analyse` command usage:

You need to convert your predictions into [COCO result json](https://cocodataset.org/#format-results), [sahi predict](https://github.com/obss/sahi/blob/main/docs/cli.md#predict-command-usage) command can be used to create that.

```bash
>>sahi coco analyse --dataset_json_path dataset.json --result_json_path result.json --out_dir output/directory
```

will calculate coco error plots and export them to given output folder directory.

If you want to specify mAP result type, set it as `--type bbox` or `--type segm`.

If you want to export extra mAP bar plots and annotation area stats add `--extraplots` argument.

If you want to specify area regions, set it as `--areas "[1024 9216 10000000000]"`.

## `env` command usage:

Print related package versions in the current env as:

```bash
>>sahi env
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   torch version 2.1.2 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   torchvision version 0.16.2 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   ultralytics version 8.3.86 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   transformers version 4.49.0 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   timm version 0.9.1 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   fiftyone version 0.14.2 is available.
```

## `version` command usage:

Print your SAHI version as:

```bash
>>sahi version
0.11.22
```

## Custom scripts

All scripts can be downloaded from [scripts directory](https://github.com/obss/sahi/tree/main/scripts) and modified by your needs. After installing `sahi` by pip, all scripts can be called from any directory as:

```bash
python script_name.py
```

# Additional Resources

Looking to dive deeper? Here are some helpful resources:

- For a detailed walkthrough of prediction parameters and visualization, check out our [prediction utilities documentation](predict.md)
- To understand slicing operations in depth, explore our [slicing utilities guide](slicing.md)
- For hands-on examples with COCO format operations, see our [COCO utilities documentation](coco.md)
- Want to see these CLI commands in action? Try our interactive notebooks in the [demo directory](../demo/)

These resources provide comprehensive examples and explanations to help you make the most of SAHI's command-line interface.
