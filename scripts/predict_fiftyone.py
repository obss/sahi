import argparse

from sahi.predict import predict_fiftyone

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_image_dir", type=str, default="", help="folder containing images")
    parser.add_argument(
        "--coco_json_path",
        type=str,
        default=None,
        help="perform detection from coco file and export results in coco json format",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="mmdet",
        help="mmdet for 'MmdetDetectionModel', 'yolov5' for 'Yolov5DetectionModel'",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="path for the model",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="",
        help="path for the model config",
    )
    parser.add_argument(
        "--conf_thresh",
        type=float,
        default=0.25,
        help="all predictions with score < conf_thresh will be discarded",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu or cuda",
    )
    parser.add_argument(
        "--category_mapping",
        type=str,
        default=None,
        help='mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}',
    )
    parser.add_argument(
        "--category_remapping",
        type=str,
        default=None,
        help='remap category ids based on category names, after performing inference e.g. {"car": 3}',
    )
    parser.add_argument("--standard_pred", action="store_true", help="dont apply sliced prediction")
    parser.add_argument("--slice_height", type=int, default=512)
    parser.add_argument("--slice_width", type=int, default=512)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2)
    parser.add_argument(
        "--postprocess_type", type=str, default="UNIONMERGE", help="postprocess type: 'UNIONMERGE' or 'NMS'"
    )
    parser.add_argument("--match_metric", type=str, default="IOS", help="match metric for postprocess: 'IOU' or 'IOS'")
    parser.add_argument("--match_thresh", type=float, default=0.5, help="match threshold for postprocess")
    parser.add_argument(
        "--class_agnostic",
        action="store_true",
        help="Postprocess will ignore category ids.",
    )
    opt = parser.parse_args()

    model_type_to_model_name = {
        "mmdet": "MmdetDetectionModel",
        "yolov5": "Yolov5DetectionModel",
    }

    model_parameters = {
        "model_path": opt.model_path,
        "config_path": opt.config_path,
        "prediction_score_threshold": opt.conf_thresh,
        "device": opt.device,
        "category_mapping": opt.category_mapping,
        "category_remapping": opt.category_remapping,
    }
    predict_fiftyone(
        model_name=model_type_to_model_name[opt.model_type],
        model_parameters=model_parameters,
        coco_json_path=opt.coco_json_path,
        coco_image_dir=opt.coco_image_dir,
        apply_sliced_prediction=not (opt.standard_pred),
        slice_height=opt.slice_height,
        slice_width=opt.slice_width,
        overlap_height_ratio=opt.overlap_height_ratio,
        overlap_width_ratio=opt.overlap_width_ratio,
        postprocess_type=opt.postprocess_type,
        postprocess_match_metric=opt.match_metric,
        postprocess_match_threshold=opt.match_thresh,
        postprocess_class_agnostic=opt.class_agnostic,
    )
