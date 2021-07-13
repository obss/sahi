import argparse

from sahi.predict import predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="", help="image file or folder")
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
    parser.add_argument("--project", default="runs/predict", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--novisual", action="store_true", help="export prediction visualizations")
    parser.add_argument("--pickle", action="store_true", help="export predictions as .pickle")
    parser.add_argument("--crop", action="store_true", help="export predictions as cropped images")
    parser.add_argument(
        "--coco_file",
        type=str,
        default=None,
        help="perform detection from coco file and export results in coco json format",
    )
    parser.add_argument("--no_sliced_pred", action="store_true", help="dont apply sliced prediction")
    parser.add_argument("--no_standard_pred", action="store_true", help="dont apply standard prediction")
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
    parser.add_argument("--visual_export_format", type=str, default="png")

    opt = parser.parse_args()

    predict(
        model_type=opt.model_type,
        model_path=opt.model_path,
        model_config_path=opt.config_path,
        model_confidence_threshold=opt.conf_thresh,
        model_device=opt.device,
        model_category_mapping=opt.category_mapping,
        model_category_remapping=opt.category_remapping,
        source=opt.source,
        project=opt.project,
        name=opt.name,
        export_visual=not (opt.novisual),
        export_pickle=opt.pickle,
        export_crop=opt.crop,
        coco_file_path=opt.coco_file,
        no_standard_prediction=opt.no_standard_pred,
        no_sliced_prediction=opt.no_sliced_pred,
        slice_height=opt.slice_height,
        slice_width=opt.slice_width,
        overlap_height_ratio=opt.overlap_height_ratio,
        overlap_width_ratio=opt.overlap_width_ratio,
        postprocess_type=opt.postprocess_type,
        postprocess_match_metric=opt.match_metric,
        postprocess_match_threshold=opt.match_thresh,
        postprocess_class_agnostic=opt.class_agnostic,
        visual_export_format=opt.visual_export_format,
    )
