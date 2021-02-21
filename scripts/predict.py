import argparse

from sahi.predict import predict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="", help="image file or folder")
    parser.add_argument(
        "--model_name",
        type=str,
        default="MmdetDetectionModel",
        help="name for the detection model",
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
    parser.add_argument(
        "--project", default="runs/predict", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--pickle", action="store_true", help="export predictions as .pickle"
    )
    parser.add_argument(
        "--crop", action="store_true", help="export predictions as cropped images"
    )
    parser.add_argument(
        "--sliced_pred", action="store_false", help="apply sliced prediction"
    )
    parser.add_argument("--slice_height", type=int, default=512)
    parser.add_argument("--slice_width", type=int, default=512)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--visual_export_format", type=str, default="png")

    opt = parser.parse_args()

    model_parameters = {
        "model_path": opt.model_path,
        "config_path": opt.config_path,
        "prediction_score_threshold": opt.conf_thresh,
        "device": opt.device,
        "category_mapping": opt.category_mapping,
        "category_remapping": opt.category_remapping,
    }
    predict(
        model_name=opt.model_name,
        model_parameters=model_parameters,
        source=opt.source,
        project=opt.project,
        name=opt.name,
        export_pickle=opt.pickle,
        export_crop=opt.crop,
        apply_sliced_prediction=opt.sliced_pred,
        slice_height=opt.slice_height,
        slice_width=opt.slice_width,
        overlap_height_ratio=opt.overlap_height_ratio,
        overlap_width_ratio=opt.overlap_width_ratio,
        match_iou_threshold=opt.iou_thresh,
        visual_export_format=opt.visual_export_format,
    )
