import argparse

from sahi.predict import predict_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="", help="image directory")
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
        "--prediction_score_threshold",
        type=float,
        default=0.25,
        help="all predictions with score < prediction_score_threshold will be discarded",
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
        "--visual_output_dir", type=str, default="", help="output visual directory"
    )
    parser.add_argument(
        "--pickle_dir", type=str, default="", help="output pickle directory"
    )
    parser.add_argument(
        "--crop_dir", type=str, default="", help="output crop directory"
    )
    parser.add_argument("--apply_sliced_prediction", type=bool, default=True)
    parser.add_argument("--slice_height", type=int, default=512)
    parser.add_argument("--slice_width", type=int, default=512)
    parser.add_argument("--overlap_height_ratio", type=float, default=0.2)
    parser.add_argument("--overlap_width_ratio", type=float, default=0.2)
    parser.add_argument("--match_iou_threshold", type=float, default=0.5)

    opt = parser.parse_args()

    model_parameters = {
        "model_path": opt.model_path,
        "config_path": opt.config_path,
        "prediction_score_threshold": opt.prediction_score_threshold,
        "device": opt.device,
        "category_mapping": opt.category_mapping,
        "category_remapping": opt.category_remapping,
    }
    predict_folder(
        model_name=opt.model_name,
        model_parameters=model_parameters,
        image_dir=opt.image_dir,
        visual_output_dir=opt.visual_output_dir,
        pickle_dir=opt.pickle_dir,
        crop_dir=opt.crop_dir,
        apply_sliced_prediction=opt.apply_sliced_prediction,
        slice_height=opt.slice_height,
        slice_width=opt.slice_width,
        overlap_height_ratio=opt.overlap_height_ratio,
        overlap_width_ratio=opt.overlap_width_ratio,
        match_iou_threshold=opt.match_iou_threshold,
    )
