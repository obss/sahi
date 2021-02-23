import argparse

from sahi.utils.coco import Coco
from sahi.utils.file import Path, increment_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, default="", help="directory for coco images"
    )
    parser.add_argument(
        "--coco_file",
        type=str,
        default=None,
        help="file path for the coco file to be converted",
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.25,
        help="set the training split ratio",
    )
    parser.add_argument(
        "--project", default="runs/coco2yolov5", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--seed", type=int, default=1, help="fix the seed for reproducibility"
    )

    opt = parser.parse_args()

    # increment run
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=False))
    # load coco dict
    coco = Coco(coco_dict_or_path=opt.coco_file)
    # export as yolov5
    coco.export_as_yolov5(
        image_dir=opt.source,
        output_dir=str(save_dir),
        train_split_rate=opt.train_split,
        numpy_seed=opt.seed,
    )
