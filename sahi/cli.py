import fire

from sahi import __version__ as sahi_version
from sahi.predict import predict, predict_fiftyone
from sahi.scripts.coco2fiftyone import main as coco2fiftyone
from sahi.scripts.coco2yolo import main as coco2yolo
from sahi.scripts.coco_error_analysis import analyse
from sahi.scripts.coco_evaluation import evaluate
from sahi.scripts.slice_coco import slice
from sahi.utils.import_utils import print_environment_info

coco_app = {
    "evaluate": evaluate,
    "analyse": analyse,
    "fiftyone": coco2fiftyone,
    "slice": slice,
    "yolo": coco2yolo,
    "yolov5": coco2yolo,
}

sahi_app = {
    "predict": predict,
    "predict-fiftyone": predict_fiftyone,
    "coco": coco_app,
    "version": sahi_version,
    "env": print_environment_info,
}


def app() -> None:
    """Cli app."""
    fire.Fire(sahi_app)


if __name__ == "__main__":
    app()
