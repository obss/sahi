import fire

from sahi import __version__ as sahi_version
from sahi.predict import predict, predict_fiftyone
from sahi.scripts.coco2fiftyone import main as coco2fiftyone
from sahi.scripts.coco2yolov5 import main as coco2yolov5
from sahi.scripts.coco_error_analysis import main as coco_error_analysis
from sahi.scripts.coco_evaluation import main as coco_evaluation
from sahi.scripts.slice_coco import main as slice_coco

coco_app = {
    "evaluate": coco_evaluation,
    "analyse": coco_error_analysis,
    "fiftyone": coco2fiftyone,
    "slice": slice_coco,
    "yolov5": coco2yolov5,
}

sahi_app = {"predict": predict, "predict-fiftyone": predict_fiftyone, "coco": coco_app, "version": sahi_version}


def app() -> None:
    """Cli app."""
    fire.Fire(sahi_app)


if __name__ == "__main__":
    app()
