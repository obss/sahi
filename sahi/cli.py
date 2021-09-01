import fire

from sahi import __version__ as sahi_version
from sahi.scripts.coco2yolov5 import main as coco2yolov5
from sahi.scripts.coco_error_analysis import main as coco_error_analysis
from sahi.scripts.coco_evaluation import main as coco_evaluation
from sahi.scripts.cocoresult2fiftyone import main as cocoresult2fiftyone
from sahi.scripts.predict import main as predict
from sahi.scripts.predict_fiftyone import main as predict_fiftyone
from sahi.scripts.slice_coco import main as slice_coco


def app() -> None:
    """Cli app."""
    fire.Fire(
        {
            "version": sahi_version,
            "coco_error_analysis": coco_error_analysis,
            "coco_evaluation": coco_evaluation,
            "coco2yolov5": coco2yolov5,
            "predict": predict,
            "predict_fiftyone": predict_fiftyone,
            "cocoresult2fiftyone": cocoresult2fiftyone,
            "slice_coco": slice_coco,
        }
    )


if __name__ == "__main__":
    app()
