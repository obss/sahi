import os

import fiftyone as fo
from fiftyone.utils.coco import COCODetectionDatasetImporter as BaseCOCODetectionDatasetImporter
from fiftyone.utils.coco import load_coco_detection_annotations


class COCODetectionDatasetImporter(BaseCOCODetectionDatasetImporter):
    def __init__(
        self,
        image_dir,
        json_path,
        load_segmentations=True,
        return_polylines=False,
        tolerance=None,
        skip_unlabeled=False,
        shuffle=False,
        seed=None,
        max_samples=None,
    ):
        super().__init__(
            dataset_dir="",
            skip_unlabeled=skip_unlabeled,
            shuffle=shuffle,
            seed=seed,
            max_samples=max_samples,
        )
        self.json_path = json_path
        self.load_segmentations = load_segmentations
        self.return_polylines = return_polylines
        self.tolerance = tolerance
        self._data_dir = image_dir
        self._info = None
        self._classes = None
        self._supercategory_map = None
        self._images_map = None
        self._annotations = None
        self._filenames = None
        self._iter_filenames = None

    def setup(self):
        self._data_dir = os.path.join(self._data_dir)

        labels_path = os.path.join(self.json_path)
        if os.path.isfile(labels_path):
            (
                info,
                classes,
                supercategory_map,
                images,
                annotations,
            ) = load_coco_detection_annotations(labels_path)
        else:
            info = {}
            classes = None
            supercategory_map = None
            images = {}
            annotations = None

        if classes is not None:
            info["classes"] = classes

        self._info = info
        self._classes = classes
        self._supercategory_map = supercategory_map
        self._images_map = {i["file_name"]: i for i in images.values()}
        self._annotations = annotations

        if self.skip_unlabeled:
            filenames = self._images_map.keys()
        else:
            filenames = [image["file_name"] for image in images.values()]

        self._filenames = self._preprocess_list(filenames)


def launch_fiftyone_app(coco_image_dir: str, coco_json_path: str):
    coco_importer = COCODetectionDatasetImporter(image_dir=coco_image_dir, json_path=coco_json_path)
    dataset = fo.Dataset.from_importer(coco_importer)
    session = fo.launch_app()
    session.dataset = dataset
    return session
