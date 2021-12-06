try:
    import fiftyone as fo
    from fiftyone.utils.coco import COCODetectionDatasetImporter
except ModuleNotFoundError:
    raise ModuleNotFoundError('Please run "pip install -U fiftyone" to install fiftyone first for fiftyone utilities.')


def create_fiftyone_dataset_from_coco_file(coco_image_dir: str, coco_json_path: str):
    coco_importer = COCODetectionDatasetImporter(data_path=coco_image_dir, labels_path=coco_json_path)
    dataset = fo.Dataset.from_importer(coco_importer)
    return dataset


def launch_fiftyone_app(coco_image_dir: str, coco_json_path: str):
    dataset = create_fiftyone_dataset_from_coco_file(coco_image_dir, coco_json_path)
    session = fo.launch_app()
    session.dataset = dataset
    return session
