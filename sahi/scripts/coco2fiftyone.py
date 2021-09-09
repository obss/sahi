import time
from collections import defaultdict
from pathlib import Path
from typing import List

import fire

from sahi.utils.cv import read_image_as_pil
from sahi.utils.file import load_json


def main(
    image_dir: str,
    dataset_json_path: str,
    *result_json_paths,
    iou_thresh: float = 0.5,
):
    """
    Args:
        image_dir (str): directory for coco images
        dataset_json_path (str): file path for the coco dataset json file
        result_json_paths (str): one or more paths for the coco result json file
        iou_thresh (str): iou threshold for coco evaluation
    """

    from sahi.utils.fiftyone import create_fiftyone_dataset_from_coco_file, fo

    coco_result_list = []
    image_id_to_coco_result_list = []
    result_name_list = []
    if result_json_paths:
        for ind, result_json_path in enumerate(result_json_paths):
            coco_result = load_json(result_json_path)
            coco_result_list.append(coco_result)

            image_id_to_coco_result = defaultdict(list)
            for result in coco_result:
                image_id_to_coco_result[result["image_id"]].append(result)
            image_id_to_coco_result_list.append(image_id_to_coco_result)

            # use file names as fiftyone name, create unique names if duplicate
            result_name_temp = Path(result_json_path).stem
            result_name = result_name_temp
            name_increment = 2
            while result_name in result_name_list:
                result_name = result_name_temp + "_" + str(name_increment)
                name_increment += 1
            result_name_list.append(result_name)

    dataset = create_fiftyone_dataset_from_coco_file(image_dir, dataset_json_path)

    # submit detections if coco result is given
    if result_json_paths:
        image_id = 1
        with fo.ProgressBar() as pb:
            for sample in pb(dataset):
                image_pil = read_image_as_pil(sample.filepath)

                # iterate over multiple coco results
                for ind, image_id_to_coco_result in enumerate(image_id_to_coco_result_list):
                    coco_result_list = image_id_to_coco_result[image_id]
                    fo_detection_list: List[fo.Detection] = []

                    for coco_result in coco_result_list:
                        fo_x = coco_result["bbox"][0] / image_pil.width
                        fo_y = coco_result["bbox"][1] / image_pil.height
                        fo_w = coco_result["bbox"][2] / image_pil.width
                        fo_h = coco_result["bbox"][3] / image_pil.height
                        fo_bbox = [fo_x, fo_y, fo_w, fo_h]
                        fo_label = dataset.default_classes[coco_result["category_id"]]
                        fo_confidence = coco_result["score"]

                        fo_detection_list.append(
                            fo.Detection(label=fo_label, bounding_box=fo_bbox, confidence=fo_confidence)
                        )

                    sample[result_name_list[ind]] = fo.Detections(detections=fo_detection_list)
                    sample.save()
                image_id += 1

    # visualize results
    session = fo.launch_app()
    session.dataset = dataset

    # order by false positives if any coco result is given
    if result_json_paths:
        # Evaluate the predictions
        first_coco_result_name = result_name_list[0]
        results = dataset.evaluate_detections(
            first_coco_result_name,
            gt_field="ground_truth",
            eval_key=f"{first_coco_result_name}_eval",
            iou=iou_thresh,
            compute_mAP=True,
        )
        # Get the 10 most common classes in the dataset
        counts = dataset.count_values("ground_truth.detections.label")
        classes_top10 = sorted(counts, key=counts.get, reverse=True)[:10]
        # Print a classification report for the top-10 classes
        results.print_report(classes=classes_top10)
        # Load the view on which we ran the `eval` evaluation
        eval_view = dataset.load_evaluation_view(f"{first_coco_result_name}_eval")
        # Show samples with most false positives
        session.view = eval_view.sort_by(f"{first_coco_result_name}_eval_fp", reverse=True)
    while 1:
        time.sleep(3)


if __name__ == "__main__":
    fire.Fire(main)
