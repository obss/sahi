import argparse
import time
from collections import defaultdict
from typing import List

import fiftyone as fo

from sahi.utils.cv import read_image_as_pil
from sahi.utils.fiftyone import create_fiftyone_dataset_from_coco_file
from sahi.utils.file import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_image_dir", type=str, default=None, help="folder containing coco images")
    parser.add_argument(
        "--coco_json_path",
        type=str,
        default=None,
        help="path for the coco annotation json file",
    )
    parser.add_argument(
        "--coco_result_path",
        type=str,
        nargs="+",
        default=[],
        help="one or more paths for the coco result json file",
    )
    parser.add_argument(
        "--coco_result_name",
        type=str,
        nargs="+",
        default=[],
        help="will be used in the fiftyone visualization",
    )
    parser.add_argument(
        "--iou_thresh",
        type=float,
        default=0.5,
        help="iou threshold for coco evaluation",
    )

    opt = parser.parse_args()

    if opt.coco_result_name and not (len(opt.coco_result_path) == len(opt.coco_result_name)):
        raise ValueError("'coco_result_path' and 'coco_result_name' should be in same length")

    coco_result_list = []
    image_id_to_coco_result_list = []
    coco_result_name_list = opt.coco_result_name if opt.coco_result_name else []
    for ind, coco_result_path in enumerate(opt.coco_result_path):
        coco_result = load_json(coco_result_path)
        coco_result_list.append(coco_result)

        image_id_to_coco_result = defaultdict(list)
        for result in coco_result:
            image_id_to_coco_result[result["image_id"]].append(result)
        image_id_to_coco_result_list.append(image_id_to_coco_result)

        if not opt.coco_result_name:
            coco_result_name_list.append(f"coco_result_{ind+1}")

    dataset = create_fiftyone_dataset_from_coco_file(opt.coco_image_dir, opt.coco_json_path)

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

                sample[coco_result_name_list[ind]] = fo.Detections(detections=fo_detection_list)
                sample.save()
            image_id += 1

    # visualize results
    session = fo.launch_app()
    session.dataset = dataset
    # Evaluate the predictions
    first_coco_result_name = coco_result_name_list[0]
    results = dataset.evaluate_detections(
        first_coco_result_name,
        gt_field="ground_truth",
        eval_key=f"{first_coco_result_name}_eval",
        iou=opt.iou_thresh,
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
    main()
