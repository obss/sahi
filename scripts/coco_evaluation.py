import itertools
import json
import warnings
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path

import numpy as np
from terminaltables import AsciiTable

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ImportError:
    raise ImportError('Please run "pip install -U pycocotools" ' "to install pycocotools first for coco evaluation.")


def _cocoeval_summarize(
    cocoeval, ap=1, iouThr=None, catIdx=None, areaRng="all", maxDets=100, catName="", nameStrLen=None
):
    p = cocoeval.params
    if catName:
        iStr = " {:<18} {} {:<{nameStrLen}} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
        nameStr = catName
    else:
        iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}"
    titleStr = "Average Precision" if ap == 1 else "Average Recall"
    typeStr = "(AP)" if ap == 1 else "(AR)"
    iouStr = "{:0.2f}:{:0.2f}".format(p.iouThrs[0], p.iouThrs[-1]) if iouThr is None else "{:0.2f}".format(iouThr)

    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
    if ap == 1:
        # dimension of precision: [TxRxKxAxM]
        s = cocoeval.eval["precision"]
        # IoU
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        if catIdx is not None:
            s = s[:, :, catIdx, aind, mind]
        else:
            s = s[:, :, :, aind, mind]
    else:
        # dimension of recall: [TxKxAxM]
        s = cocoeval.eval["recall"]
        if iouThr is not None:
            t = np.where(iouThr == p.iouThrs)[0]
            s = s[t]
        if catIdx is not None:
            s = s[:, catIdx, aind, mind]
        else:
            s = s[:, :, aind, mind]
    if len(s[s > -1]) == 0:
        mean_s = -1
    else:
        mean_s = np.mean(s[s > -1])
    if catName:
        print(iStr.format(titleStr, typeStr, nameStr, iouStr, areaRng, maxDets, mean_s, nameStrLen=nameStrLen))
    else:
        print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
    return mean_s


def evaluate_coco(
    dataset_path,
    result_path,
    metric="bbox",
    classwise=False,
    proposal_nums=(10, 100, 500),
    iou_thrs=None,
    metric_items=None,
    out_dir=None,
):
    """Evaluation in COCO protocol.
    Args:
        dataset_path (str): COCO dataset json path.
        result_path (str): COCO result json path.
        metric (str | list[str]): Metrics to be evaluated. Options are
            'bbox', 'segm', 'proposal'.
        classwise (bool): Whether to evaluating the AP for each class.
        proposal_nums (Sequence[int]): Proposal number used for evaluating
            recalls, such as recall@100, recall@500.
            Default: (10, 100, 500).
        iou_thrs (Sequence[float], optional): IoU threshold used for
            evaluating recalls/mAPs. If set to a list, the average of all
            IoUs will also be computed. If not specified, [0.50, 0.55,
            0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
            Default: None.
        metric_items (list[str] | str, optional): Metric items that will
            be returned. If not specified, ``['AR@10', 'AR@100',
            'AR@500', 'AR_s@500', 'AR_m@500', 'AR_l@500' ]`` will be
            used when ``metric=='proposal'``, ``['mAP', 'mAP50', 'mAP75',
            'mAP_s', 'mAP_m', 'mAP_l', 'mAP50_s', 'mAP50_m', 'mAP50_l']``
            will be used when ``metric=='bbox' or metric=='segm'``.
        out_dir (str): Directory to save evaluation result json.
    Returns:
        dict[str, float]: COCO style evaluation metric.
    """

    metrics = metric if isinstance(metric, list) else [metric]
    allowed_metrics = ["bbox", "segm", "proposal"]
    for metric in metrics:
        if metric not in allowed_metrics:
            raise KeyError(f"metric {metric} is not supported")
    if iou_thrs is None:
        iou_thrs = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
    if metric_items is not None:
        if not isinstance(metric_items, list):
            metric_items = [metric_items]

    eval_results = OrderedDict()
    cocoGt = COCO(dataset_path)
    cat_ids = list(cocoGt.cats.keys())
    for metric in metrics:
        msg = f"Evaluating {metric}..."
        msg = "\n" + msg
        print(msg)

        iou_type = metric
        with open(result_path) as json_file:
            results = json.load(json_file)
        try:
            if iou_type == "segm":
                # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                # When evaluating mask AP, if the results contain bbox,
                # cocoapi will use the box area instead of the mask area
                # for calculating the instance area. Though the overall AP
                # is not affected, this leads to different
                # small/medium/large mask AP results.
                for x in results:
                    x.pop("bbox")
                warnings.simplefilter("once")
                warnings.warn(
                    'The key "bbox" is deleted for more accurate mask AP '
                    "of small/medium/large instances since v2.12.0. This "
                    "does not change the overall mAP calculation.",
                    UserWarning,
                )
            cocoDt = cocoGt.loadRes(results)
        except IndexError:
            print("The testing results of the whole dataset is empty.")
            break

        cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
        cocoEval.params.catIds = cat_ids
        cocoEval.params.maxDets = list(proposal_nums)
        cocoEval.params.iouThrs = iou_thrs
        # mapping of cocoEval.stats
        coco_metric_names = {
            "mAP": 0,
            "mAP50": 1,
            "mAP75": 2,
            "mAP_s": 3,
            "mAP_m": 4,
            "mAP_l": 5,
            "AR@10": 6,
            "AR@100": 7,
            "AR@500": 8,
            "AR_s@500": 9,
            "AR_m@500": 10,
            "AR_l@500": 11,
            "mAP50_s": 12,
            "mAP50_m": 13,
            "mAP50_l": 14,
        }
        if metric_items is not None:
            for metric_item in metric_items:
                if metric_item not in coco_metric_names:
                    raise KeyError(f"metric item {metric_item} is not supported")
        if metric == "proposal":
            cocoEval.params.useCats = 0
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            if metric_items is None:
                metric_items = ["AR@10", "AR@100", "AR@500", "AR_s@500", "AR_m@500", "AR_l@500"]

            for item in metric_items:
                val = float(f"{cocoEval.stats[coco_metric_names[item]]:.3f}")
                eval_results[item] = val
        else:
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # calculate mAP50_s/m/l
            mAP50_s = _cocoeval_summarize(
                cocoEval, ap=1, iouThr=0.5, areaRng="small", maxDets=cocoEval.params.maxDets[-1]
            )
            mAP50_m = _cocoeval_summarize(
                cocoEval, ap=1, iouThr=0.5, areaRng="medium", maxDets=cocoEval.params.maxDets[-1]
            )
            mAP50_l = _cocoeval_summarize(
                cocoEval, ap=1, iouThr=0.5, areaRng="large", maxDets=cocoEval.params.maxDets[-1]
            )
            cocoEval.stats = np.append(cocoEval.stats, [mAP50_s, mAP50_m, mAP50_l], 0)

            if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
                precisions = cocoEval.eval["precision"]
                # precision: (iou, recall, cls, area range, max dets)
                assert len(cat_ids) == precisions.shape[2]

                max_cat_name_len = 0
                for idx, catId in enumerate(cat_ids):
                    nm = cocoGt.loadCats(catId)[0]
                    cat_name_len = len(nm["name"])
                    max_cat_name_len = cat_name_len if cat_name_len > max_cat_name_len else max_cat_name_len

                results_per_category = []
                for idx, catId in enumerate(cat_ids):
                    # area range index 0: all area ranges
                    # max dets index -1: typically 100 per image
                    nm = cocoGt.loadCats(catId)[0]
                    ap = _cocoeval_summarize(
                        cocoEval,
                        ap=1,
                        catIdx=idx,
                        areaRng="all",
                        maxDets=cocoEval.params.maxDets[-1],
                        catName=nm["name"],
                        nameStrLen=max_cat_name_len,
                    )
                    ap_s = _cocoeval_summarize(
                        cocoEval,
                        ap=1,
                        catIdx=idx,
                        areaRng="small",
                        maxDets=cocoEval.params.maxDets[-1],
                        catName=nm["name"],
                        nameStrLen=max_cat_name_len,
                    )
                    ap_m = _cocoeval_summarize(
                        cocoEval,
                        ap=1,
                        catIdx=idx,
                        areaRng="medium",
                        maxDets=cocoEval.params.maxDets[-1],
                        catName=nm["name"],
                        nameStrLen=max_cat_name_len,
                    )
                    ap_l = _cocoeval_summarize(
                        cocoEval,
                        ap=1,
                        catIdx=idx,
                        areaRng="large",
                        maxDets=cocoEval.params.maxDets[-1],
                        catName=nm["name"],
                        nameStrLen=max_cat_name_len,
                    )
                    ap50 = _cocoeval_summarize(
                        cocoEval,
                        ap=1,
                        iouThr=0.5,
                        catIdx=idx,
                        areaRng="all",
                        maxDets=cocoEval.params.maxDets[-1],
                        catName=nm["name"],
                        nameStrLen=max_cat_name_len,
                    )
                    ap50_s = _cocoeval_summarize(
                        cocoEval,
                        ap=1,
                        iouThr=0.5,
                        catIdx=idx,
                        areaRng="small",
                        maxDets=cocoEval.params.maxDets[-1],
                        catName=nm["name"],
                        nameStrLen=max_cat_name_len,
                    )
                    ap50_m = _cocoeval_summarize(
                        cocoEval,
                        ap=1,
                        iouThr=0.5,
                        catIdx=idx,
                        areaRng="medium",
                        maxDets=cocoEval.params.maxDets[-1],
                        catName=nm["name"],
                        nameStrLen=max_cat_name_len,
                    )
                    ap50_l = _cocoeval_summarize(
                        cocoEval,
                        ap=1,
                        iouThr=0.5,
                        catIdx=idx,
                        areaRng="large",
                        maxDets=cocoEval.params.maxDets[-1],
                        catName=nm["name"],
                        nameStrLen=max_cat_name_len,
                    )
                    results_per_category.append((f'{metric}_{nm["name"]}_mAP', f"{float(ap):0.3f}"))
                    results_per_category.append((f'{metric}_{nm["name"]}_mAP_s', f"{float(ap_s):0.3f}"))
                    results_per_category.append((f'{metric}_{nm["name"]}_mAP_m', f"{float(ap_m):0.3f}"))
                    results_per_category.append((f'{metric}_{nm["name"]}_mAP_l', f"{float(ap_l):0.3f}"))
                    results_per_category.append((f'{metric}_{nm["name"]}_mAP50', f"{float(ap50):0.3f}"))
                    results_per_category.append((f'{metric}_{nm["name"]}_mAP50_s', f"{float(ap50_s):0.3f}"))
                    results_per_category.append((f'{metric}_{nm["name"]}_mAP50_m', f"{float(ap50_m):0.3f}"))
                    results_per_category.append((f'{metric}_{nm["name"]}_mAP50_l', f"{float(ap50_l):0.3f}"))

                num_columns = min(6, len(results_per_category) * 2)
                results_flatten = list(itertools.chain(*results_per_category))
                headers = ["category", "AP"] * (num_columns // 2)
                results_2d = itertools.zip_longest(*[results_flatten[i::num_columns] for i in range(num_columns)])
                table_data = [headers]
                table_data += [result for result in results_2d]
                table = AsciiTable(table_data)
                print("\n" + table.table)

            if metric_items is None:
                metric_items = ["mAP", "mAP50", "mAP75", "mAP_s", "mAP_m", "mAP_l", "mAP50_s", "mAP50_m", "mAP50_l"]

            for metric_item in metric_items:
                key = f"{metric}_{metric_item}"
                val = float(f"{cocoEval.stats[coco_metric_names[metric_item]]:.3f}")
                eval_results[key] = val
            ap = cocoEval.stats[:6]
            eval_results[f"{metric}_mAP_copypaste"] = (
                f"{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} " f"{ap[4]:.3f} {ap[5]:.3f}"
            )
            if classwise:
                eval_results["results_per_category"] = {key: value for key, value in results_per_category}
    # set save path
    if not out_dir:
        out_dir = Path(result_path).parent
    save_path = str(out_dir / "eval.json")
    # export as json
    with open(save_path, "w", encoding="utf-8") as outfile:
        json.dump(eval_results, outfile, indent=4, separators=(",", ":"))
    return eval_results


def main():
    parser = ArgumentParser(description="COCO Evaluation Tool")
    parser.add_argument("dataset_path", type=str, help="COCO dataset (json file) path")
    parser.add_argument("result_path", type=str, help="COCO result (json file) path")
    parser.add_argument("--out_dir", type=str, default=None, help="dir to save evaluation result json")
    parser.add_argument("--metric", type=str, nargs="+", default=["bbox"], help="metric types")
    parser.add_argument("--classwise", action="store_true", help="whether to evaluate the AP for each class")
    parser.add_argument(
        "--proposal_nums",
        type=int,
        nargs="+",
        default=[10, 100, 500],
        help="Proposal number used for evaluating recalls, such as recall@100, recall@500",
    )
    parser.add_argument("--iou_thrs", type=float, default=None, help="IoU threshold used for evaluating recalls/mAPs")
    args = parser.parse_args()
    # perform coco eval
    eval_results = evaluate_coco(
        args.dataset_path,
        args.result_path,
        args.metric,
        args.classwise,
        args.proposal_nums,
        args.iou_thrs,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
