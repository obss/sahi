import copy
import logging
import os
from multiprocessing import Pool
from pathlib import Path
from typing import List, Optional, Union
import json

import fire
import numpy as np

logger = logging.getLogger(__name__)

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    has_pycocotools = True
except ImportError:
    has_pycocotools = False

try:
    import matplotlib.pyplot as plt

    has_matplotlib = True
except ImportError:
    has_matplotlib = False


COLOR_PALETTE = np.vstack(
    [
        np.array([0.8, 0.8, 0.8]),
        np.array([0.6, 0.6, 0.6]),
        np.array([0.31, 0.51, 0.74]),
        np.array([0.75, 0.31, 0.30]),
        np.array([0.36, 0.90, 0.38]),
        np.array([0.50, 0.39, 0.64]),
        np.array([1, 0.6, 0]),
    ]
)


def _makeplot(rs, ps, outDir: Union[str, Path], class_name: str, iou_type: str) -> List[str]:
    export_path_list = []

    areaNames = ["allarea", "small", "medium", "large"]
    types = ["C75", "C50", "Loc", "Sim", "Oth", "BG", "FN"]
    for i in range(len(areaNames)):
        area_ps = ps[..., i, 0]
        figure_title = iou_type + "-" + class_name + "-" + areaNames[i]
        aps = []
        ps_curve = []
        for ps_ in area_ps:
            # calculate precision recal curves
            if ps_.ndim > 1:
                ps_mean = np.zeros((ps_.shape[0],))
                for ind, ps_threshold in enumerate(ps_):
                    ps_mean[ind] = ps_threshold[ps_threshold > -1].mean()
                ps_curve.append(ps_mean)
            else:
                ps_curve.append(ps_)
            # calculate ap
            if len(ps_[ps_ > -1]):
                ap = ps_[ps_ > -1].mean()
            else:
                ap = np.array(0)
            aps.append(ap)
        ps_curve.insert(0, np.zeros(ps_curve[0].shape))
        fig = plt.figure()
        ax = plt.subplot(111)
        for k in range(len(types)):
            ax.plot(rs, ps_curve[k + 1], color=[0, 0, 0], linewidth=0.5)
            ax.fill_between(
                rs,
                ps_curve[k],
                ps_curve[k + 1],
                color=COLOR_PALETTE[k],
                label=str(f"[{aps[k]:.3f}]" + types[k]),
            )
        plt.xlabel("recall")
        plt.ylabel("precision")
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.title(figure_title)
        plt.legend()
        # plt.show()
        export_path = str(Path(outDir) / f"{figure_title}.png")
        fig.savefig(export_path)
        plt.close(fig)

        export_path_list.append(export_path)
    return export_path_list


def _autolabel(ax, rects, is_percent=True):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if is_percent and height > 0 and height <= 1:  # for percent values
            text_label = "{:2.0f}".format(height * 100)
        else:
            text_label = "{:2.0f}".format(height)
        ax.annotate(
            text_label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize="x-small",
        )


def _makebarplot(_, ps, outDir, class_name, iou_type):
    areaNames = ["allarea", "small", "medium", "large"]
    types = ["C75", "C50", "Loc", "Sim", "Oth", "BG", "FN"]
    fig, ax = plt.subplots()
    x = np.arange(len(areaNames))  # the areaNames locations
    width = 0.60  # the width of the bars
    rects_list = []
    figure_title = iou_type + "-" + class_name + "-" + "ap bar plot"
    for k in range(len(types) - 1):
        type_ps = ps[k, ..., 0]
        # calculate ap
        aps = []
        for ps_ in type_ps.T:
            if len(ps_[ps_ > -1]):
                ap = ps_[ps_ > -1].mean()
            else:
                ap = np.array(0)
            aps.append(ap)
        # create bars
        rects_list.append(
            ax.bar(
                x - width / 2 + (k + 1) * width / len(types),
                aps,
                width / len(types),
                label=types[k],
                color=COLOR_PALETTE[k],
            )
        )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Mean Average Precision (mAP)")
    ax.set_title(figure_title)
    ax.set_xticks(x)
    ax.set_xticklabels(areaNames)
    ax.legend()

    # Add score texts over bars
    for rects in rects_list:
        _autolabel(ax, rects)

    # Save plot
    export_path = str(Path(outDir) / f"{figure_title}.png")
    fig.savefig(export_path)
    plt.close(fig)

    return export_path


def _get_gt_area_group_numbers(cocoEval):
    areaRng = cocoEval.params.areaRng
    areaRngStr = [str(aRng) for aRng in areaRng]
    areaRngLbl = cocoEval.params.areaRngLbl
    areaRngStr2areaRngLbl = dict(zip(areaRngStr, areaRngLbl))
    areaRngLbl2Number = dict.fromkeys(areaRngLbl, 0)
    for evalImg in cocoEval.evalImgs:
        if evalImg:
            for gtIgnore in evalImg["gtIgnore"]:
                if not gtIgnore:
                    aRngLbl = areaRngStr2areaRngLbl[str(evalImg["aRng"])]
                    areaRngLbl2Number[aRngLbl] += 1
    return areaRngLbl2Number


def _make_gt_area_group_numbers_plot(cocoEval, outDir, verbose=True):
    areaRngLbl2Number = _get_gt_area_group_numbers(cocoEval)
    areaRngLbl = areaRngLbl2Number.keys()
    if verbose:
        print("number of annotations per area group:", areaRngLbl2Number)

    # Init figure
    fig, ax = plt.subplots()
    x = np.arange(len(areaRngLbl))  # the areaNames locations
    width = 0.60  # the width of the bars
    figure_title = "number of annotations per area group"

    rects = ax.bar(x, list(areaRngLbl2Number.values()), width)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel("Number of annotations")
    ax.set_title(figure_title)
    ax.set_xticks(x)
    ax.set_xticklabels(areaRngLbl)

    # Add score texts over bars
    _autolabel(ax, rects, is_percent=False)

    # Save plot
    export_path = str(Path(outDir) / f"{figure_title}.png")
    fig.tight_layout()
    fig.savefig(export_path)
    plt.close(fig)

    return export_path


def _make_gt_area_histogram_plot(cocoEval, outDir):
    n_bins = 100
    areas = [ann["area"] for ann in cocoEval.cocoGt.anns.values()]

    # init figure
    figure_title = "gt annotation areas histogram plot"
    fig, ax = plt.subplots()

    # Set the number of bins
    ax.hist(np.sqrt(areas), bins=n_bins)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel("Squareroot Area")
    ax.set_ylabel("Number of annotations")
    ax.set_title(figure_title)

    # Save plot
    export_path = str(Path(outDir) / f"{figure_title}.png")
    fig.tight_layout()
    fig.savefig(export_path)
    plt.close(fig)

    return export_path


def _analyze_individual_category(k, cocoDt, cocoGt, catId, iou_type, areas=None, max_detections: int = 500):
    nm = cocoGt.loadCats(catId)[0]
    print(f"--------------analyzing {k + 1}-{nm['name']}---------------")
    ps_ = {}
    dt = copy.deepcopy(cocoDt)
    nm = cocoGt.loadCats(catId)[0]
    imgIds = cocoGt.getImgIds()
    dt_anns = dt.dataset["annotations"]
    select_dt_anns = []
    for ann in dt_anns:
        if ann["category_id"] == catId:
            select_dt_anns.append(ann)
    dt.dataset["annotations"] = select_dt_anns
    dt.createIndex()
    # compute precision but ignore superclass confusion
    gt = copy.deepcopy(cocoGt)
    child_catIds = gt.getCatIds(supNms=[nm["supercategory"]])
    for idx, ann in enumerate(gt.dataset["annotations"]):
        if ann["category_id"] in child_catIds and ann["category_id"] != catId:
            gt.dataset["annotations"][idx]["ignore"] = 1
            gt.dataset["annotations"][idx]["iscrowd"] = 1
            gt.dataset["annotations"][idx]["category_id"] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [max_detections]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [
            [0**2, areas[2]],
            [0**2, areas[0]],
            [areas[0], areas[1]],
            [areas[1], areas[2]],
        ]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_supercategory = cocoEval.eval["precision"][0, :, catId, :, :]
    ps_["ps_supercategory"] = ps_supercategory
    # compute precision but ignore any class confusion
    gt = copy.deepcopy(cocoGt)
    for idx, ann in enumerate(gt.dataset["annotations"]):
        if ann["category_id"] != catId:
            gt.dataset["annotations"][idx]["ignore"] = 1
            gt.dataset["annotations"][idx]["iscrowd"] = 1
            gt.dataset["annotations"][idx]["category_id"] = catId
    cocoEval = COCOeval(gt, copy.deepcopy(dt), iou_type)
    cocoEval.params.imgIds = imgIds
    cocoEval.params.maxDets = [max_detections]
    cocoEval.params.iouThrs = [0.1]
    cocoEval.params.useCats = 1
    if areas:
        cocoEval.params.areaRng = [
            [0**2, areas[2]],
            [0**2, areas[0]],
            [areas[0], areas[1]],
            [areas[1], areas[2]],
        ]
    cocoEval.evaluate()
    cocoEval.accumulate()
    ps_allcategory = cocoEval.eval["precision"][0, :, catId, :, :]
    ps_["ps_allcategory"] = ps_allcategory
    return k, ps_


def _analyse_results(
    res_file,
    ann_file,
    res_types,
    out_dir=None,
    extraplots=None,
    areas=None,
    max_detections=500,
):
    for res_type in res_types:
        if res_type not in ["bbox", "segm"]:
            raise ValueError(f"res_type {res_type} is not supported")
    if areas is not None:
        if len(areas) != 3:
            raise ValueError("3 integers should be specified as areas, representing 3 area regions")

    if out_dir is None:
        out_dir = Path(res_file).parent
        out_dir = str(out_dir / "coco_error_analysis")

    directory = os.path.dirname(out_dir + "/")
    if not os.path.exists(directory):
        print(f"-------------create {out_dir}-----------------")
        os.makedirs(directory)

    result_type_to_export_paths = {}

    # Load annotation file and add empty 'info' field if missing
    with open(ann_file) as f:
        ann_dict = json.load(f)
    if "info" not in ann_dict:
        ann_dict["info"] = {}

    # Create temporary file with updated annotations
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
        json.dump(ann_dict, tmp_file)
        temp_ann_file = tmp_file.name

    try:
        cocoGt = COCO(temp_ann_file)
        cocoDt = cocoGt.loadRes(res_file)
        imgIds = cocoGt.getImgIds()
        for res_type in res_types:
            res_out_dir = out_dir + "/" + res_type + "/"
            res_directory = os.path.dirname(res_out_dir)
            if not os.path.exists(res_directory):
                print(f"-------------create {res_out_dir}-----------------")
                os.makedirs(res_directory)
            iou_type = res_type
            cocoEval = COCOeval(copy.deepcopy(cocoGt), copy.deepcopy(cocoDt), iou_type)
            cocoEval.params.imgIds = imgIds
            cocoEval.params.iouThrs = [0.75, 0.5, 0.1]
            cocoEval.params.maxDets = [max_detections]
            if areas is not None:
                cocoEval.params.areaRng = [
                    [0**2, areas[2]],
                    [0**2, areas[0]],
                    [areas[0], areas[1]],
                    [areas[1], areas[2]],
                ]
            cocoEval.evaluate()
            cocoEval.accumulate()

            present_cat_ids = []
            catIds = cocoGt.getCatIds()
            for k, catId in enumerate(catIds):
                image_ids = cocoGt.getImgIds(catIds=[catId])
                if len(image_ids) != 0:
                    present_cat_ids.append(catId)
            matrix_shape = list(cocoEval.eval["precision"].shape)
            matrix_shape[2] = len(present_cat_ids)
            ps = np.zeros(matrix_shape)

            for k, catId in enumerate(present_cat_ids):
                ps[:, :, k, :, :] = cocoEval.eval["precision"][:, :, catId, :, :]
            ps = np.vstack([ps, np.zeros((4, *ps.shape[1:]))])

            recThrs = cocoEval.params.recThrs
            with Pool(processes=48) as pool:
                args = [
                    (k, cocoDt, cocoGt, catId, iou_type, areas, max_detections)
                    for k, catId in enumerate(present_cat_ids)
                ]
                analyze_results = pool.starmap(_analyze_individual_category, args)

            classname_to_export_path_list = {}
            for k, catId in enumerate(present_cat_ids):
                nm = cocoGt.loadCats(catId)[0]
                print(f"--------------saving {k + 1}-{nm['name']}---------------")
                analyze_result = analyze_results[k]
                if k != analyze_result[0]:
                    raise ValueError(f"k {k} != analyze_result[0] {analyze_result[0]}")
                ps_supercategory = analyze_result[1]["ps_supercategory"]
                ps_allcategory = analyze_result[1]["ps_allcategory"]
                # compute precision but ignore superclass confusion
                ps[3, :, k, :, :] = ps_supercategory
                # compute precision but ignore any class confusion
                ps[4, :, k, :, :] = ps_allcategory
                # fill in background and false negative errors and plot
                ps[5, :, k, :, :][ps[4, :, k, :, :] == -1] = -1
                ps[5, :, k, :, :][ps[4, :, k, :, :] > 0] = 1
                ps[6, :, k, :, :] = 1.0

                normalized_class_name = nm["name"].replace("/", "_").replace(os.sep, "_")

                curve_export_path_list = _makeplot(recThrs, ps[:, :, k], res_out_dir, normalized_class_name, iou_type)

                if extraplots:
                    bar_plot_path = _makebarplot(recThrs, ps[:, :, k], res_out_dir, normalized_class_name, iou_type)
                else:
                    bar_plot_path = None
                classname_to_export_path_list[nm["name"]] = {
                    "curves": curve_export_path_list,
                    "bar_plot": bar_plot_path,
                }

            curve_export_path_list = _makeplot(recThrs, ps, res_out_dir, "allclass", iou_type)
            if extraplots:
                bar_plot_path = _makebarplot(recThrs, ps, res_out_dir, "allclass", iou_type)
                gt_area_group_numbers_plot_path = _make_gt_area_group_numbers_plot(
                    cocoEval=cocoEval, outDir=res_out_dir, verbose=True
                )
                gt_area_histogram_plot_path = _make_gt_area_histogram_plot(cocoEval=cocoEval, outDir=res_out_dir)
            else:
                bar_plot_path, gt_area_group_numbers_plot_path, gt_area_histogram_plot_path = None, None, None

            result_type_to_export_paths[res_type] = {
                "classwise": classname_to_export_path_list,
                "overall": {
                    "bar_plot": bar_plot_path,
                    "curves": curve_export_path_list,
                    "gt_area_group_numbers": gt_area_group_numbers_plot_path,
                    "gt_area_histogram": gt_area_histogram_plot_path,
                },
            }
    finally:
        # Clean up temporary file
        os.unlink(temp_ann_file)

    print(f"COCO error analysis results are successfully exported to {out_dir}")

    return result_type_to_export_paths


def analyse(
    dataset_json_path: str,
    result_json_path: str,
    out_dir: Optional[str] = None,
    type: str = "bbox",
    no_extraplots: bool = False,
    areas: List[int] = [1024, 9216, 10000000000],
    max_detections: int = 500,
    return_dict: bool = False,
) -> Optional[dict]:
    """
    Args:
        dataset_json_path (str): file path for the coco dataset json file
        result_json_paths (str): file path for the coco result json file
        out_dir (str): dir to save analyse result images
        no_extraplots (bool): dont export export extra bar/stat plots
        type (str): 'bbox' or 'mask'
        areas (List[int]): area regions for coco evaluation calculations
        max_detections (int): Maximum number of detections to consider for AP alculation. Default: 500
        return_dict (bool): If True, returns a dict export paths.
    """
    if not has_matplotlib:
        logger.error("Please run 'uv pip install -U matplotlib' first for visualization.")
        raise ModuleNotFoundError("matplotlib not installed")
    if not has_pycocotools:
        logger.error("Please run 'uv pip install -U pycocotools' first for Coco analysis.")
        raise ModuleNotFoundError("pycocotools not installed")

    result = _analyse_results(
        result_json_path,
        dataset_json_path,
        res_types=[type],
        out_dir=out_dir,
        extraplots=not no_extraplots,
        areas=areas,
        max_detections=max_detections,
    )
    if return_dict:
        return result


if __name__ == "__main__":
    fire.Fire(analyse)
