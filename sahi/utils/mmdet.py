import urllib.request
from os import path
from pathlib import Path
from typing import Optional
from importlib import import_module
import shutil
import sys


def mmdet_version_as_integer():
    import mmdet

    return int(mmdet.__version__.replace(".", ""))


class MmdetTestConstants:
    try:
        MMDET_CASCADEMASKRCNN_MODEL_URL = "http://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
        MMDET_CASCADEMASKRCNN_MODEL_PATH = (
            "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_20200203-9d4dcb24.pth"
        )
        MMDET_RETINANET_MODEL_URL = "http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"
        MMDET_RETINANET_MODEL_PATH = "tests/data/models/mmdet_retinanet/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth"

        if mmdet_version_as_integer() < 290:
            MMDET_CASCADEMASKRCNN_CONFIG_PATH = (
                "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco_v280.py"
            )
            MMDET_RETINANET_CONFIG_PATH = "tests/data/models/mmdet_retinanet/retinanet_r50_fpn_1x_coco_v280.py"
        else:
            MMDET_CASCADEMASKRCNN_CONFIG_PATH = (
                "tests/data/models/mmdet_cascade_mask_rcnn/cascade_mask_rcnn_r50_fpn_1x_coco.py"
            )
            MMDET_RETINANET_CONFIG_PATH = "tests/data/models/mmdet_retinanet/retinanet_r50_fpn_1x_coco.py"
    except ImportError:
        print("warning: mmdet installation not found, omitting MmdetTestConstants")


def download_mmdet_cascade_mask_rcnn_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            MmdetTestConstants.MMDET_CASCADEMASKRCNN_MODEL_URL,
            destination_path,
        )


def download_mmdet_retinanet_model(destination_path: Optional[str] = None):

    if destination_path is None:
        destination_path = MmdetTestConstants.MMDET_RETINANET_MODEL_PATH

    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)

    if not path.exists(destination_path):
        urllib.request.urlretrieve(
            MmdetTestConstants.MMDET_RETINANET_MODEL_URL,
            destination_path,
        )


def download_mmdet_config(
    model_name: str = "cascade_rcnn",
    config_file_name: str = "cascade_mask_rcnn_r50_fpn_1x_coco.py",
    verbose: bool = True,
) -> str:
    """
    Merges config files starting from given main config file name. Saves as single file.

    Args:
        model_name (str): mmdet model name. check https://github.com/open-mmlab/mmdetection/tree/master/configs.
        config_file_name (str): mdmet config file name.
        verbose (bool): if True, print save path.

    Returns:
        (str) abs path of the downloaded config file.
    """

    # get mmdet version
    from mmdet import __version__

    mmdet_ver = "v" + __version__

    # set main config url
    base_config_url = (
        "https://raw.githubusercontent.com/open-mmlab/mmdetection/" + mmdet_ver + "/configs/" + model_name + "/"
    )
    main_config_url = base_config_url + config_file_name

    # set config dirs
    temp_configs_dir = Path("temp_mmdet_configs")
    main_config_dir = temp_configs_dir / model_name

    # create config dirs
    temp_configs_dir.mkdir(parents=True, exist_ok=True)
    main_config_dir.mkdir(parents=True, exist_ok=True)

    # get main config file name
    filename = Path(main_config_url).name

    # set main config file path
    main_config_path = str(main_config_dir / filename)

    # download main config file
    urllib.request.urlretrieve(
        main_config_url,
        main_config_path,
    )

    # read main config file
    sys.path.insert(0, str(main_config_dir))
    temp_module_name = path.splitext(filename)[0]
    mod = import_module(temp_module_name)
    sys.path.pop(0)
    config_dict = {name: value for name, value in mod.__dict__.items() if not name.startswith("__")}

    # iterate over secondary config files
    for secondary_config_file_path in config_dict["_base_"]:
        # set config url
        config_url = base_config_url + secondary_config_file_path
        config_path = main_config_dir / secondary_config_file_path

        # create secondary config dir
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # download secondary config files
        urllib.request.urlretrieve(
            config_url,
            str(config_path),
        )

    # set final config dirs
    configs_dir = Path("mmdet_configs") / mmdet_ver
    model_config_dir = configs_dir / model_name

    # create final config dir
    configs_dir.mkdir(parents=True, exist_ok=True)
    model_config_dir.mkdir(parents=True, exist_ok=True)

    # get final config file name
    filename = Path(main_config_url).name

    # set final config file path
    final_config_path = str(model_config_dir / filename)

    # dump final config as single file
    from mmcv import Config

    config = Config.fromfile(main_config_path)
    config.dump(final_config_path)

    if verbose:
        print(f"mmdet config file has been downloaded to {path.abspath(final_config_path)}")

    # remove temp config dir
    shutil.rmtree(temp_configs_dir)

    return path.abspath(final_config_path)


if __name__ == "__main__":
    download_mmdet_config(
        model_name="cascade_rcnn",
        config_file_name="cascade_mask_rcnn_r50_fpn_1x_coco.py",
        verbose=False,
    )
