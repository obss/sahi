from typing import Union

import fire

from sahi.utils.coco import Coco
from sahi.utils.file import Path, increment_path


def main(
    image_dir: str,
    dataset_json_path: str,
    train_split: Union[int, float] = 0.9,
    project: str = "runs/coco2yolo",
    name: str = "exp",
    seed: int = 1,
    disable_symlink=False,
):
    """
    Args:
        images_dir (str): directory for coco images
        dataset_json_path (str): file path for the coco json file to be converted
        train_split (float or int): set the training split ratio
        project (str): save results to project/name
        name (str): save results to project/name"
        seed (int): fix the seed for reproducibility
        disable_symlink (bool): required in google colab env
    """

    # increment run
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))
    # load coco dict
    coco = Coco.from_coco_dict_or_path(
        coco_dict_or_path=dataset_json_path,
        image_dir=image_dir,
    )
    # export as YOLO
    coco.export_as_yolo(
        output_dir=str(save_dir),
        train_split_rate=train_split,
        numpy_seed=seed,
        disable_symlink=disable_symlink,
    )

    print(f"COCO to YOLO conversion results are successfully exported to {save_dir}")


if __name__ == "__main__":
    fire.Fire(main)
