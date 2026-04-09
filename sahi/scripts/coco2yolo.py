"""Convert COCO dataset annotations to YOLO format."""

from __future__ import annotations

import fire

from sahi.utils.coco import Coco
from sahi.utils.file import Path, increment_path


def main(
    image_dir: str,
    dataset_json_path: str,
    train_split: int | float = 0.9,
    project: str = "runs/coco2yolo",
    name: str = "exp",
    seed: int = 1,
    disable_symlink: bool = False,
) -> None:
    """Convert COCO dataset annotations to YOLO format.

    Args:
        image_dir: Directory containing COCO images.
        dataset_json_path: Path to the COCO JSON file to be converted.
        train_split: Training/validation split ratio.
        project: Project directory for results.
        name: Experiment name within project.
        seed: Random seed for reproducibility.
        disable_symlink: Disable symlinks (needed for Google Colab).
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
