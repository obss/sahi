from __future__ import annotations

from sahi.utils.coco import Coco
from sahi.utils.file import Path, increment_path


def main(
    image_dir: str | None = None,
    dataset_json_path: str | None = None,
    train_split: int | float = 0.9,
    project: str = "runs/coco2yolo",
    name: str = "exp",
    seed: int = 1,
    disable_symlink=False,
):
    """Convert COCO dataset to YOLO format.

    Args:
        image_dir: Directory containing COCO images
        dataset_json_path: Path to the COCO JSON annotation file
        train_split: Training split ratio (0.0 to 1.0)
        project: Project directory for output
        name: Experiment name for output subdirectory
        seed: Random seed for reproducibility
        disable_symlink: Disable symbolic links (required in some environments)
    """

    # Validate required parameters
    if image_dir is None:
        raise ValueError("image_dir is required")
    if dataset_json_path is None:
        raise ValueError("dataset_json_path is required")

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
