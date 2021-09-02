import os

import fire

from sahi.slicing import slice_coco
from sahi.utils.file import Path, get_base_filename, increment_path, save_json


def main(
    image_dir: str,
    dataset_json_path: str,
    slice_size: int = 512,
    overlap_ratio: float = 0.2,
    ignore_negative_samples: bool = False,
    project: str = "runs/slice_coco",
    name: str = "exp",
):
    """
    Args:
        image_dir (str): directory for coco images
        dataset_json_path (str): file path for the coco dataset json file
        slice_size (int)
        overlap_ratio (float): slice overlap ratio
        ignore_negative_samples (bool): ignore images without annotation
        project (str): save results to project/name
        name (str): save results to project/name
    """

    # assure slice_size is list
    slice_size_list = slice_size
    if isinstance(slice_size_list, int):
        slice_size_list = [slice_size_list]

    # set output dir
    output_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run

    # slice coco dataset images and annotations
    print("Slicing step is starting...")
    for slice_size in slice_size_list:
        output_images_folder_name = get_base_filename(dataset_json_path)[1] + "_sliced_images_" + str(slice_size) + "/"
        output_images_dir = os.path.join(output_dir, output_images_folder_name)
        sliced_coco_name = get_base_filename(dataset_json_path)[0].replace(".json", "_sliced_" + str(slice_size))
        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=dataset_json_path,
            image_dir=image_dir,
            output_coco_annotation_file_name="",
            output_dir=output_images_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=slice_size,
            slice_width=slice_size,
            min_area_ratio=0.1,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            out_ext=".jpg",
            verbose=False,
        )
        output_coco_annotation_file_path = os.path.join(output_dir, sliced_coco_name + ".json")
        save_json(coco_dict, output_coco_annotation_file_path)
        print(
            f"Sliced 'slice_size: {slice_size}' coco file is saved to",
            output_coco_annotation_file_path,
        )


if __name__ == "__main__":
    fire.Fire(main)
