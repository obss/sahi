import os
import argparse

from sahi.slicing import slice_coco
from sahi.utils.coco import split_coco_as_train_val, Coco
from sahi.utils.file import get_base_filename, save_json, Path, increment_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "coco_json_path", type=str, default=None, help="path to coco annotation json file",
    )
    parser.add_argument("coco_image_dir", type=str, default="", help="folder containing coco images")
    parser.add_argument("--slice_size", type=int, nargs="+", default=[512], help="slice size")
    parser.add_argument("--overlap_ratio", type=float, default=0.2, help="slice overlap ratio")
    parser.add_argument("--ignore_negative_samples", action="store_true", help="ignore images without annotation")
    parser.add_argument("--project", default="runs/slice_coco", help="save results to project/name")
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument("--seed", type=int, default=1, help="fix the seed for reproducibility")

    args = parser.parse_args()

    # assure slice_size is list
    slice_size_list = args.slice_size
    if isinstance(slice_size_list, int):
        slice_size_list = [slice_size_list]

    # set output dir
    output_dir = Path(increment_path(Path(args.project) / args.name, exist_ok=False))  # increment run

    # slice coco dataset images and annotations
    print("Slicing step is starting...")
    for slice_size in slice_size_list:
        output_images_folder_name = (
            get_base_filename(args.coco_json_path)[1] + "_sliced_images_" + str(slice_size) + "/"
        )
        output_images_dir = os.path.join(output_dir, output_images_folder_name)
        sliced_coco_name = get_base_filename(args.coco_json_path)[0].replace(".json", "_sliced_" + str(slice_size))
        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=args.coco_json_path,
            image_dir=args.coco_image_dir,
            output_coco_annotation_file_name="",
            output_dir=output_images_dir,
            ignore_negative_samples=args.ignore_negative_samples,
            slice_height=slice_size,
            slice_width=slice_size,
            min_area_ratio=0.1,
            overlap_height_ratio=args.overlap_ratio,
            overlap_width_ratio=args.overlap_ratio,
            out_ext=".jpg",
            verbose=False,
        )
        output_coco_annotation_file_path = os.path.join(output_dir, sliced_coco_name + ".json")
        save_json(coco_dict, output_coco_annotation_file_path)
        print(
            f"Sliced 'slice_size: {slice_size}' coco file is saved to", output_coco_annotation_file_path,
        )
