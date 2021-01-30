import os

from sahi.slicing import slice_coco
from sahi.utils.coco import split_coco_as_train_val
from sahi.utils.file import get_base_filename, save_json

if __name__ == "__main__":
    coco_annotation_file_path = ""
    output_coco_annotation_directory = ""
    image_dir = ""
    train_split_rate = 0.9
    ignore_negative_samples = True
    slice_size = [512]

    # split coco file into train-val coco files
    print("Splitting step is starting...")
    coco_dict_paths = split_coco_as_train_val(
        coco_file_path=coco_annotation_file_path,
        train_split_rate=train_split_rate,
    )

    # slice train-val coco dataset images
    print("Slicing step is starting...")
    sliced_output_dir = coco_annotation_file_path.replace(
        "coco.json", "sliced_input_" + str(slice_size) + "/"
    )
    for split_type in coco_dict_paths.keys():
        coco_annotation_file_path = coco_dict_paths[split_type]
        sliced_coco_name = get_base_filename(coco_annotation_file_path)[0].replace(
            ".json", "_sliced_" + str(slice_size)
        )
        coco_dict, coco_path = slice_coco(
            coco_annotation_file_path=coco_annotation_file_path,
            image_dir=image_dir,
            output_coco_annotation_file_name="",
            output_dir=sliced_output_dir,
            ignore_negative_samples=ignore_negative_samples,
            slice_height=slice_size,
            slice_width=slice_size,
            max_allowed_zeros_ratio=0.2,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
            slice_sep="_",
            out_ext=".png",
            verbose=True,
        )
        output_sliced_coco_annotation_file_path = os.path.join(
            output_coco_annotation_directory, sliced_coco_name + ".json"
        )
        save_json(coco_dict, coco_path)
        print(
            "Sliced",
            split_type,
            "coco file is saved to",
            output_sliced_coco_annotation_file_path,
        )
