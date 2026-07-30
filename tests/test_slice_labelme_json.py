"""Tests for slicing Labelme JSON annotations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from sahi.slicing import slice_labelme_json
from sahi.utils.file import save_json


def test_slice_labelme_json(tmp_path: Path) -> None:
    """Slices one image + Labelme json and exports image/json patches."""
    image_array = np.full((100, 100, 3), 255, dtype=np.uint8)
    image_path = tmp_path / "sample.png"
    Image.fromarray(image_array).save(image_path)

    labelme_dict = {
        "version": "5.4.1",
        "flags": {},
        "shapes": [
            {
                "label": "scratch",
                "points": [[10, 10], [60, 10], [60, 60], [10, 60]],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            },
            {
                "label": "dent",
                "points": [[70, 70], [90, 70], [90, 90], [70, 90]],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            },
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 100,
    }
    annotation_path = tmp_path / "sample.json"
    save_json(labelme_dict, str(annotation_path))

    output_dir = tmp_path / "sliced"
    slice_image_result, sliced_labelme_dicts, saved_json_paths = slice_labelme_json(
        image=str(image_path),
        input_json_path=str(annotation_path),
        output_dir=str(output_dir),
        slice_height=50,
        slice_width=50,
        overlap_height_ratio=0.0,
        overlap_width_ratio=0.0,
        auto_slice_resolution=False,
        min_area_ratio=0.0,
    )

    assert len(slice_image_result) == 4
    assert len(sliced_labelme_dicts) == 4
    assert len(saved_json_paths) == 4

    saved_files = list(output_dir.glob("*"))
    assert len(saved_files) == 8
    assert len(list(output_dir.glob("*.png"))) == 4
    assert len(list(output_dir.glob("*.json"))) == 4

    first_slice = next(item for item in sliced_labelme_dicts if item["imagePath"].startswith("sample_0_0_50_50"))
    assert first_slice["imageHeight"] == 50
    assert first_slice["imageWidth"] == 50
    assert len(first_slice["shapes"]) == 1
    assert first_slice["shapes"][0]["label"] == "scratch"
    assert all(0 <= point[0] <= 50 and 0 <= point[1] <= 50 for point in first_slice["shapes"][0]["points"])

    bottom_right_slice = next(
        item for item in sliced_labelme_dicts if item["imagePath"].startswith("sample_50_50_100_100")
    )
    labels = sorted(shape["label"] for shape in bottom_right_slice["shapes"])
    assert labels == ["dent", "scratch"]


def test_slice_labelme_json_min_width_height_filter(tmp_path: Path) -> None:
    """Keeps thin fragments and drops tiny fragments when min_width_height is set."""
    image_array = np.full((100, 100, 3), 255, dtype=np.uint8)
    image_path = tmp_path / "filter_sample.png"
    Image.fromarray(image_array).save(image_path)

    labelme_dict = {
        "version": "5.4.1",
        "flags": {},
        "shapes": [
            {
                "label": "tall",
                "points": [[40, 5], [95, 5], [95, 45], [40, 45]],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            },
            {
                "label": "tiny",
                "points": [[49, 49], [99, 49], [99, 99], [49, 99]],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {},
            },
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 100,
    }
    annotation_path = tmp_path / "filter_sample.json"
    save_json(labelme_dict, str(annotation_path))

    _, default_filtered_dicts, _ = slice_labelme_json(
        image=str(image_path),
        input_json_path=str(annotation_path),
        output_dir=None,
        slice_height=50,
        slice_width=50,
        overlap_height_ratio=0.0,
        overlap_width_ratio=0.0,
        auto_slice_resolution=False,
        min_area_ratio=0.2,
    )
    top_left_default = next(
        item for item in default_filtered_dicts if item["imagePath"].startswith("filter_sample_0_0_50_50")
    )
    assert top_left_default["shapes"] == []

    _, custom_filtered_dicts, _ = slice_labelme_json(
        image=str(image_path),
        input_json_path=str(annotation_path),
        output_dir=None,
        slice_height=50,
        slice_width=50,
        overlap_height_ratio=0.0,
        overlap_width_ratio=0.0,
        auto_slice_resolution=False,
        min_area_ratio=0.2,
        min_width_height=15,
    )
    top_left_custom = next(
        item for item in custom_filtered_dicts if item["imagePath"].startswith("filter_sample_0_0_50_50")
    )
    kept_labels = [shape["label"] for shape in top_left_custom["shapes"]]
    assert kept_labels == ["tall"]

def test_my():
    ret = slice_labelme_json(
        image=r"E:\Pictures\scratches\1.jpg",
        input_json_path=r"E:\Pictures\scratches\1.json",
        output_dir="E:\Pictures\scratches\sliced1",
        slice_height=512,
        slice_width=512,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        auto_slice_resolution=False,
        min_area_ratio=0.01,
        min_width_height=15,
    )
    assert len(ret)>0

if __name__ == "__main__":
    test_my()