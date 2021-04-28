# COCO Utilities

## COCO dataset creation steps:

- import required classes:

```python
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
```

- init Coco object:

```python
coco = Coco()
```

- add categories starting from id 0:

```python
coco.add_category(CocoCategory(id=0, name='human'))
coco.add_category(CocoCategory(id=1, name='vehicle'))
```

- create a coco image:

```python
coco_image = CocoImage(file_name="image1.jpg", height=1080, width=1920)
```

- add annotations to coco image:

```python
coco_image.add_annotation(
  CocoAnnotation(
    bbox=[x_min, y_min, width, height],
    category_id=0,
    category_name='human'
  )
)
coco_image.add_annotation(
  CocoAnnotation(
    bbox=[x_min, y_min, width, height],
    category_id=1,
    category_name='vehicle'
  )
)
```

- add coco image to Coco object:

```python
coco.add_image(coco_image)
```

- after adding all images, convert coco object to coco json:

```python
coco_json = coco.json
```

- you can export it as json file:

```python
from sahi.utils.file import save_json

save_json(coco_json, "coco_dataset.json")
```

## Slice COCO dataset images and annotations into grids:

```python
from sahi.slicing import slice_coco

coco_dict, coco_path = slice_coco(
    coco_annotation_file_path="coco.json",
    image_dir="source/coco/image/dir",
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

## Split COCO dataset into train/val:

```python
from sahi.utils.coco import Coco

# specify coco dataset path
coco_path = "coco.json"

# init Coco object
coco = Coco.from_coco_dict_or_path(coco_path)

# split COCO dataset with a 85% train/15% val split
result = coco.split_coco_as_train_val(
  train_split_rate=0.85
)

# export train val split files
save_json(result["train_coco"].json, "train_split.json")
save_json(result["val_coco"].json, "val_split.json")
```

## Combine COCO dataset files:

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

# specify coco dataset json paths
coco_path_1 = "coco1.json"
coco_path_2 = "coco2.json"
coco_path_3 = "coco3.json"

# init Coco object with a list of Coco path
combined_coco = Coco.from_coco_dict_or_path([coco_path_1, coco_path_2, coco_path_3])

# export combined COCO dataset
save_json(combined_coco.json, "combined_coco.json")
```

## Convert COCO dataset to ultralytics/yolov5 format:

```python
from sahi.utils.coco import Coco

# specify coco dataset path
coco_path = "coco.json"

# init Coco object
coco = Coco.from_coco_dict_or_path(coco_path)

# export converted YoloV5 formatted dataset into given output_dir with a 85% train/15% val split
coco.export_as_yolov5(
  image_dir="source/coco/image/dir",
  output_dir="output/folder/dir",
  train_split_rate=0.85
)
```

## Subsample COCO dataset file:

```python
from sahi.utils.coco import Coco

# specify coco dataset path
coco_path = "coco.json"

# init Coco object
coco = Coco.from_coco_dict_or_path(coco_path)

# create a Coco object with 1/10 of total images
subsampled_coco = coco.get_subsampled_coco(subsample_ratio=10)

# export subsampled COCO dataset
save_json(subsampled_coco.json, "subsampled_coco.json")
```
