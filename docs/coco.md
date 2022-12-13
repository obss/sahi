# COCO Utilities

<details closed>
<summary>
<big><b>COCO dataset creation:</b></big>
</summary>

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
- add predictions to coco image:

```python
coco_image.add_prediction(
  CocoPrediction(
    score=0.864434,
    bbox=[x_min, y_min, width, height],
    category_id=0,
    category_name='human'
  )
)
coco_image.add_prediction(
  CocoPrediction(
    score=0.653424,
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
- you can also export prediction array in coco prediction format and save it as json :

```python
from sahi.utils.file import save_json

predictions_array = coco.prediction_array
save_json = save_json(predictions_array, "coco_predictions.json")
```
- this prediction array can be used to get standard coco metrics for the predictions using official pycocotool api :

```python
# note:- pycocotools need to be installed seperately 
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

coco_ground_truth = COCO(annotation_file="coco_dataset.json")
coco_predictions = coco_ground_truth.loadRes("coco_predictions.json")

coco_evaluator = COCOeval(coco_ground_truth, coco_predictions, "bbox")
coco_evaluator.evaluate()
coco_evaluator.accumulate()
coco_evaluator.summarize()
```



</details>

<details closed>
<summary>
<big><b>Slice COCO dataset images and annotations into grids:</b></big>
</summary>

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
</details>

<details closed>
<summary>
<big><b>Split COCO dataset into train/val:</b></big>
</summary>

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

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
</details>

<details closed>
<summary>
<big><b>Filter/Update COCO dataset by categories:</b></big>
</summary>

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

# init Coco objects by specifying coco dataset paths and image folder directories
coco = Coco.from_coco_dict_or_path("coco.json")

# select only 3 categories; and map them to ids 1, 2 and 3
desired_name2id = {
  "big_vehicle": 1,
  "car": 2,
  "human": 3
}
coco.update_categories(desired_name2id)

# export updated/filtered COCO dataset
save_json(coco.json, "updated_coco.json")
```
</details>

<details closed>
<summary>
<big><b>Filter COCO dataset by annotation area:</b></big>
</summary>

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

# init Coco objects by specifying coco dataset paths and image folder directories
coco = Coco.from_coco_dict_or_path("coco.json")

# filter out images that contain annotations with smaller area than 50
area_filtered_coco = coco.get_area_filtered_coco(min=50)
# filter out images that contain annotations with smaller area than 50 and larger area than 10000
area_filtered_coco = coco.get_area_filtered_coco(min=50, max=10000)
# filter out images with seperate area intervals per category
intervals_per_category = {
  "human": {"min": 20, "max": 10000},
  "vehicle": {"min": 50, "max": 15000},
}
area_filtered_coco = coco.get_area_filtered_coco(intervals_per_category=intervals_per_category)

# export filtered COCO dataset
save_json(area_filtered_coco.json, "area_filtered_coco.json")
```
</details>

<details closed>
<summary>
<big><b>Filter out images that does not contain any annotation:</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# set ignore_negative_samples as False if you want images without annotations present in json and yolov5 exports
coco = Coco.from_coco_dict_or_path("coco.json", ignore_negative_samples=False)

```
</details>

<details closed>
<summary>
<big><b>Merge COCO dataset files:</b></big>
</summary>

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

# init Coco objects by specifying coco dataset paths and image folder directories
coco_1 = Coco.from_coco_dict_or_path("coco1.json", image_dir="images_1/")
coco_2 = Coco.from_coco_dict_or_path("coco2.json", image_dir="images_2/")

# merge Coco datasets
coco_1.merge(coco_2)

# export merged COCO dataset
save_json(coco_1.json, "merged_coco.json")
```
</details>

<details closed>
<summary>
<big><b>Convert COCO dataset to ultralytics/yolov5 format:</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# init Coco object
coco = Coco.from_coco_dict_or_path("coco.json", image_dir="coco_images/")

# export converted YoloV5 formatted dataset into given output_dir with a 85% train/15% val split
coco.export_as_yolov5(
  output_dir="output/folder/dir",
  train_split_rate=0.85
)

```
</details>

<details closed>
<summary>
<big><b>Convert train/val COCO dataset to ultralytics/yolov5 format:</b></big>
</summary>

```python
from sahi.utils.coco import Coco, export_coco_as_yolov5

# init Coco object
train_coco = Coco.from_coco_dict_or_path("train_coco.json", image_dir="coco_images/")
val_coco = Coco.from_coco_dict_or_path("val_coco.json", image_dir="coco_images/")

# export converted YoloV5 formatted dataset into given output_dir with given train/val split
data_yml_path = export_coco_as_yolov5(
  output_dir="output/folder/dir",
  train_coco=train_coco,
  val_coco=val_coco
)

```
</details>

<details closed>
<summary>
<big><b>Subsample COCO dataset file:</b></big>
</summary>

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

# bonus: create a Coco object with 1/10 of total images that contain first category
subsampled_coco = coco.get_subsampled_coco(subsample_ratio=10, category_id=0)

# bonus2: create a Coco object with negative samples reduced to 1/10
subsampled_coco = coco.get_subsampled_coco(subsample_ratio=10, category_id=-1)
```
</details>

<details closed>
<summary>
<big><b>Upsample COCO dataset file:</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# specify coco dataset path
coco_path = "coco.json"

# init Coco object
coco = Coco.from_coco_dict_or_path(coco_path)

# create a Coco object with each sample is repeated 10 times
upsampled_coco = coco.get_upsampled_coco(upsample_ratio=10)

# export upsampled COCO dataset
save_json(upsampled_coco.json, "upsampled_coco.json")

# bonus: create a Coco object with images that contain first category repeated 10 times
subsampled_coco = coco.get_subsampled_coco(upsample_ratio=10, category_id=0)

# bonus2: create a Coco object with negative samples upsampled by 10 times
upsampled_coco = coco.get_upsampled_coco(upsample_ratio=10, category_id=-1)
```
</details>

<details closed>
<summary>
<big><b>Get dataset stats:</b></big>
</summary>

```python
from sahi.utils.coco import Coco

# init Coco object
coco = Coco.from_coco_dict_or_path("coco.json")

# get dataset stats
coco.stats
{
  'num_images': 6471,
  'num_annotations': 343204,
  'num_categories': 2,
  'num_negative_images': 0,
  'num_images_per_category': {'human': 5684, 'vehicle': 6323},
  'num_annotations_per_category': {'human': 106396, 'vehicle': 236808},
  'min_num_annotations_in_image': 1,
  'max_num_annotations_in_image': 902,
  'avg_num_annotations_in_image': 53.037243084530985,
  'min_annotation_area': 3,
  'max_annotation_area': 328640,
  'avg_annotation_area': 2448.405738278109,
  'min_annotation_area_per_category': {'human': 3, 'vehicle': 3},
  'max_annotation_area_per_category': {'human': 72670, 'vehicle': 328640},
}

```
</details>

<details closed>
<summary>
<big><b>Remove invalid coco results:</b></big>
</summary>

```python
from sahi.utils.file import save_json
from sahi.utils.coco import remove_invalid_coco_results

# remove invalid predictions from COCO results JSON
coco_results = remove_invalid_coco_results("coco_result.json")

# export processed COCO results
save_json(coco_results, "fixed_coco_result.json")

# bonus: remove invalid predictions from COCO results JSON by giving COCO
# dataset path to also filter out bbox results exceeding image height&width
coco_results = remove_invalid_coco_results("coco_result.json", "coco_dataset.json")
```
</details>

<details closed>
<summary>
<big><b>Get COCO with clipped bounding boxes:</b></big>
</summary>

- import required classes:

```python
from sahi.utils.coco import Coco
from sahi.utils.file import save_json
```
Usage:

```python
# Clip overflowing bounding boxes to image width & height
coco = Coco.from_coco_dict_or_path(coco_path, clip_bboxes_to_img_dims=True)
```
or,

```python
# apply to your already created coco object
coco = coco.get_coco_with_clipped_bboxes()
```
- Export your clipped_bboxed_coco:
```python
save_json(coco.json, "coco.json")
```
</details>
