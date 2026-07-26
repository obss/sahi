---
tags:
  - inference
  - slicing
  - batch-inference
  - visualization
  - object-detection
  - small-object-detection
---

# Prediction Araçları

## Sliced Inference

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init any model
detection_model = AutoDetectionModel.from_pretrained(model_type='mmdet',...) # for MMDetection models
detection_model = AutoDetectionModel.from_pretrained(model_type='ultralytics',...) # for YOLOv8/YOLO11/YOLO26 models
detection_model = AutoDetectionModel.from_pretrained(model_type='huggingface',...) # for HuggingFace detection models
detection_model = AutoDetectionModel.from_pretrained(model_type='torchvision',...) # for Torchvision detection models
detection_model = AutoDetectionModel.from_pretrained(model_type='rtdetr',...) # for RT-DETR models
detection_model = AutoDetectionModel.from_pretrained(model_type='yoloe',...) # for YOLOE models
detection_model = AutoDetectionModel.from_pretrained(model_type='yolov5',...) # for YOLOv5 models
detection_model = AutoDetectionModel.from_pretrained(model_type='yolo-world',...) # for YOLOWorld models
detection_model = AutoDetectionModel.from_pretrained(model_type='roboflow',...) # for Roboflow RFDETR detection/segmentation models

# get sliced prediction result
result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

```

## Standart Inference

```python
from sahi.predict import get_prediction
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get standard prediction result
result = get_prediction(
    image,
    detection_model,
)

```

## Batch Inference

### Klasör veya dosya listesi üzerinde batch prediction

Birden fazla görsel üzerinde aynı anda Sliced Inference çalıştırmak ve sonuçları otomatik dışa aktarmak için yüksek seviyeli `predict` fonksiyonunu kullanın:

```python
from sahi.predict import predict
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get batch predict result
result = predict(
    model_type=..., # one of 'ultralytics', 'mmdet', 'huggingface'
    model_path=..., # path to model weight file
    model_config_path=..., # for mmdet models
    model_confidence_threshold=0.5,
    model_device='cpu', # or 'cuda:0'
    source=..., # image or folder path
    no_standard_prediction=True,
    no_sliced_prediction=False,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    export_pickle=False,
    export_crop=False,
    progress_bar=False,
)
```

### Düşük seviyeli batch inference API'si

`perform_batch_inference`, tek bir çağrıda bir modeli birden fazla görsel üzerinde çalıştırmanıza ve görsel başına prediction listelerini almanıza olanak tanır. Ultralytics YOLO modelleri yerel GPU batching kullanır; diğer tüm modeller aynı API ile sıralı tekil görsel inference işlemini yürütür.

```python
import cv2
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(
    model_type="ultralytics",
    model_path="yolo26n.pt",
    confidence_threshold=0.25,
    device="cuda:0",
)

# Load a batch of images as numpy arrays (H, W, C) in RGB
images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in image_paths]

# Run batch inference (native GPU batching for Ultralytics)
detection_model.perform_batch_inference(images)

# Provide per-image shift amounts and full image sizes (use [[0, 0]] defaults
# when images are not slices)
shift_amount_list = [[0, 0]] * len(images)
full_shape_list   = [[img.shape[0], img.shape[1]] for img in images]

detection_model.convert_original_predictions(
    shift_amount=shift_amount_list,
    full_shape=full_shape_list,
)

# Access predictions per image
for i, preds in enumerate(detection_model.object_prediction_list_per_image):
    print(f"Image {i}: {len(preds)} detections")
    for pred in preds:
        print(pred.category.name, pred.score.value, pred.bbox.to_xyxy())
```

!!! note "Tekil görsel uyumluluğu"
    Mevcut `object_prediction_list` özelliği değişmemiştir ve ilk görsel için tahminleri döndürür; böylece `perform_inference` + `convert_original_predictions` + `object_prediction_list` kullanan kodlar herhangi bir değişiklik gerektirmeden çalışmaya devam eder.

## İlerleme Çubuğu (Progress-Bar)

Çok sayıda dilim üzerinde Sliced Inference çalıştırırken ilerleme güncellemelerini kontrol etmek ve almak için iki seçenek eklenmiştir:

- `progress_bar` (bool): True olduğunda, dilim işleme sırasında bir tqdm ilerleme çubuğu gösterir. Terminal ve notebook'larda görsel geri bildirim için kullanışlıdır. Varsayılan False'tur.
- `progress_callback` (callable): Her dilim (veya dilim grubu) işlendikten sonra çağrılacak bir geri çağırma (callback) fonksiyonu. Callback iki tamsayı argüman alır: `(current_slice_index, total_slices)`. Bunu özel ilerleme raporlamasını entegre etmek için kullanın (örneğin bir GUI öğesini güncellemek veya bir dosyaya ilerleme kaydetmek).

Callback kullanımına örnek:

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init model
detection_model = AutoDetectionModel.from_pretrained(...)

def my_progress_callback(current, total):
    print(f"Processed {current}/{total} slices")

result = get_sliced_prediction(
    image,
    detection_model,
    slice_height=512,
    slice_width=512,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
    progress_bar=False,           # disable tqdm bar
    progress_callback=my_progress_callback,  # use callback to receive updates
)
```

!!! tip "Notlar"
    - `progress_bar` ve `progress_callback` birlikte kullanılabilir. İkisi de sağlandığında, tqdm çubuğu görüntülenir ve her dilim grubu işlendikten sonra callback çağrılır.
    - `progress_callback` 1 tabanlı indekslerle çağrılır (yani ilk çağrı `(1, total)` olacaktır).

## Inference Sırasında Özel Sınıfları Hariç Tutma

```python
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# define the class names to exclude from custom model inference
exclude_classes_by_name = ["car"]

# or exclude classes by its custom id
exclude_classes_by_id = [0]

result = get_sliced_prediction(
    image,
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2,
    exclude_classes_by_name = exclude_classes_by_name
    # exclude_classes_by_id = exclude_classes_by_id
)

```

## Görselleştirme Parametreleri ve Format Dışa Aktarımı

```python
from sahi.predict import get_prediction
from sahi import AutoDetectionModel
from PIL import Image

# init a model
detection_model = AutoDetectionModel.from_pretrained(...)

# get prediction result
result = get_prediction(
    image,
    detection_model,
)

# Export with custom visualization parameters
result.export_visuals(
    export_dir="outputs/",
    text_size=1.0,  # Size of the class label text
    rect_th=2,      # Thickness of bounding box lines
    text_th=2,      # Thickness of the text
    hide_labels=False,  # Set True to hide class labels
    hide_conf=False,    # Set True to hide confidence scores
    color=(255, 0, 0),  # Custom color in RGB format (red in this example)
    file_name="custom_visualization",
    export_format="jpg"  # Supports 'jpg' and 'png'
)

# Export as COCO format annotations
coco_annotations = result.to_coco_annotations()
# Example output: [{'image_id': None, 'bbox': [x, y, width, height], 'category_id': 0, 'area': width*height, ...}]

# Export as COCO predictions (includes confidence scores)
coco_predictions = result.to_coco_predictions(image_id=1)
# Example output: [{'image_id': 1, 'bbox': [x, y, width, height], 'score': 0.98, 'category_id': 0, ...}]

# Export as imantics format
imantics_annotations = result.to_imantics_annotations()
# For use with imantics library: https://github.com/jsbroks/imantics

# Export for FiftyOne visualization
fiftyone_detections = result.to_fiftyone_detections()
# For use with FiftyOne: https://github.com/voxel51/fiftyone
```

!!! tip "Etkileşimli İncelemeler"
    Bu tahmin araçlarını aksiyonda görmek ister misiniz? Desteklenen tüm framework'ler için uygulamalı örneklerin bulunduğu [etkileşimli notebook'larımıza](notebooks.md) göz atın.
