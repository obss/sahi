---
tags:
  - cli
  - inference
  - coco
  - slicing
  - evaluation
  - fiftyone
---

# CLI Komutları

SAHI, object detection görevleri için kapsamlı bir komut satırı arayüzü (CLI) sunar. Bu kılavuz, tüm kullanılabilir komutları ayrıntılı örnekler ve seçeneklerle kapsar.

## `predict` command usage

Daha iyi küçük nesne tespiti (small object detection) amacıyla Sliced Inference kullanarak görseller veya videolar üzerinde object detection inference gerçekleştirin.

### Temel Kullanım

```bash
sahi predict --source image/file/or/folder --model_path path/to/model --model_config_path path/to/config
```

Bu işlem varsayılan parametrelerle Sliced Inference gerçekleştirir ve prediction görsellerini `runs/predict/exp` klasörüne aktarır.

### Video Girdisi Desteği

SAHI, aynı komut yapısıyla video inference desteği sunar:

```bash
sahi predict --model_path yolo26s.pt --model_type ultralytics --source video.mp4
```

#### Gerçek Zamanlı Video Görselleştirmesi

`--view_video` bayrağı ile inference sırasında video oluşturmayı izleyin:

```bash
sahi predict --model_path yolo26s.pt --model_type ultralytics --source video.mp4 --view_video
```

**Klavye Kontrolleri:**

- **`D`** - 100 kare ileri
- **`A`** - 100 kare geri
- **`G`** - 20 kare ileri
- **`F`** - 20 kare geri
- **`Esc`** - İzleyiciden çıkış

> **İpucu:** `--view_video` yavaşsa, 20 karelik aralıkları atlamak için `--frame_skip_interval=20` ekleyin.

### Gelişmiş Dilimleme Parametreleri

Optimum tespit için dilimleme davranışını özelleştirin:

```bash
sahi predict --slice_width 512 --slice_height 512 \
  --overlap_height_ratio 0.1 --overlap_width_ratio 0.1 \
  --model_confidence_threshold 0.25 \
  --source image/file/or/folder \
  --model_path path/to/model \
  --model_config_path path/to/config
```

#### Model Konfigürasyonu

**Tespit Framework'ü:**

- `--model_type mmdet` - MMDetection modelleri için
- `--model_type ultralytics` - Ultralytics/YOLOv5/YOLO11/YOLO26 modelleri için
- `--model_type huggingface` - HuggingFace modelleri için
- `--model_type torchvision` - Torchvision modelleri için

**Confidence Threshold (Güven Eşiği):**

- `--model_confidence_threshold 0.25` - Tespitler için minimum güven skorunu belirler

#### Postprocessing Seçenekleri

**Postprocess Tipi:**

- `--postprocess_type GREEDYNMM` - Greedy non-maximum merging (varsayılan)
- `--postprocess_type NMS` - Standart non-maximum suppression

**Eşleşme Metrikleri:**

- `--postprocess_match_metric IOS` - Intersection over smaller area
- `--postprocess_match_metric IOU` - Intersection over union (varsayılan)

**Ek Seçenekler:**

- `--postprocess_match_threshold 0.5` - Eşleşme eşiğini belirler
- `--postprocess_class_agnostic` - Postprocessing sırasında kategori ID'lerini göz ardı eder

#### Dışa Aktarma Seçenekleri

**Görsel Dışa Aktarımlar:**

- `--novisual` - Prediction görselleştirme dışa aktarımlarını devre dışı bırakır
- `--visual_export_format JPG` - Dışa aktarma formatını belirler (JPG, PNG vb.)

**Veri Dışa Aktarımları:**

- `--export_pickle` - Prediction pickle dosyalarını dışa aktarır
- `--export_crop` - Kırpılmış tespitleri dışa aktarır

#### Inference Modları

SAHI varsayılan olarak çok aşamalı inference gerçekleştirir (hem standart hem sliced prediction):

- `--no_sliced_prediction` - Sliced inference'ı devre dışı bırakır (yalnızca standart)
- `--no_standard_prediction` - Standart inference'ı devre dışı bırakır (yalnızca sliced)

### COCO Veri Kümesi Değerlendirmesi

Değerlendirme amacıyla bir COCO annotation dosyası kullanarak tahmin yürütün:

```bash
sahi predict --dataset_json_path dataset.json \
  --source path/to/coco/image/folder \
  --model_path path/to/model
```

Tahminler bir COCO JSON dosyası olarak `runs/predict/exp/results.json` yoluna aktarılacaktır. Ardından şunları kullanabilirsiniz:

- `sahi coco evaluate` - COCO değerlendirme metriklerini hesaplar
- `sahi coco analyse` - Detaylı hata analizi grafiklerini oluşturur

### İlerleme Raporlama

Inference ilerlemesini takip etmek için bir ilerleme çubuğunu etkinleştirin:

```bash
sahi predict --model_path path/to/model --source images/ \
  --slice_width 512 --slice_height 512 --progress_bar
```

> **Not:** `--progress_bar` bayrağı CLI görsel ilerlemesini (tqdm) kontrol eder. `progress_callback` parametresi Python API'sinde mevcuttur ancak CLI seçeneği olarak sunulmamıştır.

---

## `predict-fiftyone` command usage

Sliced Inference gerçekleştirin ve sonuçları FiftyOne uygulaması kullanarak etkileşimli olarak görselleştirin.

### Temel Kullanım

```bash
sahi predict-fiftyone --image_dir image/file/or/folder \
  --dataset_json_path dataset.json \
  --model_path path/to/model \
  --model_config_path path/to/config
```

Bu işlem varsayılan parametrelerle Sliced Inference gerçekleştirir ve etkileşimli inceleme için FiftyOne uygulamasını başlatır.

### Ek Parametreler

[`sahi predict`](#predict-command-usage) komutundaki tüm parametreler desteklenmektedir.

---

## `coco fiftyone` command usage

FiftyOne kullanıcı arayüzünü kullanarak COCO veri kümenizdeki birden fazla tespit sonucunu görselleştirin ve karşılaştırın.

### Temel Kullanım

Tahminlerinizi [COCO result JSON formatına](https://cocodataset.org/#format-results) dönüştürmeniz gerekir. Bu formatı oluşturmak için [`sahi predict`](#predict-command-usage) komutunu kullanın.

```bash
sahi coco fiftyone --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  cocoresult1.json cocoresult2.json
```

Bu komut, veri kümesini görselleştiren ve yanlış tespitlere göre sıralanmış 2 tespit sonucunu karşılaştıran bir FiftyOne uygulamasını açar.

### Seçenekler

- `--iou_threshold 0.5` - FP/TP sınıflandırması için IOU eşiğini belirler

---

## `coco slice` command usage

Büyük görselleri ve COCO formatındaki annotation'larını daha küçük dilimlere (tiles) dilimleyin.

### Temel Kullanım

```bash
sahi coco slice --image_dir dir/to/images \
  --dataset_json_path dataset.json
```

Görselleri ve COCO annotation'larını dilimler, çıktı klasörüne dışa aktarır.

### Parametreler

**Dilim Boyutları:**

- `--slice_size 512` - Dilim yüksekliği ve genişliğini belirler (varsayılan: 512)

**Örtüşme (Overlap):**

- `--overlap_ratio 0.2` - Yükseklik/genişlik için örtüşme oranını belirler (varsayılan: 0.2)

**Filtreleme:**

- `--ignore_negative_samples` - Annotation içermeyen görselleri harici tutar

**Çıktı:**

- `--out_dir output/folder` - Çıktı dizinini belirtir

---

## `coco yolo` command usage

COCO formatındaki veri kümelerini Ultralytics ile eğitim için YOLO formatına dönüştürün.

> **Windows Kullanıcıları:** Sembolik bağlantıları (symlinks) düzgün oluşturmak için Anaconda komut istemini veya Windows CMD'yi **yönetici olarak** açın.

### Temel Kullanım

```bash
sahi coco yolo --image_dir dir/to/images \
  --dataset_json_path dataset.json \
  --train_split 0.9
```

COCO veri kümesini YOLO formatına dönüştürür ve `runs/coco2yolo/exp` klasörüne aktarır.

### Parametreler

- `--train_split 0.9` - Eğitim bölme oranını belirler (varsayılan: 0.9)
- `--out_dir output/folder` - Çıktı dizinini belirtir

---

## `coco evaluate` command usage

Tahminleriniz için COCO değerlendirme metriklerini (mAP, mAR) hesaplayın.

### Temel Kullanım

Tahminlerinizi [COCO result JSON formatına](https://cocodataset.org/#format-results) dönüştürmeniz gerekir. Bu formatı oluşturmak için [`sahi predict`](#predict-command-usage) komutunu kullanın.

```bash
sahi coco evaluate --dataset_json_path dataset.json \
  --result_json_path result.json
```

COCO değerlendirme metriklerini hesaplar ve sonuçları çıktı klasörüne aktarır.

### Parametreler

**Metrik Tipi:**

- `--type bbox` - Bounding box tespitlerini değerlendirir (varsayılan)
- `--type mask` - Instance segmentation maskelerini değerlendirir

**Skorlama Seçenekleri:**

- `--classwise` - Genel metriklerin yanı sıra sınıf bazında skorları hesaplar

**Tespit Limitleri:**

- `--proposal_nums "[10 100 500]"` - Görsel başına maksimum tespiti belirler (varsayılan: `[100, 300, 1000]`)

**IOU Eşikleri:**

- `--iou_thrs 0.5` - IOU eşiğini belirtir (varsayılan: 0.50:0.95 ve 0.5)

**Çıktı:**

- `--out_dir output/folder` - Çıktı dizinini belirtir

---

## `coco analyse` command usage

COCO tahminleri için ayrıntılı hata analizi grafikleri oluşturun.

### Temel Kullanım

Tahminlerinizi [COCO result JSON formatına](https://cocodataset.org/#format-results) dönüştürmeniz gerekir. Bu formatı oluşturmak için [`sahi predict`](#predict-command-usage) komutunu kullanın.

```bash
sahi coco analyse --dataset_json_path dataset.json \
  --result_json_path result.json \
  --out_dir output/directory
```

Kapsamlı hata analizi grafikleri oluşturur ve bunları belirtilen klasöre aktarır.

### Parametreler

**Analiz Tipi:**

- `--type bbox` - Bounding box tespitlerini analiz eder (varsayılan)
- `--type segm` - Instance segmentation maskelerini analiz eder

**Ek Grafikler:**

- `--extraplots` - Ek mAP çubuk grafikleri ve annotation alanı istatistikleri oluşturur

**Alan Bölgeleri:**

- `--areas "[1024 9216 10000000000]"` - Analiz için alan bölgelerini tanımlar (varsayılan: küçük/orta/büyük COCO alanları)

---

## `env` command usage

SAHI ile ilgili yüklü paket sürümlerini görüntüleyin.

### Kullanım

```bash
sahi env
```

### Örnek Çıktı

```text
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   torch version 2.1.2 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   torchvision version 0.16.2 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   ultralytics version 8.3.86 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   transformers version 4.49.0 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   timm version 0.9.1 is available.
06/19/2022 21:24:52 - INFO - sahi.utils.import_utils -   fiftyone version 0.14.2 is available.
```

---

## `version` command usage

Mevcut yüklü SAHI sürümünüzü görüntüleyin.

### Kullanım

```bash
sahi version
0.11.22
```

---

## Özel Komut Dosyaları (Custom Scripts)

Tüm komut dosyaları [scripts dizininden](https://github.com/obss/sahi/tree/main/scripts) indirilebilir ve özel ihtiyaçlarınıza göre değiştirilebilir.

SAHI'yi pip ile kurduktan sonra tüm komut dosyaları herhangi bir dizinden çağrılabilir:

```bash
python script_name.py
```

---

## Ek Kaynaklar

- [Prediction Araçları](predict.md): Prediction parametreleri ve görselleştirme için Python API'si
- [COCO Araçları](coco.md): COCO veri kümesi operasyonları için Python API'si
- [Model Entegrasyonları](guides/models.md): Kütüphaneye özel kurulum kılavuzları
- [Etkileşimli Notebook'lar](notebooks.md): Tüm kütüphaneler için uygulamalı örnekler
