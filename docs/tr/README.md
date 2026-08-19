<div align="center">
<img width="90" alt="SAHI logo" src="https://raw.githubusercontent.com/obss/sahi/main/docs/images/sahi-logo.svg">
<h1>
  SAHI: Slicing Aided Hyper Inference
</h1>

<h4>
  Geniş ölçekli object detection & instance segmentation için hafif bir bilgisayarlı görü kütüphanesi
</h4>

<h4>
    <img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sahi-sliced-inference-overview.avif">
</h4>

<!-- Downloads & Version -->
<div>
  <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi" alt="Toplam İndirme"></a>
  <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi/month" alt="Aylık İndirme"></a>
  <a href="https://badge.fury.io/py/sahi"><img src="https://badge.fury.io/py/sahi.svg" alt="PyPI Sürümü"></a>
  <a href="https://anaconda.org/conda-forge/sahi"><img src="https://anaconda.org/conda-forge/sahi/badges/version.svg" alt="Conda Sürümü"></a>
  <a href="https://github.com/obss/sahi/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/sahi" alt="Lisans"></a>
</div>

<!-- CI & Quality -->
<div>
  <a href="https://github.com/obss/sahi/actions/workflows/ci.yml"><img src="https://github.com/obss/sahi/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://security.snyk.io/package/pip/sahi"><img src="https://img.shields.io/badge/Snyk_security-monitored-8A2BE2" alt="Bilinen Zafiyetler"></a>
  <a href="https://www.codefactor.io/repository/github/onuralpszr/sahi"><img src="https://www.codefactor.io/repository/github/onuralpszr/sahi/badge" alt="CodeFactor"></a>
  <a href="https://ieeexplore.ieee.org/document/9897990"><img src="https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg" alt="DOI"></a>
</div>

<!-- AI & Docs -->
<div>
  <a href="https://context7.com/obss/sahi"><img src="https://img.shields.io/badge/Context7%20MCP-Indexed-blue" alt="Context7 MCP"></a>
  <a href="https://context7.com/obss/sahi/llms.txt"><img src="https://img.shields.io/badge/llms.txt-✓-brightgreen" alt="llms.txt"></a>
  <a href="https://deepwiki.com/obss/sahi"><img src="https://img.shields.io/badge/DeepWiki-obss%2Fsahi-blue.svg?logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHVknoxNWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==" alt="DeepWiki"></a>
  <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>
</div>

</div>

## <div align="center">Genel Bakış</div>

SAHI, büyük görsellerdeki küçük nesneleri tespit etmek için **Sliced Inference** tekniğini etkinleştirerek geliştiricilerin gerçek dünyadaki object detection zorluklarını aşmasına yardımcı olur. Çeşitli popüler tespit modellerini destekler ve kullanımı kolay API'ler sunar.

<div align="center">

🌐 [English](../../README.md) | 🇨🇳 [简体中文](../zh/README.md) | 🇹🇷 [Türkçe](README.md)

</div>

| Komut                                                     | Açıklama                                                                                                                                                                                                                                                                                                                                                                                                                           |
| --------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [predict](cli.md#predict-command-usage)                   | Herhangi bir [ultralytics](https://github.com/ultralytics/ultralytics) / [mmdet](https://github.com/open-mmlab/mmdetection) / [huggingface](https://huggingface.co/models?pipeline_tag=object-detection&sort=downloads) / [torchvision](https://pytorch.org/vision/stable/models.html#object-detection) modelini kullanarak sliced/standart video/görsel tahmini gerçekleştirin, bkz. [CLI kılavuzu](cli.md#predict-command-usage) |
| [predict-fiftyone](cli.md#predict-fiftyone-command-usage) | Desteklenen herhangi bir modeli kullanarak sliced/standart tahmin gerçekleştirin ve sonuçları [fiftyone uygulamasında](https://github.com/voxel51/fiftyone) inceleyin, [daha fazla bilgi](fiftyone.md)                                                                                                                                                                                                                             |
| [coco slice](cli.md#coco-slice-command-usage)             | COCO annotation ve görsel dosyalarını otomatik dilimleyin, bkz. [dilimleme araçları](slicing.md)                                                                                                                                                                                                                                                                                                                                   |
| [coco fiftyone](cli.md#coco-fiftyone-command-usage)       | COCO veri kümenizdeki birden fazla tahmin sonucunu yanlış tespit sayısına göre sıralı olarak [fiftyone arayüzü](https://github.com/voxel51/fiftyone) ile inceleyin                                                                                                                                                                                                                                                                 |
| [coco evaluate](cli.md#coco-evaluate-command-usage)       | Verilen tahminler ve ground truth için sınıf bazında COCO AP ve AR değerlerini değerlendirin, bkz. [COCO araçları](coco.md)                                                                                                                                                                                                                                                                                                        |
| [coco analyse](cli.md#coco-analyse-command-usage)         | Birçok hata analizi grafiği hesaplayın ve dışa aktarın, bkz. [kapsamlı kılavuz](index.md)                                                                                                                                                                                                                                                                                                                                          |
| [coco yolo](cli.md#coco-yolo-command-usage)               | Herhangi bir COCO veri kümesini otomatik olarak [ultralytics](https://github.com/ultralytics/ultralytics) formatına dönüştürün                                                                                                                                                                                                                                                                                                     |

### Topluluk Tarafından Onaylandı

[📜 SAHI'ye atıfta bulunan yayınların listesi (şu anda 600+)](https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites=14065474760484865747&scipsc=&q=&scisbd=1)

[🏆 SAHI kullanan yarışma kazananlarının listesi](https://github.com/obss/sahi/discussions/688)

### Yapay Zeka Araçları Tarafından Onaylandı

SAHI dokümantasyonu [Context7 MCP bünyesinde indekslenmiştir](https://context7.com/obss/sahi); bu da AI kodlama asistanlarına güncel, sürüme özel kod örnekleri ve API referansları sağlar. Ayrıca yapay zeka tarafından okunabilir dokümantasyonlar için gelişen standarda uygun bir [llms.txt](https://context7.com/obss/sahi/llms.txt) dosyası sunuyoruz.

## <div align="center">Kurulum</div>

### Temel Kurulum

```bash
pip install sahi
```

<details closed>
<summary>
<big><b>Detaylı Kurulum (Açmak için tıklayın)</b></big>
</summary>

- İstediğiniz pytorch ve torchvision sürümünü yükleyin:

```console
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126
```

(mmdet desteği için torch 2.1.2 gereklidir):

```console
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
```

- İstediğiniz tespit kütüphanesini (ultralytics) yükleyin:

```console
pip install ultralytics>=8.3.161
```

- İstediğiniz tespit kütüphanesini (huggingface) yükleyin:

```console
pip install transformers>=4.49.0 timm
```

- İstediğiniz tespit kütüphanesini (yolov5) yükleyin:

```console
pip install yolov5==7.0.14 sahi==0.12.1
```

- İstediğiniz tespit kütüphanesini (mmdet) yükleyin:

```console
pip install mim
mim install mmdet==3.3.0
```

- İstediğiniz tespit kütüphanesini (roboflow) yükleyin:

```console
pip install inference>=0.51.5 rfdetr>=1.6.2
```

</details>

## <div align="center">Hızlı Başlangıç</div>

### Öğrenme Kaynakları

| Kaynak                                                                                                                                       | Tip         |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| [SAHI'ye Giriş](https://medium.com/codable/sahi-a-vision-library-for-performing-sliced-inference-on-large-images-small-objects-c8b086af3b80) | Blog Yazısı |
| [2025 Video Öğretici](https://www.youtube.com/watch?v=ILqMBah5ZvI) ⭐                                                                        | Video       |
| [Resmi Makale](https://ieeexplore.ieee.org/document/9897990) (ICIP 2022 oral)                                                                | Makale      |
| [Ön Eğitimli Ağırlıklar & ICIP 2022 Makale Dosyaları](https://github.com/fcakyon/small-object-detection-benchmark)                           | Benchmark   |
| [FiftyOne ile SAHI Tahminlerini Görselleştirme ve Değerlendirme](https://voxel51.com/blog/how-to-detect-small-objects/)                      | Blog Yazısı |
| [SAHI'yi İnceleme, learnopencv.com](https://learnopencv.com/slicing-aided-hyper-inference/)                                                  | Makale      |
| [Encord Tarafından Açıklanan Slicing Aided Hyper Inference](https://encord.com/blog/slicing-aided-hyper-inference-explained/)                | Makale      |
| [Video Öğretici: Küçük Nesne Tespiti İçin SAHI](https://www.youtube.com/watch?v=UuOJKxn-M8&t=270s)                                           | Video       |
| [Uydu Görüntülerinde Nesne Tespiti](https://blog.ml6.eu/how-to-detect-small-objects-in-very-large-images-70234bab0f98)                       | Blog Yazısı |
| [COCO Veri Kümesi Dönüştürme](https://medium.com/codable/convert-any-dataset-to-coco-object-detection-format-with-sahi-95349e1fe2b7)         | Blog Yazısı |
| [Kaggle Notebook](https://www.kaggle.com/remekkinas/sahi-slicing-aided-hyper-inference-yv5-and-yx)                                           | Notebook    |
| [Hata Analizi Grafikleri & Değerlendirme](https://github.com/obss/sahi/discussions/622) ⭐                                                   | Tartışma    |
| [Etkileşimli Sonuç Görselleştirme ve İnceleme](https://github.com/obss/sahi/discussions/624) ⭐                                              | Tartışma    |
| [Video Inference Desteği](https://github.com/obss/sahi/discussions/626)                                                                      | Tartışma    |
| [Dilimleme Operasyonları Notebook'u](https://github.com/obss/sahi/blob/main/demo/slicing.ipynb)                                              | Notebook    |
| [Tam Dokümantasyon](index.md)                                                                                                                | Doküman     |

### Notebook'lar & Demolar

| Kütüphane          | Notebook                                                                                                                                                                              | Demo                                                                                                                                                      |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| YOLO26             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb)       | -                                                                                                                                                         |
| YOLO11             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb)       | -                                                                                                                                                         |
| YOLO11-OBB         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb)       | -                                                                                                                                                         |
| YOLOE              | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics_yoloe.ipynb) | -                                                                                                                                                         |
| Roboflow / RF-DETR | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_roboflow.ipynb)          | -                                                                                                                                                         |
| RT-DETR v2         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb)       | -                                                                                                                                                         |
| RT-DETR            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_rtdetr.ipynb)            | -                                                                                                                                                         |
| HuggingFace        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_huggingface.ipynb)       | -                                                                                                                                                         |
| GroundingDINO      | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_groundingdino.ipynb)     | -                                                                                                                                                         |
| YOLOv5             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_yolov5.ipynb)            | -                                                                                                                                                         |
| MMDetection        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_mmdetection.ipynb)       | -                                                                                                                                                         |
| Detectron2         | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_detectron2.ipynb)        | -                                                                                                                                                         |
| TorchVision        | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_torchvision.ipynb)       | -                                                                                                                                                         |
| YOLOX              | -                                                                                                                                                                                     | [![HuggingFace Spaces](https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg)](https://huggingface.co/spaces/fcakyon/sahi-yolox) |

<a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img width="600" src="https://user-images.githubusercontent.com/34196005/144092739-c1d9bade-a128-4346-947f-424ce00e5c4f.gif" alt="sahi-yolox"></a>

### Bağımsız Sliced/Standart Tahmin

<img width="700" alt="sahi-predict" src="https://user-images.githubusercontent.com/34196005/149310540-e32f504c-6c9e-4691-8afd-59f3a1a457f0.gif">

`sahi predict` komutunu kullanma hakkında detaylı bilgiyi [CLI dokümantasyonunda](cli.md#predict-command-usage) bulabilir ve gelişmiş kullanım için [prediction API'sini](predict.md) inceleyebilirsiniz.

### Hata Analizi Grafikleri & Değerlendirme

<img width="700" alt="sahi-analyse" src="https://user-images.githubusercontent.com/34196005/149537858-22b2e274-04e8-4e10-8139-6bdcea32feab.gif">

### Etkileşimli Görselleştirme & İnceleme

<img width="700" alt="sahi-fiftyone" src="https://user-images.githubusercontent.com/34196005/149321540-e6dd5f3-36dc-4267-8574-a985dd0c6578.gif">

Etkileşimli görselleştirme ve inceleme için [FiftyOne entegrasyonunu](fiftyone.md) inceleyin.

### Diğer Araçlar

YOLO dönüştürme, veri kümesi dilimleme, alt örnekleme, filtreleme, birleştirme ve bölme operasyonları için [kapsamlı COCO araçları kılavuzunu](coco.md) kontrol edin. Görsel ve veri kümesi dilimleme parametreleri üzerinde detaylı kontrol için [dilimleme araçları dokümanını](slicing.md) inceleyin.

## <div align="center">Atıf (Citation)</div>

Çalışmalarınızda bu paketi kullanıyorsanız lütfen şu şekilde atıfta bulunun:

```bibtex
@article{akyon2022sahi,
  title={Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection},
  author={Akyon, Fatih Cagatay and Altinuc, Sinan Onur and Temizel, Alptekin},
  journal={2022 IEEE International Conference on Image Processing (ICIP)},
  doi={10.1109/ICIP46576.2022.9897990},
  pages={966-970},
  year={2022}
}
```

```bibtex
@software{obss2021sahi,
  author       = {Akyon, Fatih Cagatay and Cengiz, Cemil and Altinuc, Sinan Onur and Cavusoglu, Devrim and Sahin, Kadir and Eryuksel, Ogulcan},
  title        = {{SAHI: A lightweight vision library for performing large scale object detection and instance segmentation}},
  month        = nov,
  year         = 2021,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.5718950},
  url          = {https://doi.org/10.5281/zenodo.5718950}
}
```

## <div align="center">Katkıda Bulunma</div>

Katkılarınızı memnuniyetle karşılıyoruz! Başlamak için lütfen [Katkıda Bulunma Kılavuzumuzu](contributing.md) inceleyin. Tüm katkıda bulunanlara teşekkür ederiz! 🙏

<p align="center">
    <a href="https://github.com/obss/sahi/graphs/contributors">
      <img src="https://contrib.rocks/image?repo=obss/sahi" />
    </a>
</p>
