---
hide:
  - navigation
  - toc
tags:
  - getting-started
  - object-detection
  - small-object-detection
  - slicing
  - instance-segmentation
---

<div align="center">
<img width="90" alt="SAHI logo" src="https://raw.githubusercontent.com/obss/sahi/main/docs/images/sahi-logo.svg">
<h1>
  SAHI: Slicing Aided Hyper Inference
</h1>

<h4>
  Geniş ölçekli object detection ve instance segmentation için hafif bir bilgisayarlı görü kütüphanesi
</h4>

<h4>
    <img width="700" alt="teaser" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif">
</h4>

<div>
    <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi" alt="downloads"></a>
    <a href="https://pepy.tech/project/sahi"><img src="https://pepy.tech/badge/sahi/month" alt="downloads"></a>
    <a href="https://github.com/obss/sahi/blob/main/LICENSE.md"><img src="https://img.shields.io/pypi/l/sahi" alt="License"></a>
    <a href="https://badge.fury.io/py/sahi"><img src="https://badge.fury.io/py/sahi.svg" alt="pypi version"></a>
    <a href="https://anaconda.org/conda-forge/sahi"><img src="https://anaconda.org/conda-forge/sahi/badges/version.svg" alt="conda version"></a>
    <a href="https://github.com/obss/sahi/actions/workflows/ci.yml"><img src="https://github.com/obss/sahi/actions/workflows/ci.yml/badge.svg" alt="Continuous Integration"></a>
  <br>
    <a href="https://context7.com/obss/sahi"><img src="https://img.shields.io/badge/Context7%20MCP-Indexed-blue" alt="Context7 MCP"></a>
    <a href="https://context7.com/obss/sahi/llms.txt"><img src="https://img.shields.io/badge/llms.txt-✓-brightgreen" alt="llms.txt"></a>
    <a href="https://ieeexplore.ieee.org/document/9897990"><img src="https://img.shields.io/badge/DOI-10.1109%2FICIP46576.2022.9897990-orange.svg" alt="ci"></a>
    <a href="https://arxiv.org/abs/2202.06934"><img src="https://img.shields.io/badge/arXiv-2202.06934-b31b1b.svg" alt="arXiv"></a>
    <a href="https://colab.research.google.com/github/obss/sahi/blob/main/demo/inference_for_ultralytics.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://huggingface.co/spaces/fcakyon/sahi-yolox"><img src="https://raw.githubusercontent.com/obss/sahi/main/resources/hf_spaces_badge.svg" alt="HuggingFace Spaces"></a>
</div>
</div>

## SAHI Nedir?

SAHI (Slicing Aided Hyper Inference), küçük nesne tespiti (small object detection) için jenerik bir slicing destekli inference ve fine-tuning hattı sağlayan açık kaynaklı bir kütüphanedir. Küçük nesneleri ve kameraya uzak nesneleri tespit etmek, güvenlik ve gözetleme uygulamalarında büyük bir zorluktur; çünkü bu nesneler az sayıda piksel ile temsil edilir ve geleneksel dedektörler için yeterli detay barındırmaz.

SAHI, ekstra bir fine-tuning gerektirmeden herhangi bir object detector ile kullanılabilen özgün bir metodoloji uygulayarak bu sorunu çözer. Visdrone ve xView havadan nesne tespiti veri kümeleri üzerindeki deneysel değerlendirmeler, SAHI'nin object detection AP değerini FCOS için %6.8'e, VFNet için %5.1'e ve TOOD dedektörleri için %5.3'e kadar artırabildiğini göstermektedir. Slicing destekli fine-tuning ile doğruluk daha da geliştirilerek sırasıyla %12.7, %13.4 ve %14.5 AP kümülatif artış sağlanabilir. Bu teknik; [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8), [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11), [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26), HuggingFace Transformers (detection, segmentation), RT-DETR, TorchVision, MMDetection, Detectron2, YOLOv5, YOLOE, YOLO-World ve Roboflow RF-DETR modelleri ile başarıyla entegre edilmiştir.

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Hızlı Başlangıç__

    ---

    `sahi` kütüphanesini pip ile kurun ve birkaç dakika içinde kullanmaya başlayın.

    [:octicons-arrow-right-24: Hızlı Başlangıç](quick-start.md)

-   :material-lightbulb-outline:{ .lg .middle } __Nasıl Çalışır__

    ---

    Slicing algoritmasını, ne zaman kullanılacağını ve parametrelerin nasıl ayarlanacağını öğrenin.

    [:octicons-arrow-right-24: Sliced Inference](guides/sliced-inference.md)

-   :material-puzzle-outline:{ .lg .middle } __Model Entegrasyonları__

    ---

    SAHI'yi Ultralytics, HuggingFace, MMDetection, TorchVision ve daha fazlası ile kullanın.

    [:octicons-arrow-right-24: Tüm Modeller](guides/models.md)

-   :material-image:{ .lg .middle } __Predict__

    ---

    SAHI ile yeni görseller, videolar ve stream'ler üzerinde prediction yürütün.

    [:octicons-arrow-right-24: Daha Fazla Bilgi](predict.md)

-   :material-content-cut:{ .lg .middle } __Slicing__

    ---

    Inference için büyük görselleri ve veri kümelerini nasıl dilimleyeceğinizi (slice) öğrenin.

    [:octicons-arrow-right-24: Daha Fazla Bilgi](slicing.md)

-   :material-database:{ .lg .middle } __COCO Araçları__

    ---

    Oluşturma, bölme ve filtreleme dahil COCO formatındaki veri kümeleriyle çalışın.

    [:octicons-arrow-right-24: Daha Fazla Bilgi](coco.md)

-   :material-console:{ .lg .middle } __CLI Komutları__

    ---

    Prediction ve veri kümesi operasyonları için SAHI'yi komut satırından kullanın.

    [:octicons-arrow-right-24: Daha Fazla Bilgi](cli.md)

-   :material-eye:{ .lg .middle } __FiftyOne__

    ---

    Tepki ve tespit sonuçlarını etkileşimli olarak görselleştirin ve karşılaştırın.

    [:octicons-arrow-right-24: Daha Fazla Bilgi](fiftyone.md)

-   :material-notebook:{ .lg .middle } __Notebook'lar__

    ---

    Desteklenen tüm kütüphaneler için uygulamalı Colab notebook'ları.

    [:octicons-arrow-right-24: Notebook'ları İncele](notebooks.md)

</div>
