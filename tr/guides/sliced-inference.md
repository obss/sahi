---
tags:
  - slicing
  - inference
  - small-object-detection
  - conceptual
---

# Sliced Inference Nasıl Çalışır

## Problem: Büyük Görsellerde Küçük Nesneler

Standart object detector'lar, inference yapmadan önce girdi görsellerini sabit bir çözünürlüğe (örn. 640x640) göre yeniden boyutlandırır (resize). Kaynak görseliniz çok daha büyük olduğunda, örneğin bir 4K drone fotoğrafı veya uydu görseli, küçük nesneler sadece birkaç piksellik boyuta düşer ve geleneksel dedektörler tarafından tespit edilemez hale gelir.

<div align="center">
  <img width="700" alt="sliced inference" src="https://raw.githubusercontent.com/obss/sahi/main/resources/sliced_inference.gif">
</div>

## Çözüm: Dilimle, Tespit Et, Birleştir (Slice, Detect, Merge)

SAHI bu sorunu üç adımda çözer:

### 1. Görseli örtüşen (overlapping) dilimlere (slice/tile) ayırma

Girdi görseli daha küçük parçalardan (patch) oluşan bir ızgaraya (grid) bölünür. Her parçanın boyutu dedektörün beklediği boyuta göre ayarlanır (örn. 512x512); böylece her parça içindeki nesneler güvenilir tespit için yeterli piksel detayını korur.

Dilimler arasındaki örtüşen (overlap) bölgeler, dilim sınırında kalan nesnelerin en az bir parçada tamamen görünür olmasını sağlar.

```text
+--------+--------+--------+
|        |overlap |        |
|  tile  |<------>|  tile  |
|   1    |        |   2    |
+--------+--------+--------+
|overlap |        |overlap |
|  tile  |  tile  |  tile  |
|   3    |   4    |   5    |
+--------+--------+--------+
```

Önemli parametreler:

| Parametre | Ne kontrol eder |
| ----------- | ----------------- |
| `slice_height` / `slice_width` | Piksel cinsinden her bir dilimin boyutu |
| `overlap_height_ratio` / `overlap_width_ratio` | Bitişik dilimler arasındaki örtüşme oranı (0.0 ile 1.0 arasında) |
| `auto_slice_resolution` | SAHI'nin dilim boyutlarını görsel çözünürlüğüne göre otomatik seçmesini sağlar |

### 2. Dedektörü her dilimde çalıştırma

Her dilim bağımsız olarak object detection modeline iletilir. Dilimler küçük olduğu için, tam görselde çok küçük olan nesneler artık girdinin anlamlı bir bölümünü kaplar ve güvenilir bir şekilde tespit edilebilir.

İsteğe bağlı olarak SAHI, dedektörü orijinal çözünürlükte **tam görsel** üzerinde de çalıştırır (`perform_standard_pred=True`, varsayılan). Bu işlem, birden fazla dilime bölünebilecek büyük nesneleri yakalar.

### 3. Prediction'ları tam görsele geri birleştirme

Dilim (slice) seviyesindeki prediction'lar tam görsel koordinatlarına geri haritalanır. Dilimler birbiriyle örtüştüğü (overlap ettiği) için aynı nesne genellikle birden fazla dilimde tespit edilir. SAHI, bu tekrarlanan (duplicate) tespitleri birleştirmek veya bastırmak için bir postprocessing adımı uygular:

- **GreedyNMM** (varsayılan): Örtüşen (overlapping) kutuları (bounding box) koordinatlarını ve skorlarını ortalayarak greedy bir şekilde birleştirir. Çoğu kullanım senaryosu için en iyisidir.
- **NMM**: Non-Maximum Merging. GreedyNMM'e benzer ancak tüm örtüşmeleri eşzamanlı olarak işler.
- **NMS**: Non-Maximum Suppression. En yüksek skorlu kutuyu tutar ve örtüşen diğer kutuları eler. Kesin, birleştirilmemiş tespitler istediğinizde kullanın.
- **LSNMS**: Location-Sensitive NMS. Konumsal konumu faktör olarak ekleyen bir varyant.

Birleştirme adımı farklı örtüşme metrikleri kullanabilir:

- **IOS** (Intersection over Smaller): Daha agresif birleştirme; nesne boyutları geniş bir aralıkta değiştiğinde iyidir.
- **IOU** (Intersection over Union): Standart metrik; daha muhafazakardır.

## Sliced Inference Ne Zaman Kullanılmalı?

Sliced Inference en çok şu durumlarda fayda sağlar:

- Görselleriniz modelin girdi çözünürlüğünden belirgin şekilde büyük olduğunda
- **Küçük nesneleri** tespit etmeniz gerektiğinde (uydu görüntülerindeki araçlar, geniş açılı güvenlik kameralarındaki insanlar, yüksek çözünürlüklü denetim fotoğraflarındaki kusurlar)
- Standart tespit nesneleri kaçırdığında veya düşük güven skorları ürettiğinde

Şu durumlarda gerekli olmayabilir:

- Görselleriniz zaten modelin girdi boyutuna yakınsa
- Yalnızca büyük ve belirgin nesnelerle ilgileniyorsanız
- Inference hızı recall (yakalama oranı) değerinden daha önemliyse

## İnce Ayar (Tuning) İpuçları

**Dilim boyutu (Tile size)**: Dedektörün eğitim çözünürlüğü ile eşleştirin. 640x640 boyutunda eğitilmiş YOLO modelleri için 512--640 arası dilimler iyi çalışır.

**Örtüşme oranı (Overlap ratio)**: 0.2 (%20) ile başlayın. Dilim sınırlarında kaçırılan tespitler fark ederseniz 0.3--0.4 seviyesine çıkarın. Daha yüksek örtüşme, daha fazla dilim ve daha yavaş inference demektir.

**Standart prediction**: İlgilendiğiniz tüm nesnelerin küçük olduğundan emin değilseniz `perform_standard_pred=True` olarak tutun. Tam görsel geçişi, dilimler arasında kalacak büyük nesneleri yakalar.

**Postprocessing eşiği**: `postprocess_match_threshold` tekrarlanan (duplicate) tespitlerin ne kadar agresif birleştirileceğini kontrol eder. Düşük değerler daha fazla birleştirir; yüksek değerler ayrı kutuları korur. Varsayılan 0.5 değeri çoğu durum için uygundur.

## Sonraki Adımlar

- [Hızlı Başlangıç](../quick-start.md): SAHI'yi kullanmaya başlayın
- [Model Entegrasyonları](models.md): SAHI'yi kendi tespit kütüphanenizle kullanın
- [Postprocessing Backend'leri](../postprocess/backends.md): Hız için NMS/NMM backend'ini yapılandırın
