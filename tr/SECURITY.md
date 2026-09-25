# Güvenlik Politikası

## Desteklenen Sürümler

Aşağıdaki sürümler için güvenlik zafiyetlerine yönelik yamalar yayınlıyoruz:

| Sürüm  | Desteklenen        |
| ------ | ------------------ |
| 0.11.x | :white_check_mark: |
| < 0.11 | :x:                |

## Zafiyet Bildirimi

SAHI güvenliğini ciddiye alıyoruz. SAHI bünyesinde bir güvenlik zafiyeti bulduğunuzu düşünüyorsanız, lütfen bunu aşağıda açıklandığı şekilde bize bildirin.

### Nereye Bildirilmeli

**Lütfen güvenlik zafiyetlerini halka açık GitHub issue'ları üzerinden bildirmeyin.**

Bunun yerine lütfen aşağıdaki yöntemlerden biriyle bildirin:

1. **GitHub Security Advisories** (Tercih edilen)
   - [Security Advisories](https://github.com/obss/sahi/security/advisories) sayfasına gidin
   - "Report a vulnerability" seçeneğine tıklayın
   - Zafiyet ayrıntılarını doldurun

2. **E-posta**
   - GitHub üzerinden bakımlayıcılara e-posta gönderin
   - Konu satırına "SECURITY" ifadesini ekleyin
   - Zafiyet hakkında detaylı bilgi verin

### Neler Dahil Edilmeli

Lütfen raporunuza aşağıdaki bilgileri ekleyin:

- Zafiyet türü (örn. uzaktan kod çalıştırma, bilgi ifşası vb.)
- Zafiyetle ilgili kaynak dosya(lar)ın tam yolları
- Etkilenen kaynak kodun konumu (etiket/dal/commit veya doğrudan URL)
- Sorunu yeniden üretmek için gereken özel konfigürasyonlar
- Sorunu adım adım yeniden üretme talimatları
- Kavram kanıtı (Proof-of-concept) veya istismar kodu (mümkünse)
- Bir saldırganın bunu nasıl istismar edebileceği dahil sorunun etkisi

### Yanıt Zaman Çizelgesi

- Raporunuzu **3 iş günü** içinde onaylayacağız
- Sonraki adımları belirten detaylı bir yanıtı **7 iş günü** içinde sunacağız
- Düzeltme ve tam duyuru sürecindeki ilerleme hakkında sizi bilgilendireceğiz
- Ek bilgi veya rehberlik talep edebiliriz

### İfşa Politikası

- Güvenlik sorunları yüksek öncelikle ele alınacaktır
- Bir düzeltme hazır olduğunda:
  1. Bir yama sürümü yayınlayacağız
  2. GitHub'da bir güvenlik danışma belgesi (security advisory) yayınlayacağız
  3. Sizi kredi olarak ekleyeceğiz (anonim kalmayı tercih etmediğiniz sürece)
  4. CHANGELOG belgesini güvenlik düzeltmesi bilgisiyle güncelleyeceğiz

### Kullanıcılar İçin Güvenlik En İyi Uygulamaları

SAHI kullanırken şunları öneriyoruz:

1. **SAHI'yi en son sürüme güncel tutun**
2. Bilinen zafiyetler için **bağımlılıkları düzenli olarak gözden geçirin**
3. Güvenilmeyen görselleri veya modelleri işlerken **girdileri doğrulayın**
4. SAHI ve bağımlılıklarını izole etmek için **sanal ortamlar (virtual environments) kullanın**
5. SAHI'yi prodüksiyonda çalıştırırken **en az yetki ilkesine (least privilege) uyun**
6. Güvenilmeyen kaynaklardan gelen **model ağırlıklarına (weights) karşı dikkatli olun**

### Bilinen Güvenlik Değerlendirmeleri

- **Model Yükleme**: Güvenilmeyen kaynaklardan model ağırlıkları yüklerken dikkatli olun
- **Görsel İşleme**: Özellikle güvenilmeyen kaynaklardan gelen görsel girdilerini doğrulayın ve temizleyin (sanitize edin)
- **Dosya İşlemleri**: SAHI dosya I/O operasyonları gerçekleştirir; uygun izinleri ve yol doğrulamasını sağlayın
- **Bağımlılıklar**: Bazı isteğe bağlı bağımlılıkların (PyTorch vb.) kendi güvenlik değerlendirmeleri olabilir

### Güvenlik Güncellemeleri

Güvenlik güncellemeleri şu kanallardan duyurulacaktır:

- [GitHub Security Advisories](https://github.com/obss/sahi/security/advisories)
- [GitHub Releases](https://github.com/obss/sahi/releases)
- [CHANGELOG.md](changelog.md)

## Bug Bounty Programı

Şu anda bir bug bounty programımız bulunmamaktadır. Ancak, zafiyetleri bize sorumluluk bilinciyle bildiren güvenlik araştırmacılarına son derece müteşekkiriz.

## İletişim

Güvenlikle ilgili her türlü soru veya endişeniz için lütfen GitHub üzerinden bakımlayıcılarla iletişime geçin.

---

SAHI ve kullanıcılarının güvende kalmasına yardımcı olduğunuz için teşekkür ederiz!
