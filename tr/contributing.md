---
hide:
  - navigation
tags:
  - contributing
  - development
---

# SAHI'ye Katkıda Bulunma

SAHI'ye katkıda bulunmak istediğiniz için teşekkür ederiz! Bu kılavuz hızlıca başlamanıza yardımcı olacaktır.

## Geliştirme Ortamını Kurma

### 1. Fork ve Klonlama

```bash
git clone https://github.com/KULLANICI_ADINIZ/sahi.git
cd sahi
```

### 2. Ortam Oluşturma

Geliştirme için Python 3.10 kullanmanızı öneririz:

```bash
pip install uv
uv venv --python 3.10
source .venv/bin/activate  # Windows kullanıcıları: .venv\Scripts\activate
```

### 3. Bağımlılıkları Yükleme

```bash
# Çekirdek + geliştirme bağımlılıklarını yükleyin
uv sync --extra dev

# Belirli bir modeli test etmek isterseniz onun bağımlılıklarını da yükleyin.
```

## Kod Formatlama

Kod formatlama ve lint kontrolleri için `ruff` kullanıyoruz. Kodu formatlamak için:

```bash
# Format kontrolü
uv run ruff check .
uv run ruff format --check .

# Formatı düzeltme
uv run ruff check --fix .
uv run ruff format .
```

Veya kolaylık sağlayan komut dosyasını kullanın:

```bash
# Format kontrolü
python scripts/format_code.py check

# Formatı düzeltme
python scripts/format_code.py fix
```

## Testleri Çalıştırma

```bash
# Tüm testleri çalıştırın
uv run pytest

# Belirli bir test dosyasını çalıştırın
uv run pytest tests/test_predict.py

# Kapsam (coverage) raporu ile çalıştırın
uv run pytest --cov=sahi
```

## Pull Request Gönderme

1. Yeni bir dal oluşturun: `git checkout -b feature-name`
2. Değişikliklerinizi yapın
3. Kodu formatlayın: `python scripts/format_code.py fix`
4. Testleri çalıştırın: `uv run pytest`
5. Net bir commit mesajı yazın: `git commit -m "Add feature X"`
6. Push yapın ve PR oluşturun: `git push origin feature-name`

## CI Derleme Başarısızlıkları

CI derlemesi format sorunları nedeniyle başarısız olursa:

1. CI çıktısını inceleyin ve başarısız olan Python sürümünü onaylayın
2. O sürümle bir ortam oluşturun:

    ```bash
    uv venv --python 3.X  # X yerine CI sürümünü yazın
    source .venv/bin/activate
    ```

3. Geliştirme bağımlılıklarını yükleyin:

    ```bash
    uv sync --extra dev
    ```

4. Formatı düzeltin:

    ```bash
    python scripts/format_code.py fix
    ```

5. Değişiklikleri commit edin ve push yapın

## Yeni Model Desteği Ekleme

Yeni bir tespit kütüphanesi desteği eklemek için:

1. `sahi/models/your_framework.py` altında yeni bir dosya oluşturun
2. `DetectionModel` sınıfından türeyen bir sınıf yazın
3. `sahi/auto_model.py` içinde `MODEL_TYPE_TO_MODEL_CLASS_NAME` sözlüğüne kütüphanenizi ekleyin
4. `tests/test_yourframework.py` altına test ekleyin
5. `demo/inference_for_your_framework.ipynb` altına örnek bir notebook ekleyin
6. Yeni modelinizi içerecek şekilde [`README.md`](https://github.com/obss/sahi/blob/main/README.md) ve `docs/` altındaki ilgili dokümanları güncelleyin

Lütfen `sahi/models/ultralytics.py` gibi mevcut uygulamaları referans alın.

## Sorunuz Var mı?

Herhangi bir sorunuz varsa, [bir tartışma (discussion) başlatabilirsiniz](https://github.com/obss/sahi/discussions)!
