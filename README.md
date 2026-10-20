# UA-DETRAC Traffic Analytics

UA-DETRAC araç görüntülerini YOLO biçimine hazırlayan, sekans düzeyinde
train/validation ayrımı yapan ve YOLO11 eğitimi çalıştıran proje.
Dört sınıf kullanılır: `car`, `bus`, `van`, `others`.
Şu anki uygulama veri hazırlama, doğrulama ve model eğitimini kapsar;
takip ve trafik analitiği henüz bu kullanım akışına dahil değildir.

## Kurulum

Referans ortam Python **3.11** ve macOS Apple Silicon'dur. Otomatik test akışı
Linux CPU için de tanımlıdır. Komutları repo kökünde çalıştırın:

```sh
git clone https://github.com/ardaatikk/ua-detrac-traffic-analytics.git
cd ua-detrac-traffic-analytics
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
```

`requirements.txt` uygulama bağımlılıklarını, `requirements-dev.txt` bunlara
ilaveten pytest'i kurar. Doğrudan bağımlılıklar çalışan ortamın sürümlerine
sabitlenmiştir; bu dosyalar tüm dolaylı bağımlılıkları kapsayan bir lock dosyası
değildir. Python ortamını sistem paketlerinden ayrı tutun.

Linux'ta yalnızca CPU kullanacaksanız requirements kurulumundan önce:

```sh
python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
```

CUDA için donanımınıza uygun PyTorch kurulumunu
[resmî PyTorch kurulum sayfasından](https://pytorch.org/get-started/locally/)
seçin; mevcut torch/torchvision sabitlemeleriyle uyumluluğu kontrol edin.
Windows yerel kurulum ve symlink izinleri henüz doğrulanmadı; Windows kullanıcıları
Linux ortamındaki akış için WSL kullanabilir, ancak bu yol da ayrıca test edilmelidir.

## Veriyi indirmeden projeyi denemek

```sh
python -m pytest -q
```

Testler geçici klasörlerde sentetik dosyalar oluşturur. Veri indirmez, önceden
indirilmiş ağırlık istemez ve GPU eğitimi başlatmaz. Smoke seçiminin sınıf kapsamı,
sekans ayrımı, symlink hedefleri ve eğitim veri seti yönlendirmesi kontrol edilir.
GitHub Actions aynı testleri Linux CPU ortamında çalıştıracak şekilde hazırlanmıştır.
Gerçek görüntü yükleme ve model öğrenme davranışı bu testlerin kapsamında değildir.

## Gerçek veriyle hızlı eğitim kontrolü

Bu yol UA-DETRAC verisinin indirilmesini ve tamamının hazırlanmasını gerektirir.
Smoke seçeneği eğitim ve validation hacmini azaltır; ilk veri indirme ve dönüştürme
maliyetini azaltmaz. Veri ve ağırlık indirmeleri internet bağlantısı, yeterli disk
alanı ve indirme kaynağına erişim gerektirir. Veri kullanım koşullarını
[UA-DETRAC proje sayfasından](https://detrac-db.rit.albany.edu/) inceleyin.
Veri ve önceden eğitilmiş ağırlıklar bu repo ile dağıtılmaz.

```sh
python -m src.pipeline.prepare_dataset
python -m src.data.create_smoke_dataset
python -m src.training.train --smoke --epochs 1 --batch-size 8 --device cpu --workers 0 --name yolo11n_smoke_cpu
```

Pipeline sırasıyla indirme/kontrol, ham veri doğrulaması, sekans ayrımı,
XML → YOLO dönüşümü ve YOLO doğrulaması yapar. Repo'daki
`data/processed/dataset_split.json` yeniden kullanılır; yoksa yapılandırılmış
seed ile üretilir. Train ve validation sekansları birbirine karıştırılmaz.
Mevcut doğrulanmış YOLO çıktıları tekrar kullanılır.

Smoke modülü varsayılan olarak **640 train / 320 val** görüntüsü seçer.
Her split içinde dört sınıfın temsil edilmesi zorunludur; kalan kareler sekanslara
ve zaman boyunca yayılır. Görüntü ve etiketler kopyalanmaz, ayrı smoke dizinine
symlink oluşturulur. Test split'i kullanılmaz. Ham veri ve tam YOLO veri seti
smoke oluşturma sırasında değiştirilmez.

Normal dataset `data/processed/yolo`, smoke dataset `data/processed/yolo_smoke`
altındadır. `dataset.yaml` yerel mutlak yolu içerir; başka bilgisayara taşıdığınızda
veri hazırlama ve smoke oluşturmayı o bilgisayarda tekrar çalıştırın.
`manifest.json` seçilen kareleri, sekansları, seed'i ve sınıf başına görüntü
sayılarını kaydeder. Aynı kaynak veri ve seed aynı seçimi üretir.

Smoke dizini zaten varsa oluşturma komutu üzerine yazmadan durur; hazırlanmış
veriyle doğrudan eğitim komutunu çalıştırabilirsiniz. Farklı boyutta yeni bir
seçim gerekiyorsa ayrı hedefi Python API ile belirtin. CLI boyutları
`--train-size` ve `--val-size` ile değiştirilebilir.

## Donanım ve normal eğitim

| Donanım | Eğitim seçeneği |
|---|---|
| CPU | `--device cpu --workers 0` |
| Apple Silicon / MPS | `--device mps` |
| NVIDIA CUDA | `--device 0` |

Örnek Apple Silicon smoke denemesi:

```sh
python -m src.training.train --smoke --epochs 1 --batch-size 8 --device mps --name yolo11n_smoke_mps
```

Tam train ve validation setiyle eğitim:

```sh
python -m src.training.train --epochs 100 --batch-size 16 --device cpu --workers 0 --name yolo11n_full_cpu
```

Tam eğitim CPU üzerinde uzun sürebilir. Bellek yetersizse batch boyutunu azaltın.
`--smoke` ile `--fraction` değeri 1.0 kalmalıdır; küçültmek sınıf kapsamını bozabilir.
Model ağırlığı yerelde yoksa Ultralytics ilk eğitim çağrısında indirmeyi dener.
Çıktılar `artifacts/training/<name>/` altında; ağırlıklar `weights/best.pt` ve
`weights/last.pt`, kullanılan ayarlar `training_config.json` dosyasındadır.
Mevcut deney adı üzerine yazılmaz; her yeni eğitimde farklı `--name` kullanın.

Seed `src/config/settings.py` içinde 42'dir. MPS eğitimi deterministik olmayan
işlemlere izin verir; aynı seed sonuçların birebir aynı olmasını garanti etmez.
Smoke metrikleri nihai model karşılaştırması için kullanılmamalıdır.

## Yapılandırma ve doğrulama durumu

Yollar `src/config/paths.py`, sınıflar `src/config/dataset.py`, eğitim ve seçim
varsayılanları `src/config/settings.py` içindedir. `UA_DETRAC_YOLO_DIR` ortam
değişkeni YOLO kökünü değiştirir; smoke kökü bu dizinin `_smoke` ekli kardeşidir.
Bu değişkeni veri hazırlama ve eğitim boyunca aynı değerle kullanın.

Apple M2 üzerinde Python 3.11.2, torch 2.14.0 ve Ultralytics 8.4.173 ile gerçek
640/320 karelik, 1 epoch smoke eğitimi tamamlanmıştır. Bu başarı CPU/CUDA
üzerinde gerçek eğitimin doğrulandığı anlamına gelmez. Temiz sanal ortamda paket indirme/kurulum kontrolü henüz tamamlanmadı.
GitHub Actions sonucu,
akış GitHub'a gönderilip çalıştıktan sonra görülebilir. Büyük veri, üretilmiş
symlinkler, model ağırlıkları ve eğitim çıktıları Git'e eklenmez.
