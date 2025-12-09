# Model Prediksi Penyakit Kulit Wajah

## Deskripsi Singkat

Model deep learning berbasis transfer learning (MobileNetV2) untuk klasifikasi kondisi kulit wajah menjadi 6 kategori:

- Acne
- Blackheads
- Dark Spots
- Normal Skin
- Oily Skin
- Wrinkles

Selain klasifikasi, aplikasi menyediakan rekomendasi produk skincare berdasarkan hasil prediksi. Model juga dapat diekspor ke ONNX untuk dipakai pada web inference di proyek terpisah (opsional).

## Fitur Utama

- Klasifikasi otomatis kondisi kulit wajah dari gambar (MobileNetV2 + head kustom).
- Rekomendasi produk skincare sesuai kategori hasil prediksi (berdasarkan `treatment.csv/json`).
- Penanganan ketidakseimbangan data dengan oversampling berbasis `tf.data` dan decoding gambar robust via PIL.
- Evaluasi otomatis: Confusion Matrix (heatmap) dan Classification Report tersimpan sebagai artefak.
- Ekspor model: `.h5` (Keras) dan `.onnx` (ONNX) untuk deployment multiplatform.
- Inference lokal via fungsi Python helper pada `prediction_model.py`.

## Struktur Proyek (ringkas)

```
.
├── best_skin_model.h5                 # Model Keras tersimpan
├── best_skin_model.onnx               # Model ONNX untuk web inference
├── face_prediction_model.py           # (Opsional) Script/eksperimen terkait
├── prediction_model.py                # Pipeline training/evaluasi/ekspor
├── prediction_model.ipynb             # Notebook (alternatif)
├── skincare_product/
│   ├── treatment.csv
│   ├── treatment.json
│   └── gambar_produk/
└── dataset/
      ├── Acne/
      ├── Blackheads/
      ├── Dark Spots/
      ├── Normal Skin/
      ├── Oily Skin/
      └── Wrinkles/
```

## Persiapan Lingkungan

1) (Disarankan) Buat virtual environment Python 3.x.
2) Install dependensi dari file yang tersedia di project:

```cmd
pip install -r requirments.txt
```

Catatan: File bernama `requirments.txt` tersedia di repo ini dan mencakup TensorFlow, scikit-learn, matplotlib, seaborn, pandas, tf2onnx, onnx, dan Pillow. ONNX Runtime Web hanya diperlukan jika Anda membangun frontend web di proyek terpisah.

## Training Model

Jalankan salah satu:

- Notebook: buka `prediction_model.ipynb` dan jalankan sel berurutan.
- Skrip Python: jalankan `prediction_model.py`.

```cmd
python prediction_model.py
```

Selama training, pipeline melakukan:

- Stratified split train/val dan oversampling seimbang per-kelas dengan `tf.data.experimental.sample_from_datasets`.
- Decoding gambar robust menggunakan PIL untuk menghindari error format.
- Transfer learning MobileNetV2 (freeze backbone) + head klasifikasi.
- Callbacks: EarlyStopping, ReduceLROnPlateau, dan ModelCheckpoint (`best_skin_model.h5`).

## Evaluasi (Confusion Matrix & Report)

Setelah training selesai, evaluasi validation set berjalan otomatis di `prediction_model.py` dan menghasilkan:

- `confusion_matrix.png` — heatmap Confusion Matrix.
- `classification_report.txt` — precision/recall/F1 per kelas.

Anda dapat membuka PNG/TXT tersebut untuk menilai performa model per kategori.

## Ekspor Model

Skrip mengekspor model Keras terbaik ke ONNX:

- File: `best_skin_model.onnx`
- Tool: `tf2onnx`

## Inference Lokal (Python)

Anda dapat melakukan prediksi pada gambar tunggal menggunakan helper yang telah disediakan pada `prediction_model.py`.

Contoh ringkas:

```python
from prediction_model import predict_skin_condition, show_recommendations, categories
import tensorflow as tf

model = tf.keras.models.load_model('best_skin_model.h5')
img_path = 'path_ke_gambar_uji.jpg'
pred_idx, pred_name, conf = predict_skin_condition(model, img_path, categories)
print(pred_idx, pred_name, conf)
show_recommendations(pred_name)
```

Jika kelak Anda ingin membuat web inference, gunakan file `best_skin_model.onnx` pada proyek frontend terpisah dengan ONNX Runtime Web.

## Konversi .py ke .ipynb (opsional)

Repositori menyertakan notebook konverter `convert_py_to_ipynb.ipynb`. Atur path input/output pada sel pertama dan jalankan semua sel untuk membuat notebook dari file `.py`.

Alternatif via Jupytext:

```cmd
pip install jupytext
jupytext --to notebook "c:\Users\pael\Documents\pengumpulan dicoding\PMLDI\prediction_model.py"
```

## Troubleshooting

- Error decoding gambar: Pastikan hanya format JPEG/PNG/BMP/GIF dan gunakan pipeline dengan PIL (sudah diaktifkan di skrip).
- Gambar produk tidak muncul: Periksa penamaan file di `skincare_product/gambar_produk/` sesuai `Id` pada CSV/JSON.
- Performa kurang stabil: Coba perpanjang training atau aktifkan fine-tuning sebagian layer MobileNetV2, serta sesuaikan `steps_per_epoch` dengan ukuran data.

---

Lisensi: Proyek ini untuk keperluan edukasi dan non-komersial.
