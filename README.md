# Model Prediksi Penyakit Kulit Wajah

## Deskripsi Project

Project ini membangun model deep learning berbasis transfer learning (MobileNetV2) untuk mengklasifikasikan kondisi kulit wajah dari gambar ke dalam 6 kategori:

- Acne
- Blackheads
- Dark Spots
- Normal Skin
- Oily Skin
- Wrinkles

Selain klasifikasi, aplikasi ini juga memberikan rekomendasi produk skincare yang relevan berdasarkan hasil prediksi.

## Fitur Utama

- **Klasifikasi otomatis** kondisi kulit wajah dari gambar.
- **Rekomendasi produk skincare** sesuai kategori kulit.
- **Ekspor model** ke format `.h5` (Keras) dan `.onnx` (ONNX) untuk deployment multiplatform.
- **Inference web**: Prediksi gambar baru langsung dari browser menggunakan ONNX Runtime Web.

## Struktur Folder

```
.
├── acne.jpeg
├── best_skin_model.h5
├── best_skin_model.onnx
├── ds.webp
├── face_prediction_model.py
├── inference_test.html
├── oily.webp
├── prediction_model.ipynb
├── skincare_product/
│   ├── treatment.csv
│   ├── treatment.json
│   └── gambar_produk/
├── dataset/
│   ├── Acne/
│   ├── Blackheads/
│   ├── Dark Spots/
│   ├── Normal Skin/
│   ├── Oily Skin/
    └── Wrinkles/
```

## Cara Training Model

1. **Jalankan Notebook**

   - Buka [`prediction_model.ipynb`](prediction_model.ipynb) atau [`face_prediction_model.py`](face_prediction_model.py) di Jupyter Notebook/VSCode.
   - Ikuti cell secara berurutan untuk training, fine-tuning, evaluasi, dan ekspor model.
   - Model hasil training akan tersimpan sebagai `best_skin_model.h5` dan diekspor ke `best_skin_model.onnx`.
2. **Dataset**

   - Struktur dataset harus mengikuti format folder per kelas di dalam folder `dataset/`.

## Cara Menjalankan Inference Web

### 1. Pastikan File Berikut Ada di Root Project:

- `inference_test.html`
- `best_skin_model.onnx`
- Folder `skincare_product/` beserta `treatment.json` dan gambar produk (opsional untuk rekomendasi).

### 2. Jalankan Python Live Server

Buka terminal di folder project, lalu jalankan:

```sh
# Untuk Python 3.x
python -m http.server 8000
```

### 3. Akses Web Inference

- Buka browser dan akses: [http://localhost:8000/inference_test.html](http://localhost:8000/inference_test.html)
- Upload gambar wajah pada form yang tersedia.
- Klik tombol **Prediksi**.
- Hasil prediksi kategori kulit dan confidence akan muncul, beserta rekomendasi produk skincare.

### 4. Catatan

- Model ONNX (`best_skin_model.onnx`) digunakan di browser dengan ONNX Runtime Web.
- Rekomendasi produk diambil dari `skincare_product/treatment.json` atau `treatment.csv`.
- Jika gambar produk tidak tampil, pastikan file gambar produk tersedia di `skincare_product/gambar_produk/` dengan nama sesuai ID produk.

## Dependencies

- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib, seaborn, pandas, glob
- tf2onnx, onnx (untuk ekspor model)
- [ONNX Runtime Web](https://www.npmjs.com/package/onnxruntime-web) (sudah di-load via CDN di HTML)

## Referensi Kode

- Training & evaluasi model: [`face_prediction_model.py`](face_prediction_model.py), [`prediction_model.ipynb`](prediction_model.ipynb)
- Web inference: [`inference_test.html`](inference_test.html)
- Dataset & produk: [`skincare_product/treatment.csv`](skincare_product/treatment.csv), [`skincare_product/treatment.json`](skincare_product/treatment.json)

---

**Lisensi:** Project ini untuk keperluan edukasi dan non-komersial.
