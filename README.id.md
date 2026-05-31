# Prediksi Kegagalan Pinjaman Lending Club

**Bahasa Indonesia** | [English](README.md)

Proyek magang: klasifikasi biner pinjaman Lending Club (2007–2014) untuk memprediksi apakah pinjaman berakhir **baik (lunas)** atau **buruk (macet / gagal bayar)**, hanya dengan informasi yang tersedia saat **pengajuan pinjaman**.

---

## Masalah

| Label | Arti | Contoh nilai `loan_status` |
|-------|------|----------------------------|
| **1 (good)** | Pinjaman lunas dengan baik | Fully Paid |
| **0 (bad)** | Gagal bayar / keterlambatan serius | Charged Off, Default, Late (31–120 days) |

Data **tidak seimbang** (~79% good, ~21% bad setelah pembersihan). Akurasi tinggi saja bisa menipu jika model jarang mendeteksi pinjaman berisiko. Notebook mengoptimalkan dan melaporkan **F1, precision, dan recall untuk kelas bad (0)**.

Pinjaman yang masih berjalan (`Current`, `In Grace Period`, dll.) dibuang agar label mencerminkan **status akhir**.

---

## Data

| Berkas | Keterangan |
|--------|------------|
| `loan_data_2007_2014.csv` | Ekspor mentah Lending Club (~466 ribu baris). **Tidak ada di git** (lihat `.gitignore`). |
| `LCDataDictionary.xlsx` | Kamus kolom (referensi opsional). **Tidak ada di git**. |

Letakkan `loan_data_2007_2014.csv` di folder ini (`internship/`) sebelum menjalankan notebook.

**Setelah pembersihan** (`CustomTransformer`): ~238 ribu baris, **32 fitur** + `label`.

---

## Struktur proyek

```
internship/
├── README.md              # Versi Inggris
├── README.id.md           # Berkas ini (Bahasa Indonesia)
├── main.ipynb             # Alur lengkap: bersih → model → simpan
├── data_learning.ipynb    # Eksplorasi pembersihan (referensi)
├── LCDataDictionary.xlsx  # Kamus data (lokal saja)
├── loan_data_2007_2014.csv
└── saved_models/          # Artefak joblib setelah pelatihan
```

Repo induk: `requirements.txt` di `Part 2/requirements.txt`.

---

## Persiapan lingkungan

Dari akar repositori (`Part 2/`):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
pip install xgboost             # opsional; grid search melewati XGBoost jika belum terpasang
pip install joblib seaborn      # dipakai di main.ipynb
```

Jalankan Jupyter dari folder `internship/` agar path relatif benar:

```bash
cd internship
jupyter notebook main.ipynb
```

Jalankan sel **dari atas ke bawah**. Grid search dan pelatihan MLP bisa memakan waktu lama di CPU.

---

## Notebook

### `main.ipynb` (deliverable utama)

1. **Muat & bersihkan** — `CustomTransformer` menerapkan logika yang sama dengan `data_learning.ipynb` dalam transformer yang kompatibel scikit-learn.
2. **Pisah data** — Train/test stratified 80/20 (`random_state=42`).
3. **Praproses** — `ColumnTransformer`: numerik → imputasi + `StandardScaler`; kategorikal → imputasi + `TargetEncoder` (target biner).
4. **EDA** — Heatmap korelasi pada data latih yang sudah diproses.
5. **Grid search** — Logistic Regression, Linear SVC, Random Forest, HistGradientBoosting, XGBoost (jika terpasang); CV stratified 3-fold; skor = **F1 kelas 0**; penanganan ketidakseimbangan kelas.
6. **Simpan** — Artefak di `saved_models/`.
7. **MLP** — Jaringan feedforward Keras pada matriks yang sama; early stopping; simpan joblib.
8. **Ringkasan** — Narasi dan infografis (sel terakhir).

### `data_learning.ipynb`

Pembersihan data langkah demi langkah untuk eksplorasi. **`main.ipynb` tidak wajib dijalankan setelah notebook ini**; `CustomTransformer` meniru langkah-langkahnya untuk pipeline yang dapat direproduksi.

---

## Gambaran alur

```
CSV mentah
  → CustomTransformer (bersih, fitur, label)
  → train_test_split
  → preprocessor (fit hanya di train)
  → model (grid search + MLP)
  → saved_models/
```

### `CustomTransformer` (ringkas)

1. Buang kolom indeks dan kolom kosong 100%  
2. String kosong → NaN pada kolom teks  
3. Parse `term`, tanggal, ID  
4. Filter status non-final; petakan `loan_status` → `label`  
5. Rekayasa fitur (`has_desc`, `emp_length`, `home_ownership`, `emp_pay_tier`, dll.)  
6. Enkode kolom bulan-sejak; tangani `next_pymnt_d`  
7. Buang kebocoran informasi / pasca-origination dan kolom jarang  
8. Enkoding numerik akhir (hari epoch, sentinel, flag)  
9. Isi NaN `emp_length` dan `annual_inc` dengan 0  

Kolom kebocoran (mis. `out_prncp`, `total_pymnt`, `last_pymnt_d`) dihapus agar model hanya melihat informasi **saat origination**.

### Fitur pemodelan (32)

- **Numerik (25):** semua kolom selain tujuh kategorikal di bawah.  
- **Kategorikal (7):** `grade`, `sub_grade`, `home_ownership`, `verification_status`, `purpose`, `addr_state`, `emp_pay_tier`.

Ambang klasifikasi default: **0,5** pada probabilitas prediksi (belum di-tune di notebook).

---

## Model

| Model | Catatan |
|-------|---------|
| Logistic Regression | Baseline linear, `class_weight="balanced"` |
| Linear SVC | Klasifikasi linear skala besar |
| Random Forest | Ensemble non-linear |
| HistGradientBoosting | Gradient boosting sklearn |
| XGBoost | Opsional; sering F1 terbaik pada run contoh |
| MLP (Keras) | 128 → 64 → 32 + dropout, sigmoid, `binary_crossentropy` |

**Ketidakseimbangan kelas:** `class_weight="balanced"` (sklearn), `scale_pos_weight` (XGBoost), `class_weight` Keras dari proporsi label latih.

---

## Artefak `saved_models/`

Dibuat setelah menjalankan sel penyimpanan di `main.ipynb`:

| Pola berkas | Isi |
|-------------|-----|
| `preprocessor.joblib` | `ColumnTransformer` yang sudah di-fit |
| `{Model}_best_estimator.joblib` | Estimator terbaik dari grid search |
| `{Model}_gridsearch.joblib` | Objek `GridSearchCV` lengkap |
| `{Model}_pipeline.joblib` | `preprocess` + `classifier` untuk inferensi |
| `grid_search_summary.csv` | Metrik CV dan uji per model |
| `loan_default_mlp.joblib` | Model Keras + metadata |
| `mlp_test_metrics.joblib` | Kamus metrik uji MLP |

Contoh inferensi dengan pipeline sklearn yang disimpan:

```python
import joblib

pipe = joblib.load("saved_models/XGBoost_pipeline.joblib")
# X_raw: satu atau banyak baris, kolom sama seperti X_train sebelum praproses
# y_pred = pipe.predict(X_raw)
```

Untuk data baru mentah, terapkan `CustomTransformer` dulu, atau bungkus pembersihan + `*_pipeline.joblib` dalam pipeline luar Anda sendiri.

---

## Hasil contoh (referensi)

Dari satu run yang selesai (angka Anda bisa sedikit berbeda):

| Model | Test F1 (bad) | Test ROC-AUC |
|-------|----------------|--------------|
| XGBoost | ~0,497 | ~0,765 |
| HistGradientBoosting | ~0,493 | ~0,765 |
| MLP | ~0,483 | — |

Bentuk data terproses: latih **(190156, 32)**, uji **(47539, 32)**.


## Lisensi dan data

Data historis Lending Club dipakai untuk pembelajaran. Periksa ketentuan Lending Club untuk redistribusi. Jangan commit CSV/XLSX besar (lihat `.gitignore`).
