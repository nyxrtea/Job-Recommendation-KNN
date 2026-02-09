# Job-Recommendation-KNN

Rekomendasi Karir Berdasarkan Minat dan Bakat Menggunakan Algoritma KNN  

---

## Ringkasan Singkat
Proyek ini membangun sebuah sistem rekomendasi pekerjaan berbasis K-Nearest Neighbors (KNN) dan TF-IDF untuk fitur teks. Antarmuka pengguna dibuat dengan Streamlit (file utama: `main.py`). Logika pelatihan, preprocessing, dan KNN manual ada di `knn.py`. Model dan objek preprocessing disimpan/dimuat menggunakan `joblib`.

Tujuan: membantu pengguna (mahasiswa / pencari kerja) menemukan kategori karir dan contoh job title berdasarkan profil mereka (jenis kelamin, jurusan, minat, keterampilan, nilai, sertifikat, status kerja).

---

## Struktur File Utama
- `main.py` — Aplikasi Streamlit: UI, pemuatan data & model, input pengguna, filter rekomendasi, logika prediksi dan tampilan.
- `knn.py` — Implementasi KNN (manual) serta pipeline persiapan data, TF-IDF, normalisasi, pembagian data, pelatihan, evaluasi, penyimpanan model dan vectorizer.
- `dataset_fiks.csv` — (dipakai oleh skrip) dataset utama berisi data pengguna / historis / mapping kategori (harus ada di folder).
- `categorized_jobs.csv` — file mapping category → daftar job title (dipakai untuk menampilkan contoh job).
- `knn_model.joblib` — model KNN terlatih (disimpan saat pipeline pelatihan dijalankan).
- `vectorizer.joblib` — TF-IDF vectorizer yang disimpan.
- `normalization_params.joblib` — scaler yang menyimpan parameter normalisasi.
- `requirement.txt` — daftar dependency yang direkomendasikan.
- `bg1.jpg` — background image (opsional) untuk Streamlit UI.

---

## Cara Menjalankan (Quick Start)
1. Buat virtual environment dan aktifkan.
2. Install dependensi:
   ```bash
   pip install -r requirement.txt
   ```
3. Pastikan file dataset dan model (jika ingin langsung prediksi) tersedia:
   - `dataset_fiks.csv`, `categorized_jobs.csv`, `knn_model.joblib`, `vectorizer.joblib`, `normalization_params.joblib`
4. Jalankan UI Streamlit:
   ```bash
   streamlit run main.py
   ```
5. Buka browser pada alamat yang ditampilkan (biasanya http://localhost:8501)

---

## Alur Kerja Aplikasi (User Flow)
1. Pengguna memilih menu lewat sidebar: "Tentang Kami", "Tentang Aplikasi", atau "Rekomendasi Pekerjaan".
2. Pada menu "Rekomendasi Pekerjaan":
   - Pengguna memasukkan profil: gender, jurusan (major), minat (interests), keterampilan (skills), rata-rata nilai (CGPA), apakah punya sertifikat, status kerja, dan (opsional) judul sertifikat.
   - Tekan tombol "Cari Pekerjaan".
3. Aplikasi melakukan dua jalur:
   - Filter berbasis aturan/kesamaan dari dataset lama (`filter_jobs`) — menghitung kecocokan Interest Match + Skill Match untuk menampilkan rekomendasi terurut.
   - Prediksi kategori karir dari model KNN:
     - Gabungkan teks (interests + skills + certification_title + major)
     - Transform TF-IDF → pad jika dimensi berbeda → normalisasi (scaler) → prediksi `knn_model.predict`
4. Tampilkan hasil: Predicted Career Category dan daftar job titles contoh per kategori (diambil dari `categorized_jobs.csv`).

---

## Penjelasan Modul & Fungsi Utama

### main.py
- UI & Layout
  - Mengatur tampilan halaman Streamlit dan background via `set_background()`.
  - Sidebar: navigasi menu utama.
  - Menu "Rekomendasi Pekerjaan": seluruh form input profil ada di sini.
- Pemuatan resource
  - `load_data(file_path)` dibungkus `@st.cache_resource` untuk cache pembacaan CSV.
  - Memuat model dan objek preprocessing dari joblib:
    ```python
    knn_model = joblib.load('knn_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    scaler = joblib.load('normalization_params.joblib')
    ```
  - Jika file model tidak ditemukan → tampilan error dan stop.
- Input pengguna
  - Gender → dipetakan ke angka
  - Major → pilihan dari dataset lama
  - Interests, Skills → input teks (dipisah koma)
  - CGPA → slider
  - Certification (Yes/No) + optional certification title
  - Status kerja
- Logika prediksi di UI
  - `filter_jobs(data_old, data_new, ...)` — melakukan filtering dataset historis berdasarkan beberapa kondisi (gender, major, CGPA, certification, status), lalu menghitung:
    - Interest Match: jumlah irisan antara minat input dan minat di dataset.
    - Skill Match: jumlah irisan keterampilan.
    - Total Match = Interest Match + Skill Match
  - Ambil top-N hasil (default head(5)), tambahkan kolom `Job Titles` (mengambil job titles dari `categorized_jobs.csv` berdasarkan `Mapped Category`).
  - Prediksi KNN: gabungkan teks → vectorize TF-IDF → jika dimensi TF-IDF lebih kecil dari scaler.n_features_in_, lakukan padding zero → scaling → prediksi.
  - Tampilkan prediksi kategori dan daftar job titles contoh.

Catatan implementasi praktis:
- Ada perlakuan ketika TF-IDF vectorizer menghasilkan jumlah fitur lebih kecil dari input scaler: kode menambahkan padding nol untuk menyesuaikan dimensi agar bisa di-transform oleh scaler.
- Jika model/objek joblib tidak tersedia, UI akan berhenti dengan error yang jelas ke pengguna.

### knn.py
- Class KNN (manual)
  - Implementasi manual sederhana:
    - init(k=5), fit(X, y), _euclidean_distance, _predict, predict
    - _predict: hitung jarak Euclidean ke semua sampel training, ambil k tetangga terdekat, pilih kelas mayoritas via Counter.
- Fungsi utilitas
  - `normalize(X)` — standardisasi manual (mean/std).
  - `train_test_split(X, y, test_size=0.2, random_state=None)` — implementasi shuffling dan split manual.
  - `generate_confusion_matrix(y_true, y_pred)` — buat confusion matrix manual.
  - `accuracy(y_true, y_pred)` — menghitung akurasi.
  - `save_model(model, filename)` dan `load_model(filename)` — menggunakan joblib untuk serialisasi.
- Pipeline pelatihan (ada di bagian bawah `knn.py`)
  1. Baca `dataset_fiks.csv` → gabungkan kolom teks menjadi `combined_text` (`Interests + Skills + Certification Course Title + UG Specialization (Major)`).
  2. TF-IDF: `vectorizer = TfidfVectorizer(max_features=1000)` → fit_transform `combined_text`.
  3. Gabungkan fitur TF-IDF dengan fitur numerik/kategorikal yang sudah dikodekan.
  4. Hapus kolom teks asli.
  5. Normalisasi fitur via `StandardScaler` (disimpan dengan `save_model()`).
  6. Split train/test.
  7. Buat model KNN (contoh k=17 di file) → `fit`, `predict`.
  8. Evaluasi: akurasi manual + confusion matrix.
  9. Simpan `knn_model.joblib`, `vectorizer.joblib`, dan scaler (`normalization_params.joblib`).

---

## Format Data & Kolom Penting
- `dataset_fiks.csv` (contoh kolom, berdasarkan pemakaian di kode)
  - Gender → numeric mapping di UI (0/1/2)
  - UG Specialization (Major)
  - Interests (teks, dipisah koma)
  - Skills (teks, dipisah koma)
  - Certification Courses (0/1)
  - Certification Course Title (opsional teks)
  - Average CGPA/Percentage (numerik)
  - Working Status (0/1)
  - Mapped Category (target label)
- `categorized_jobs.csv`
  - Kolom: `Category`, `Job Title` — digunakan untuk menampilkan contoh job titles per category.

---

## Detil Teknis & Pertimbangan Implementasi
- TF-IDF:
  - Dipakai untuk representasi teks gabungan (`combined_text`).
  - `max_features` disetel (contoh 1000) untuk mengontrol dimensi.
- Pemetaan kategorikal:
  - Kolom kategori non-teks di-encode ke numeric menggunakan `.astype('category').cat.codes`.
- Scaling:
  - Standarisasi fitur numerik + TF-IDF dilakukan dengan `StandardScaler` → scaler disimpan untuk digunakan saat inference.
- Penanganan mismatch dimensi TF-IDF:
  - Saat vectorizer.transform menghasilkan jumlah fitur berbeda dari scaler.n_features_in_, kode menambahkan padding nol agar dimensi cocok. Ini praktis tetapi bukan solusi ideal — versi vectorizer yang dipakai saat training harus sama saat inference.
- KNN Manual vs scikit-learn:
  - Proyek menggunakan KNN manual (class KNN di `knn.py`). Alternatif yang lebih reliable adalah memakai `sklearn.neighbors.KNeighborsClassifier` dengan optimasi indeks untuk skalabilitas.

---

## Contoh Potongan Perilaku Sistem (UI)
- Prediksi kategori:
  - Input teks gabungan → TF-IDF → padding jika perlu → scaling → `knn_model.predict` → tampilkan `Predicted Career Category`.
- Rekomendasi job list:
  - Hasil filter berbasis match dari dataset lama ditampilkan sebagai top suggestions, setiap entri menunjukkan `Mapped Category` dan contoh job titles (top 5).

---

## Troubleshooting Umum
- Error: model/joblib file tidak ditemukan
  - Pastikan `knn_model.joblib`, `vectorizer.joblib`, dan `normalization_params.joblib` ada di folder kerja.
- Hasil prediksi tidak konsisten/dimensi mismatch
  - Pastikan vectorizer yang digunakan untuk inference identik dengan vectorizer saat training.
  - Pastikan `normalization_params.joblib` menyimpan `n_features_in_` yang sesuai.
- Performa lambat saat training KNN manual
  - KNN manual O(n * d) per prediksi; gunakan sklearn KNN atau struktur indeks (KDTree/BallTree) untuk dataset besar.
- TF-IDF fitur tidak mencakup kata baru
  - Jika ada kata baru di input, vectorizer akan mengabaikan kata tsb (baik) tetapi total fitur tetap harus sesuai; jangan gunakan vectorizer yang berbeda dari versi pelatihan.

---

## Rekomendasi Perbaikan & Pengembangan
1. Gunakan sklearn KNeighborsClassifier untuk performa dan integrasi pipeline:
   - Memanfaatkan pipeline scikit-learn (TF-IDF → Scaler → KNN) dan `Pipeline`.
2. Serialisasi model & pipeline menggunakan `joblib.dump(pipeline, 'pipeline.joblib')` agar tidak perlu padding manual dimensi TF-IDF.
3. Evaluasi lebih lengkap:
   - Tambahkan precision, recall, F1 per kelas.
   - Cross-validation untuk memilih k terbaik.
4. Tangani imbalance data:
   - Gunakan oversampling (SMOTE) atau penyesuaian weight jika kategori tidak seimbang.
5. UI/UX:
   - Validasi input minat/keterampilan (contoh: normalisasi lowercase, trim).
   - Tambahkan contoh input di UI dan tooltip.
6. Tambahkan unit tests untuk fungsi penting (`filter_jobs`, TF-IDF pipeline, KNN predict).
7. Logging dan error handling lebih baik (log file, level: INFO/ERROR).

---

## Saran File Tambahan yang Berguna
- `requirements.txt` (lebih lengkap) — pastikan dependency versi konsisten.
- `CONTRIBUTING.md` — cara kontribusi, format data yang diharapkan, standar code style.
- `models/` — tempat menaruh model joblib dan vectorizer.
- `notebooks/` — notebook eksplorasi data dan training model (untuk reproducibility).
- `examples/` — contoh input & output dan contoh penggunaan.

---

## Checklist Deploy / Produksi
- Simpan semua artefak model dan vectorizer di folder `models/`.
- Gunakan environment reproducible (lockfile/requirements).
- Siapkan proses update model: script training terpisah (tidak men-gabungkan UI).
- Atur backup dataset dan model versione.
- Jalankan servable via Streamlit di server dengan supervisord / systemd atau containerize (Docker).

---

---

## Penutup
Dokumentasi ini merangkum cara kerja dan struktur proyek Job-Recommendation-KNN sehingga mudah dipahami dan dikembangkan lebih lanjut.  
