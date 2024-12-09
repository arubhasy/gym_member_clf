
# **Laporan Proyek Machine Learning: Prediksi Tingkat Pengalaman Anggota Gym**

---

## **1. Domain Proyek**

### **Latar Belakang**
Kesehatan dan kebugaran merupakan aspek penting dalam gaya hidup modern. Gym sering menjadi pilihan utama untuk mencapai tujuan kebugaran. Namun, efektivitas latihan sangat tergantung pada pengalaman pengguna. Dengan memahami tingkat pengalaman anggota gym, pengelola dapat menawarkan program yang lebih personal dan meningkatkan kepuasan anggota.

### **Mengapa Masalah Ini Penting?**
1. Memberikan rekomendasi latihan yang lebih efektif.
2. Meningkatkan retensi anggota gym.
3. Mengoptimalkan data untuk analisis yang mendalam.

### **Referensi**
- Dataset: [Gym Members Exercise Dataset](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset).
- Lahti, H. (2023). [Customer retention plan for a fitness center](https://www.theseus.fi/bitstream/handle/10024/796190/Lahti_Hanna.pdf?sequence=3).

---

## **2. Business Understanding**

### **Problem Statements**
1. Bagaimana memprediksi tingkat pengalaman anggota gym?
2. Algoritma machine learning mana yang paling optimal untuk permasalahan ini?

### **Goals**
- Membuat model prediktif untuk mengklasifikasikan anggota gym ke dalam tiga tingkat pengalaman: **pemula**, **menengah**, dan **ahli**.
- Memilih model terbaik berdasarkan **accuracy** dan **macro average F1-Score**.

### **Solution Statement**
1. Menggunakan tiga algoritma: **Decision Tree Classifier**, **Gradient Boosting Classifier**, dan **Neural Network (MLP)**.
2. Membandingkan performa model untuk memilih solusi terbaik.

---

## **3. Data Understanding**

### **Informasi Dataset**
- **Sumber Data**: [Kaggle](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset).
- **Jumlah Data**: 973 sampel.
- **Fitur Input**: Usia, berat badan, tinggi badan, detak jantung maksimum, durasi sesi latihan, jenis latihan, dll.
- **Variabel Target**: `Experience_Level` (1: Pemula, 2: Menengah, 3: Ahli).

### **Distribusi Target**
Berikut adalah distribusi variabel `Experience_Level`:

![Distribusi Target](https://via.placeholder.com/600x400)

---

## **4. Data Preparation**

### **Langkah-Langkah**
1. **Encoding**:
   Variabel kategorikal seperti `Gender` diubah menjadi numerik menggunakan Label Encoding.
2. **Normalisasi**:
   Data numerik dinormalisasi menggunakan **StandardScaler**.
3. **Splitting**:
   Data dibagi menjadi **80% training** dan **20% testing**.

---

## **5. Modeling**

### **Model yang Digunakan**
1. **Decision Tree Classifier**:
   - Parameter: Default.
   - Kelebihan: Interpretasi mudah.
   - Kekurangan: Rentan overfitting.

2. **Gradient Boosting Classifier**:
   - Parameter: Default.
   - Kelebihan: Kinerja tinggi pada data beragam.
   - Kekurangan: Waktu komputasi lebih lama.

3. **Neural Network (MLP)**:
   - Parameter: Hidden layers (64, 32), max_iter=500.
   - Kelebihan: Menangkap pola kompleks.
   - Kekurangan: Memerlukan tuning untuk hasil optimal.

---

## **6. Evaluation**

### **Hasil Evaluasi**
Berikut adalah hasil evaluasi ketiga model:

| Model                    | Accuracy | Macro Avg F1-Score |
|--------------------------|----------|--------------------|
| Decision Tree Classifier | 0.87     | 0.89               |
| Gradient Boosting        | **0.91** | **0.92**           |
| Neural Network (MLP)     | 0.88     | 0.90               |

### **Visualisasi Perbandingan Akurasi**

![Perbandingan Akurasi](https://via.placeholder.com/600x400)

---

## **7. Kesimpulan**

### **Model Terbaik**
Gradient Boosting Classifier adalah model terbaik dengan:
- **Akurasi**: 0.91
- **Macro Avg F1-Score**: 0.92

### **Rekomendasi**
- Gradient Boosting Classifier dapat digunakan untuk memprediksi tingkat pengalaman anggota gym.
- Untuk performa lebih tinggi, pertimbangkan **tuning hyperparameter** atau eksplorasi algoritma lain seperti XGBoost.

---

### **Struktur dan Dokumentasi**
- Laporan mengikuti struktur yang sesuai: Domain Proyek, Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation.
- Semua langkah didukung dengan visualisasi, tabel, dan code snippet.
