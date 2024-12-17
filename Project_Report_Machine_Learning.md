
# **Laporan Proyek Machine Learning: Prediksi Tingkat Pengalaman Anggota Gym**

---

## **1. Domain Proyek**

### **Latar Belakang**
Kesehatan dan kebugaran merupakan aspek penting dalam kehidupan. Bagi masyarakat yang tinggal di perkotaan, gym sering menjadi pilihan utama untuk mencapai tujuan kebugaran. Namun, efektivitas latihan sangat tergantung pada pengalaman pengguna. Dengan memahami tingkat pengalaman anggota gym, pengelola dapat menawarkan program yang lebih personal dan meningkatkan kepuasan anggota gym.

### **Mengapa Masalah Ini Penting?**
1. Memberikan rekomendasi latihan yang lebih efektif.
2. Meningkatkan retensi anggota gym.
3. Mengoptimalkan data untuk analisis yang mendalam.

### **Referensi**
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
- **Dataset**: [Gym Members Exercise Dataset (Kaggle)](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset).
- **Jumlah Data**:
   - Jumlah Baris: 973
   - Jumlah Kolom: 15
- **Fitur Input**:
   - `Age`: usia gym member.
   - `Gender`: jenis kelamin member (`Male`, `Female`).
   - `Weight (kg)`: berat badan member dalam kilogram.
   - `Height (m)`: tinggi badan member dalam kilogram.
   - `Max_BPM`: denyut jantung maksimum selama latihan (beats per minute).
   - `Avg_BPM`: denyut jantung rata-rata selama latihan (beats per minute).
   - `Resting_BPM`: denyut jantung saat istirahat sebelum istirahat.
   - `Session_Duration (hours)`: durasi setiap sesi latihan dalam jam.
   - `Calories_Burned`: total kalori yang terbakar selama setiap sesi.
   - `Workout_Type`: jenis latihan yang dilakukan (`Cardio`, `Strength`, `Yoga`, `HIIT`).
   - `Fat_Percentage`: persentase lemak tubuh anggota.
   - `Water_Intake (liters)`: asupan air harian selama latihan.
   - `Workout_Frequency (days/week)`: jumlah sesi latihan per minggu.
   - `BMI`: Body Mass Index, dihitung dari tinggi dan berat badan.
- **Variabel Target**:
   - `Experience_Level`: Tingkat pengalaman member (`1`: Pemula, `2`: Menengah, `3`: Ahli).
- **Tipe Data**:
   - Numerik: 13
   - Objek: 2

### **Telaah Data**

Untuk lebih memahami distribusi data numerik, berikut visualisasi data numeriknya:
![Visualisasi Data Numerik](https://github.com/arubhasy/gym_member_clf/blob/main/Distribusi%20Data%20Numerik.png)

Dari histogram, terlihat beberapa variabel yg memiliki distribusi normal dan miring (skew).

Terlihat juga ada 2 variabel numerik yang lebih cocok bila dikonversi menjadi tipe data kategorik, yaitu `Workout_Frequency` dan `Experience_Level`.

Berikut visualisasi data numerik setelah konversi tipe data:
![Visualisasi Data Numerik Setelah Konversi](https://github.com/arubhasy/gym_member_clf/blob/main/Distribusi%20Data%20Numerik%20Setelah%20Konversi.png)

Selanjutnya kita tampilkan visualisasi `BMI` berdasarkan `Gender`:
![Visualisasi BMI Berdasarkan Gender](https://github.com/arubhasy/gym_member_clf/blob/main/Boxplot%20BMI%20Berdasarkan%20Gender.png)

Terlihat dari visualisasi boxplot, member laki-laki memiliki kecenderungan overweight karena median BMI > 25. Sedangkan member perempuang memiliki kecenderungan BMI normal. (Referensi: [Calculate Your Body Mass Index](https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/))

Selanjutnya kita tampilkan visualisasi variabel `Water_Intake` dan `Gender`:
![Visualisasi Water Intake Berdasarkan Gender](https://github.com/arubhasy/gym_member_clf/blob/main/Boxplot%20BMI%20Berdasarkan%20Gender.png)

Berdasarkan visualisasi, terlihat bahwa member laki-laki lebih banyak konsumsi air dibandingkan member perempuan.

Berikutnya kita tampilkan visualisasi `Session_Duration` dan `Calories_Burned`:
![Visualisasi Session_Duration dan Calories_Burned](https://github.com/arubhasy/gym_member_clf/blob/main/Scatterplot%20Session%20Duration%20dan%20Calories%20Burned.png)

Berdasarkan visualisasi dapat dilihat bahwa semakin lama durasi latihan, maka akan semakin besar kalori yang terbakar.

Berikutnya adalah visualisasi variabel dengan tipe data objek dan kategorik:
![Visualisasi Variabel dengan Tipe Data Objek dan Kategorik](https://github.com/arubhasy/gym_member_clf/blob/main/Visualisasi%20Data%20Kategorik%20dan%20Objek.png)

Kita telaah lebih lanjut pada variabel dg tipe data objek dan kategorik.

Selanjutnya kita akan telaah tipe latihan yang populer di kalangan member laki-laki dan perempuan.
![Pie Chart Tipe Latihan](https://github.com/arubhasy/gym_member_clf/blob/main/Pie%20Chart%20Jenis%20Latihan%20Berdasarkan%20Gender.png)

Berdasarkan visualisasi pie chart, dapat dilihat bahwa jenis latihan yang **paling populer** bagi member:
- Laki-laki: `Cardio`.
- Perempuan: `Strength`.

Sedangkan jenis latihan yang **paling tidak populer** bagi member:
- Laki-laki: `Yoga`.
- Perempuan: `HIIT`.

Berikutnya kita telaah `Workout_Frequency` berdasarkan `Experience_Level`:
![Barplot Frekuensi Latihan Berdasarkan Pengalaman](https://github.com/arubhasy/gym_member_clf/blob/main/Barplot%20Frekuensi%20Latihan%20dan%20Pengalaman.png)

Dapat dilihat bahwa semakin tinggi frekuensi latihan member, menunjukkan bahwa member tersebut semakin berpengalaman dalam latihan di gym.

### **Kondisi Data**

**Missing & Duplicate Value**
Setelah pengecekan, tidak ditemukan nilai hilang dan duplikasi. 

**Data Anomali**
Pada variabel dg tipe data objek dan kategorik, tidak ditemukan anomali.

**Outlier**
Pada variabel dg tipe data numerik, ditemukan outlier melalui visualisasi box plot.

Box Plot Variabel `Weights`:

![Boxplot Weights](https://github.com/arubhasy/gym_member_clf/blob/main/Boxplot%20Weight.png)

Box Plot Variabel `Calories_Burned`:

![Boxplot Calories](https://github.com/arubhasy/gym_member_clf/blob/main/Boxplot%20Calories.png)

Box Plot Variabel `BMI`:

![Boxplot BMI](https://github.com/arubhasy/gym_member_clf/blob/main/Boxplot%20BMI.png)

Persentase outlier pada setiap variabel adalah sbb:
- Outlier pada variabel `Weights` = 0.92%
- Outlier pada variabel `Calories_Burned` = 1.03%
- Outlier pada variabel `BMI` = 2.57%

### **Analisis Korelasi**

Korelasi mengukur hubungan linear antara dua variabel, baik hubungan yang positif maupun negatif. Namun sebelumnya, kita lakukan encoding terlebih dahulu dengan ubah tipe data objek dan kategorik menjadi numerik. Hal ini perlu dilakukan karena perhitungan matriks korelasi hanya dapat dilakukan terhadap data numerik.

Setelah seluruh variabel bertipe numerik, selanjutnya kita hitung matriks korelasinya dan divisualisasikan menggunakan heatmap:
![Matriks Korelasi](https://github.com/arubhasy/gym_member_clf/blob/main/Korelasi%20Antar%20Variabel%20Numerik.png)

Berikut terdapat beberapa insight yang menarik berdasarkan interpretasi matriks korelasi:
- `Weight` memiliki korelasi positif yang sangat kuat dengan `BMI` (0.83), artinya semakin tinggi berat badan akan semakin tinggi juga BMI.
- `Experience_Level` memiliki korelasi positif yang sangat kuat dengan `Workout_Frequency` (0.84), artinya semakin berpengalaman seseorang di gym akan semakin tinggi frekuensi latihannya.
- `Session_Duration` memiliki korelasi positif yang sangat kuat dengan `Calories_Burned` (0.91), artinya semakin lama durasi workout akan semakin banyak kalori yang terbakar.

Meskipun korelasi tinggi tidak selalu berarti terdapat hubungan sebab-akibat, tapi di sini kita dapat memperoleh insight yang cukup berharga dari data.

---

## **4. Data Preparation**

### **Langkah-Langkah**
1. **Handling Missing Values**
2. **Handling Duplicate Values**
3. **Handling Outlier**
4. **Encoding Variabel Kategorik**
5. **Normalisasi Data Numerik**
6. **Data Splitting**
7. **Finalisasi Data**

### Handling Missing Values
Pastikan dataset tidak memiliki missing values atau tidak ada nilai yang kosong (NaN). Berdasarkan hasil telaah data, tidak ditemukan missing values. Dengan demikian, penanganan missing values tidak dilakukan.

### Handling Duplicate Values
Pastikan dataset tidak memiliki duplicate values atau duplikasi entri data. Berdasarkan hasil telaah data, tidak ditemukan duplicate values. Dengan demikian, penanganan duplicate values tidak dilakukan.

### Handling Outliers
Berdasarkan telaah data, terdapat outlier pada variabel:
- Outlier pada variabel `Weights` = 0.92%
- Outlier pada variabel `Calories_Burned` = 1.03%
- Outlier pada variabel `BMI` = 2.57%

Karena outlier secara persentase cukup kecil, maka diputuskan untuk menghapus nilai outlier pada ketiga variabel tersebut. Setelah drop outlier, terdapat 948 entri data dari semula 973 entri. Artinya terdapat 25 entri yang didrop.

### Encoding Variabel Kategorik
Encoding bertujuan untuk mengubah variabel kategorikal menjadi format numerik agar dapat diproses oleh algoritma machine learning. Terdapat 2 variabel yang dikonversi menjadi numerik:
- `Gender`
- `Workout_Type`

Setelah proses encoding, kondisi data seluruhnya berubah menjadi numerik.

### Normalisasi Data Numerik
Setelah seluruh tipe data dikonversi menjadi numerik, selanjutnya dilakukan normalisasi agar semua fitur numerik memiliki skala yang seragam. Menggunakan **StandarScaler**, seluruh data diubah ke distribusi dengan rata-rata 0 dan deviasi standar 1. Hal ini dilakukan agar dapat mempercepat proses pelatihan model dan mencegah fitur dengan nilai besar mendominasi perhitungan.

### Data Splitting
Splitting bertujuan untuk membagi dataset menjadi data training dan testing untuk memastikan evaluasi yang adil. Setelah melakukan pemisahan antara fitur (X) dan variabel target (y), dataset dibagi menjadi dua bagian: data training (80%) dan data testing (20%). 

Data training digunakan untuk melatih model dan data testing digunakan untuk mengukur performa model terhadap data baru. Hal ini perlu dilakukan untuk memastikan model dapat dievaluasi pada data yang belum pernah dilihat sebelumnya untuk menghindari overfitting (model hanya bekerja baik pada data training tetapi buruk pada data baru).

### Finalisasi Data
Setelah semua langkah di atas, dataset sudah siap untuk digunakan dalam proses pemodelan:
- Data sudah bersih (tidak ada missing values, duplicate values, anomali dan outlier).
- Fitur numerik telah dinormalisasi.
- Variabel kategorikal telah di-encode.
- Data training dan testing telah dipisahkan.

---

## **5. Modeling**

### **Model yang Digunakan**
Pada project ini kita gunakan 3 model klasifikasi berikut:
1. Decision Tree Classifier
2. Gradient Boosting Classifier
3. Neural Network (MLP)

### Decision Tree Classifier
Decision Tree Classifier adalah algoritma machine learning berbasis pohon yang digunakan untuk menyelesaikan masalah klasifikasi dengan memisahkan dataset ke dalam subset berdasarkan aturan keputusan (decision rules). Model ini membangun struktur pohon dari atas ke bawah melalui proses pembagian fitur (splitting) yang optimal.

**Konsep Dasar**
1. Struktur Pohon:
   - Root Node: Node pertama yang mewakili fitur awal untuk membagi dataset.
   - Internal Nodes: Node yang berisi keputusan untuk memisahkan data lebih lanjut.
   - Leaf Nodes: Node akhir yang mewakili kelas atau label prediksi.
2. Pembagian Dataset (Splitting):
   - Proses memisahkan dataset berdasarkan fitur yang menghasilkan pemisahan terbaik (optimal split).
   - Algoritma mencari fitur yang meminimalkan ketidaksamaan data setelah pemisahan.
3. Aturan Keputusan (Decision Rules):
   - Setiap node internal menggunakan aturan berbasis fitur 

**Langkah Kerja**
1. Pemilihan Fitur Terbaik (Splitting Criterion)
   - Pada setiap node, algoritma memilih fitur yang paling baik memisahkan data berdasarkan **impurity** (ketidaksamaan data) atau **information gain**.
   - Beberapa metode untuk mengukur pemisahan optimal:
   - Gini Index:
     
     ![Gini Index](https://miro.medium.com/v2/resize:fit:640/format:webp/1*vRlwRFknvfgWLBed1vsGoQ.jpeg)
     Di mana pi adalah probabilitas sebuah data jatuh ke kelas i.
   - Entropy (Information Gain):
     
     ![Entropy](https://miro.medium.com/v2/resize:fit:640/format:webp/1*efLrD1ECWl-utII0KYb7tQ.jpeg)
     Entropy mengukur ketidakpastian pada node; semakin kecil entropy, semakin baik pemisahan.
2. Pembagian Dataset
   - Setelah fitur terbaik dipilih, dataset dipisahkan menjadi dua subset berdasarkan nilai threshold fitur tersebut.
   - Proses ini dilakukan secara rekursif di setiap cabang pohon.
3. Pembangunan Pohon Secara Rekursif
   - Algoritma melanjutkan pembagian hingga kondisi stop terpenuhi, yaitu:
      - Semua data dalam node memiliki label yang sama (pure node).
      - Tidak ada fitur yang tersisa untuk membagi data.
      - Kedalaman maksimum pohon telah tercapai (max_depth).
4. Prediksi
   - Untuk membuat prediksi, input data dilewatkan dari root node ke leaf node mengikuti aturan keputusan yang telah dibuat.
   - Leaf node memberikan label kelas sebagai prediksi akhir.

### Gradient Boosting Classifier
   - Parameter: Default.
   - Kelebihan: Kinerja tinggi pada data beragam.
   - Kekurangan: Waktu komputasi lebih lama.

### Neural Network (MLP)
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
```
# Plotting accuracy comparison
models = ['Decision Tree', 'Gradient Boosting', 'Neural Network']
accuracies = [
    accuracy_score(y_test, y_pred_dt),
    accuracy_score(y_test, y_pred_gb),
    accuracy_score(y_test, y_pred_mlp)
]

plt.bar(models, accuracies)
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")
plt.show()
```
![Perbandingan Akurasi](https://github.com/arubhasy/gym_member_clf/blob/a2a7c0161204e7525a6cfbd392f7e342ea0d896c/Model%20Accuracy.png)

---

## **7. Kesimpulan**

### **Model Terbaik**
Gradient Boosting Classifier adalah model terbaik dengan:
- **Akurasi**: 0.91
- **Macro Avg F1-Score**: 0.92

### **Rekomendasi**
- Gradient Boosting Classifier dapat digunakan untuk memprediksi tingkat pengalaman anggota gym.
- Untuk performa lebih tinggi, pertimbangkan **tuning hyperparameter** atau eksplorasi algoritma lain seperti XGBoost.
