
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

![image](https://github.com/user-attachments/assets/14168e67-ad4a-414c-b71d-a256c346057e)

Dari histogram, terlihat beberapa variabel yg memiliki distribusi normal dan miring (skew).

Terlihat juga ada 2 variabel numerik yang lebih cocok bila dikonversi menjadi tipe data kategorik, yaitu `Workout_Frequency` dan `Experience_Level`.

Berikut visualisasi data numerik setelah konversi tipe data:

![image](https://github.com/user-attachments/assets/71ae5025-8626-4e16-ba65-b0e580d056f0)


Selanjutnya kita tampilkan visualisasi `BMI` berdasarkan `Gender`:

![image](https://github.com/user-attachments/assets/e4623f99-8bb2-4f18-9347-449b4fde8f84)

Terlihat dari visualisasi boxplot, member laki-laki memiliki kecenderungan overweight karena median BMI > 25. Sedangkan member perempuang memiliki kecenderungan BMI normal. (Referensi: [Calculate Your Body Mass Index](https://www.nhlbi.nih.gov/health/educational/lose_wt/BMI/))

Selanjutnya kita tampilkan visualisasi variabel `Water_Intake` dan `Gender`:

![image](https://github.com/user-attachments/assets/1aec58a1-3c3b-4ff0-9cae-873fa50c6450)

Berdasarkan visualisasi, terlihat bahwa member laki-laki lebih banyak konsumsi air dibandingkan member perempuan.

Berikutnya kita tampilkan visualisasi `Session_Duration` dan `Calories_Burned`:

![image](https://github.com/user-attachments/assets/f594c6cc-10f0-4e96-b0b0-799777f444ea)

Berdasarkan visualisasi dapat dilihat bahwa semakin lama durasi latihan, maka akan semakin besar kalori yang terbakar.

Berikutnya adalah visualisasi variabel dengan tipe data objek dan kategorik:

![image](https://github.com/user-attachments/assets/645c41e6-dac4-45d1-b6bb-c2ca1f7990c9)

Kita telaah lebih lanjut pada variabel dg tipe data objek dan kategorik.

Selanjutnya kita akan telaah tipe latihan yang populer di kalangan member laki-laki dan perempuan.

![image](https://github.com/user-attachments/assets/cf5d6174-03fc-4e2e-8910-d476e2e7b836)

Berdasarkan visualisasi pie chart, dapat dilihat bahwa jenis latihan yang **paling populer** bagi member:
- Laki-laki: `Cardio`.
- Perempuan: `Strength`.

Sedangkan jenis latihan yang **paling tidak populer** bagi member:
- Laki-laki: `Yoga`.
- Perempuan: `HIIT`.

Berikutnya kita telaah `Workout_Frequency` berdasarkan `Experience_Level`:

![image](https://github.com/user-attachments/assets/35a797b7-99f8-4c36-9185-5ef82fd1e436)

Dapat dilihat bahwa semakin tinggi frekuensi latihan member, menunjukkan bahwa member tersebut semakin berpengalaman dalam latihan di gym.

### **Kondisi Data**

**Missing & Duplicate Value**
Setelah pengecekan, tidak ditemukan nilai hilang dan duplikasi. 

**Data Anomali**
Pada variabel dg tipe data objek dan kategorik, tidak ditemukan anomali.

**Outlier**
Pada variabel dg tipe data numerik, ditemukan outlier melalui visualisasi box plot.

Box Plot Variabel `Weights`:

![image](https://github.com/user-attachments/assets/7762f4df-9208-479f-9c5f-ddda42bd82cb)

Box Plot Variabel `Calories_Burned`:

![image](https://github.com/user-attachments/assets/56ce8984-a7b0-44bd-a195-6c935eb33f1f)

Box Plot Variabel `BMI`:

![image](https://github.com/user-attachments/assets/67eb1e27-a2d4-4edb-ab65-a2791dfeecb4)

Persentase outlier pada setiap variabel adalah sbb:
- Outlier pada variabel `Weights` = 0.92%
- Outlier pada variabel `Calories_Burned` = 1.03%
- Outlier pada variabel `BMI` = 2.57%

### **Analisis Korelasi**

Korelasi mengukur hubungan linear antara dua variabel, baik hubungan yang positif maupun negatif. Namun sebelumnya, kita lakukan encoding terlebih dahulu dengan ubah tipe data objek dan kategorik menjadi numerik. Hal ini perlu dilakukan karena perhitungan matriks korelasi hanya dapat dilakukan terhadap data numerik.

Setelah seluruh variabel bertipe numerik, selanjutnya kita hitung matriks korelasinya dan divisualisasikan menggunakan heatmap:

![image](https://github.com/user-attachments/assets/3358b8d6-906b-4359-9466-19f4ffa66d00)

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

**Keunggulan dan Kelemahan**
Keunggulan Decision Tree
- Mudah Dipahami: Struktur pohon memungkinkan interpretasi yang intuitif.
- Non-linear Relationship: Dapat menangkap hubungan non-linear antara fitur dan target.
- Tidak Membutuhkan Scaling: Fitur tidak perlu dinormalisasi atau di-standardisasi.

Kelemahan Decision Tree
- Overfitting: Pohon yang terlalu dalam dapat mempelajari noise dari data.
- Kurang Stabil: Perubahan kecil pada data dapat menyebabkan perubahan besar pada struktur pohon.
- Bias Fitur: Decision Tree cenderung memilih fitur dengan banyak nilai unik sebagai pemisah pertama.

### Gradient Boosting Classifier
Gradient Boosting Classifier adalah algoritma machine learning berbasis ensemble yang digunakan untuk menyelesaikan masalah klasifikasi. Algoritma ini bekerja dengan membangun serangkaian weak learners (biasanya pohon keputusan dengan kedalaman rendah) secara bertahap dalam sebuah proses iteratif. Setiap model bertujuan memperbaiki kesalahan dari model sebelumnya.

**Konsep Dasar**
1. Ensemble Model:
   - Gradient Boosting menggabungkan beberapa model sederhana (weak learners) menjadi satu model kuat.
   - Biasanya, model dasar yang digunakan adalah Decision Tree dengan kedalaman rendah (stump).
2. Boosting:
   - Boosting adalah teknik ensemble yang melatih model secara berurutan.
   - Setiap model baru berfokus untuk memperbaiki kesalahan prediksi dari model sebelumnya.
3. Gradient Descent:
   - Gradient Boosting menggunakan gradient descent untuk mengoptimalkan fungsi loss (loss function).
   - Model bekerja dengan mengurangi selisih antara prediksi dan target dengan menyesuaikan bobot weak learners.

**Langkah Kerja**
1. Inisialisasi Model
   - Model pertama memprediksi nilai awal yang paling sederhana (misalnya rata-rata atau nilai default).
   - Residual error (selisih antara nilai aktual dan prediksi awal)   - 
2. Pembangunan Weak Learners
   - Pada setiap iterasi, pohon keputusan kecil (weak learner) dibangun untuk memprediksi residual error dari model sebelumnya.
   - Pohon ini berfungsi untuk menemukan pola kesalahan yang belum dipelajari.
3. Update Model
   - Model yang baru ditambahkan ke ensemble dengan bobot tertentu untuk meminimalkan loss function.
   - Formula pembaruan prediksi adalah:

![Update formula](https://doimages.nyc3.cdn.digitaloceanspaces.com/010AI-ML/content/images/2019/11/image-43.png)

   - Di mana:
   - Fm(x): Prediksi pada iterasi ke-m
   - Fm-1(x): Prediksi pada iterasi sebelumnya
   - v: Learning rate (untuk mengontrol besarnya langkah update)

4. Iterasi Hingga Konvergensi
   - Proses diulangi beberapa kali dengan menambahkan weak learners hingga model mencapai konvergensi atau jumlah iterasi maksimum tercapai.
   - Residual error berkurang di setiap iterasi sehingga prediksi semakin mendekati target aktual.

**Keunggulan dan Kelemahan**
Keunggulan Gradient Boosting
- Performa Tinggi: Sangat efektif untuk data yang kompleks.
- Fleksibilitas: Mendukung berbagai fungsi loss (klasifikasi, regresi).
- Kontrol Overfitting: Dengan parameter seperti learning rate dan max depth.

Kelemahan Gradient Boosting
- Lambat: Pelatihan bisa memakan waktu karena proses iteratif.
- Sensitif terhadap Outlier: Model bisa dipengaruhi oleh data outlier.
- Tuning Parameter: Membutuhkan tuning parameter agar performanya optimal.

### Neural Network (MLP)
MLP Classifier (Multi-Layer Perceptron) adalah jenis neural network yang digunakan untuk klasifikasi dalam machine learning. MLP bekerja dengan cara menyebarkan input melalui lapisan-lapisan jaringan saraf (neural network) untuk menghasilkan prediksi. MLP termasuk dalam algoritma supervised learning, yang mempelajari hubungan antara input dan output berdasarkan data latih.

**Konsep Dasar**
MLP memiliki tiga komponen utama:
- Input Layer: Menerima fitur dari dataset.
- Hidden Layers: Lapisan tersembunyi di mana komputasi non-linear dilakukan melalui aktivasi neuron.
- Output Layer: Menghasilkan prediksi dalam bentuk probabilitas atau kelas target.

**Langkah Kerja**
1. Forward Propagation
   - Forward propagation adalah proses menyebarkan input dari lapisan input ke lapisan output melalui lapisan tersembunyi.
2. Fungsi Aktivasi
   - Fungsi aktivasi menambahkan non-linearitas ke jaringan, sehingga jaringan mampu menangkap pola yang kompleks.
3. Loss Function
   - Loss function mengukur perbedaan antara prediksi model dan nilai aktual. Untuk klasifikasi, loss function yang umum digunakan adalah Cross-Entropy Loss.
4. Backpropagation
   - Backpropagation adalah proses menghitung gradien dari loss function terhadap bobot jaringan saraf menggunakan aturan rantai (chain rule).
   - Gradien ini digunakan untuk memperbarui bobot dan bias agar error menjadi minimal.
5. Training dan Konvergensi
   - Proses forward propagation dan backpropagation diulang selama beberapa iterasi (epochs).
   - Model dilatih hingga fungsi loss konvergen (menjadi minimal).

**Keunggulan dan Kelemahan**
Kelebihan MLP Classifier
- Dapat Menangkap Pola Non-Linear: MLP sangat baik dalam memodelkan hubungan kompleks.
- Fleksibilitas: Dapat digunakan untuk klasifikasi multi-kelas dan regresi.
- Kinerja Tinggi: Dengan tuning parameter yang tepat, MLP dapat mencapai hasil yang baik.

Kelemahan MLP Classifier
- Waktu Komputasi Lama: Jaringan saraf memerlukan banyak iterasi untuk konvergen.
- Memerlukan Normalisasi Data: MLP sensitif terhadap skala fitur.
- Tuning Parameter: Perlu hyperparameter tuning seperti jumlah hidden layers dan learning rate untuk performa optimal.

### **Ringkasan Parameter yang Digunakan**
Berikut adalah penjelasan parameter yang digunakan pada ketiga model:

| Model                    | Parameter                     | Deskripsi                                                           |
|--------------------------|-------------------------------|---------------------------------------------------------------------|
| Decision Tree Classifier | `criterion='gini'`            | Menggunakan Gini Index untuk pemisahan data.                        |
|                          | `max_depth=3`                 | Membatasi kedalaman pohon menjadi 3 untuk mencegah overfitting.     |
|                          | `random_state=42`             | Menjaga konsistensi hasil eksperimen.                               |
| Gradient Boosting        | `n_estimators=100`            | Jumlah pohon dalam ensemble.                                        |
|                          | `learning_rate=0.1`           | Mengontrol kontribusi masing-masing pohon terhadap model akhir.     |
|                          | `max_depth=3`                 | Kedalaman maksimum setiap pohon.                                    |
|                          | `random_state=42`             | Menjaga konsistensi hasil eksperimen.                               |
| Neural Network (MLP)     | `hidden_layer_sizes=(64, 32)` | Dua lapisan tersembunyi dengan 64 dan 32 neuron masing-masing.      |
|                          | `activation='relu'`           | Fungsi aktivasi untuk menangkap hubungan non-linear.                |
|                          | `max_iter=500`                | Jumlah iterasi maksimum untuk pelatihan model.                      |
|                          | `max_iter=500`                | Jumlah iterasi maksimum untuk pelatihan model.                      |
|                          | `random_state=42`             | Menjaga konsistensi hasil eksperimen.                               |

---

## **6. Evaluation**
Pada tahap ini akan dilakukan evaluasi terhadap performa dari model Decision Tree Classifier, Gradient Boosting, dan MLP menggunakan:
- Confusion Matrix
- Nilai accuaracy dan macro average F1-Score
- Visualisasi accuaracy dan macro average F1-Score
- Feature Importance

### **Confusion Matrix**

Berikut adalah perbandingan visualisasi confusion matrix model Decision Tree Classifier, Gradient Boosting, dan MLP:

![image](https://github.com/user-attachments/assets/9611a5ac-d9f1-4b4c-9aea-750fef50cea1)

![image](https://github.com/user-attachments/assets/cbab6e60-bee6-4553-a6d0-dd6fde00de90)

![image](https://github.com/user-attachments/assets/1eb33abb-ebcd-4a5f-8bfc-e9eb864e146d)

Berdasarkan perbandingan ketiga confusion matriks, dapat dilihat bahwa:
- DT Classifier melakukan prediksi secara benar sebanyak 149 dan salah prediksi sebanyak 41.
- GB Classifier melakukan prediksi secara benar sebanyak 171 dan salah prediksi sebanyak 19.
- MLP Classifier melakukan prediksi secara benar sebanyak 161 dan salah prediksi sebanyak 29.

Berdasarkan perbandingan confusion matrix, dapat disimpulkan bahwa model Gradient Boosting Classifier berhasil melakukan perdiksi secara benar lebih banyak dibandingkan kedua model lainnya dan model juga memiliki kesalahan prediksi yang paling sedikit.

### **Nilai accuaracy dan macro average F1-Score**
Berikut adalah hasil evaluasi ketiga model:

| Model                    | Accuracy | Macro Avg F1-Score |
|--------------------------|----------|--------------------|
| Decision Tree Classifier | 0.78     | 0.82               |
| Gradient Boosting        | **0.90** | **0.92**           |
| Neural Network (MLP)     | 0.85     | 0.87               |

Berdasarkan perhitungan akurasi dan macro average (F1-score), dapat dilihat Gradient Boosting memiliki performa terbaik dibandingkan kedua model lainnya.

### **Visualisasi accuaracy dan macro average F1-Score**

Berikut adalah visualisasi perbandingan accuracy dan macro average F1-Score model Decision Tree Classifier, Gradient Boosting, dan MLP:

![image](https://github.com/user-attachments/assets/f20b32ba-65a9-432c-9f0c-49b734aaf927)

Secara visual dapat dilihat bahwa performa Gradient Boosting mengungguli performa kedua model lainnya.

### **Feature Importance**
Feature importance membantu dalam memahami fitur yang memiliki pengaruh besar terhadap model.

Berikut Feature Importance model Decision Tree Classifier:

![image](https://github.com/user-attachments/assets/43de2143-b7bd-4e82-8f56-1e6d76b9e754)

Pada model DT Classifier, top 3 fitur yang memiliki pengaruh besar:
- `Workout_Type`
- `Water_Intake`
- `Resting_BPM`

Berikut Feature Importance model Gradient Boosting Classifier:

![image](https://github.com/user-attachments/assets/a5b55c14-b23a-4548-b5bb-f53aa4d367c0)

Pada model GB Classifier, top 3 fitur yang memiliki pengaruh besar:
- `Water_Intake`
- `Resting_BPM`
- `Workout_Type`

Neural Network (seperti MLP) tidak secara langsung memberikan metrik pentingnya fitur (feature importance) karena struktur jaringannya berbasis lapisan tersembunyi (hidden layers) dengan bobot dan bias yang terdistribusi di antara banyak unit

---

## **7. Kesimpulan**

### **Perbandingan Kinerja Model**

**1. Decision Tree Classifier**
   - Accuracy: 0.78
   - Macro Avg F1-Score: 0.82
   - Kelebihan:
      - Kinerja sangat baik pada kelas minoritas (kelas 2) dengan F1-Score 1.00.
   - Kekurangan:
      - Sedikit lebih lemah pada kelas mayoritas (kelas 0 dan 1) dibanding model lain.

**2. Gradient Boosting Classifier**
   - Accuracy: 0.90
   - Macro Avg F1-Score: 0.92
   - Kelebihan:
      - Kinerja terbaik secara keseluruhan pada kelas mayoritas (kelas 0 dan 1)
      - Konsistensi tinggi pada semua metrik.
   - Kekurangan:
   - Tidak ada kelemahan yang mencolok.

**3. Neural Network (MLP)**
   - Accuracy: 0.85
   - Macro Avg F1-Score: 0.87
   - Kelebihan:
      - Performanya mendekati Gradient Boosting, dengan hasil yang sangat baik pada semua kelas.
      - Lebih sederhana untuk memproses pola non-linear dalam dataset.
   - Kekurangan:
      - Kinerja keseluruhan sedikit lebih rendah dibanding Gradient Boosting.

### **Kesimpulan**
Model terbaik adalah Gradient Boosting Classifier karena:
- Memiliki akurasi tertinggi (0.90).
- Memiliki macro average F1-Score tertinggi (0.92).
- Konsistensi kinerjanya tinggi di semua kelas.

### **Rekomendasi**
- Gradient Boosting Classifier dapat digunakan untuk memprediksi tingkat pengalaman anggota gym.
- Untuk performa lebih tinggi, pertimbangkan **tuning hyperparameter** atau eksplorasi algoritma lain seperti XGBoost.
