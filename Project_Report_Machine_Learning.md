
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
```
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data['Experience_Level'])
plt.title("Distribusi Tingkat Pengalaman")
plt.xlabel("Experience Level")
plt.ylabel("Jumlah Anggota")
plt.show()
```
![Distribusi Target](https://github.com/arubhasy/gym_member_clf/blob/main/Proportion%20of%20Experience%20Levels.png)
---

## **4. Data Preparation**

### **Langkah-Langkah**
1. **Encoding**:
   Variabel kategorikal seperti `Gender` diubah menjadi numerik menggunakan Label Encoding.
2. **Normalisasi**:
   Data numerik dinormalisasi menggunakan **StandardScaler**.
3. **Splitting**:
   Data dibagi menjadi **80% training** dan **20% testing**.
```
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Encoding
encoder = LabelEncoder()
data['Gender'] = encoder.fit_transform(data['Gender'])

# Normalisasi
scaler = StandardScaler()
X = scaler.fit_transform(data.drop('Experience_Level', axis=1))
y = data['Experience_Level']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```
---

## **5. Modeling**

### **Model yang Digunakan**
1. **Decision Tree Classifier**:
   - Parameter: Default.
   - Kelebihan: Interpretasi mudah.
   - Kekurangan: Rentan overfitting.
   - Kode untuk Decision Tree Classifier:
```
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
# Train model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Predict
y_pred_dt = dt_model.predict(X_test)

# Evaluate
print("Decision Tree Classifier:")
print(classification_report(y_test, y_pred_dt))
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
```
2. **Gradient Boosting Classifier**:
   - Parameter: Default.
   - Kelebihan: Kinerja tinggi pada data beragam.
   - Kekurangan: Waktu komputasi lebih lama.
   - Kode untuk Gradient Boosting Classifier:
```
from sklearn.ensemble import GradientBoostingClassifier

# Train model
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# Predict
y_pred_gb = gb_model.predict(X_test)

# Evaluate
print("Gradient Boosting Classifier:")
print(classification_report(y_test, y_pred_gb))
print("Accuracy:", accuracy_score(y_test, y_pred_gb))
```
3. **Neural Network (MLP)**:
   - Parameter: Hidden layers (64, 32), max_iter=500.
   - Kelebihan: Menangkap pola kompleks.
   - Kekurangan: Memerlukan tuning untuk hasil optimal.
   - Kode untuk Neural Network (MLP):
```
from sklearn.neural_network import MLPClassifier

# Train model
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
mlp_model.fit(X_train, y_train)

# Predict
y_pred_mlp = mlp_model.predict(X_test)

# Evaluate
print("Neural Network (MLP):")
print(classification_report(y_test, y_pred_mlp))
print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
```
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
