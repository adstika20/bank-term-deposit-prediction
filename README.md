# bank-term-deposit-prediction
Proyek machine learning untuk memprediksi nasabah yang berpotensi berlangganan produk “term deposit” menggunakan dataset Bank Marketing dari UCI.

<img src="https://github.com/adstika20/bank-term-deposit-prediction/blob/main/assets/image.png" width="950"/>


## 📊 Ringkasan Proyek

Dataset yang digunakan berasal dari **[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)**, yang berisi data kampanye pemasaran langsung dari sebuah institusi perbankan di Portugal.  
Tujuan utama proyek ini adalah untuk **memprediksi kemungkinan nasabah akan berlangganan produk term deposit** (`yes` atau `no`) berdasarkan atribut demografis, sosial, dan data kampanye.

**Permasalahan Utama:**
> Bagaimana cara memprediksi nasabah potensial yang kemungkinan besar akan berlangganan term deposit, sehingga kampanye pemasaran menjadi lebih efisien dan hemat biaya?

---

## 🎯 Tujuan Proyek

1. Melakukan *data preprocessing* dan *feature engineering* pada dataset Bank Marketing dari UCI.  
2. Membangun dan membandingkan empat model klasifikasi:
   - **K-Nearest Neighbors (KNN)**
   - **Support Vector Classifier (SVC)**
   - **Random Forest Classifier**
   - **Gradient Boosting Classifier**
3. Menentukan model terbaik berdasarkan metrik evaluasi.  
4. Menerapkan model terbaik ke dalam aplikasi interaktif di **Hugging Face Spaces**.

---

## 🧠 Alur Machine Learning Pipeline

### 1. Pemahaman Data
- Jumlah data: 45.211 baris  
- Jumlah fitur: 16 fitur input  
- Target: `y` (apakah nasabah berlangganan term deposit atau tidak)  
- Tidak terdapat missing values  

### 2. Pra-pemrosesan Data (*Data Preprocessing*)
- Encoding variabel kategorikal menggunakan `OneHotEncoder` dan `LabelEncoder`  
- Normalisasi fitur numerik dengan `StandardScaler`  
- Pembagian data: 80% untuk train, 20% untuk test  

### 3. Pelatihan Model (*Model Training*)
Model yang digunakan:
- `KNeighborsClassifier`
- `SVC`
- `RandomForestClassifier`
- `GradientBoostingClassifier`
