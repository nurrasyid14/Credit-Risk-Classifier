# Credit Risk Classifier

Sistem Penilaian Risiko Kredit berbasis Machine Learning menggunakan XGBoost dengan PyCaret.

## 📊 Model Performance

**Model Terbaik: XGBoost (Extreme Gradient Boosting)**
- **AUC: 0.9785** (sangat baik untuk imbalanced data)
- **Accuracy: 93.36%**
- **Recall: 80.30%** (mendeteksi 80% nasabah berisiko)
- **Precision: 88.75%** (88% prediksi risiko adalah benar)
- **F1-Score: 0.8431**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Via conda (Recommended untuk Windows)
conda install -c conda-forge pycaret -y

# Atau via pip
pip install pycaret streamlit plotly pandas
```

### 2. Run Application

```bash
streamlit run app.py
```

## 📁 Project Structure

```
Credit-Risk-Classifier/
│
├── data/
│   └── loan_data.csv              # Dataset (44,990 baris)
│
├── src/
│   ├── preprocessing.py           # Data cleaning & feature engineering
│   ├── modeling.py                # Random Forest (fallback model)
│   ├── rules.py                   # Business rules
│   ├── eda.ipynb                  # Exploratory Data Analysis
│   └── plots/                     # Visualisasi EDA
│
├── models/
│   └── best_pycaret_model.pkl     # XGBoost model (PyCaret)
│
├── plots_pycaret/                 # Evaluasi model
│   ├── pycaret_confusion_matrix.png
│   ├── pycaret_feature_importance.png
│   └── pycaret_roc_auc.png
│
├── app.py                         # Streamlit web application
└── isengisengPyCaret.ipynb        # Notebook eksperimen AutoML
```

## 🔧 How It Works

### Model Selection (app.py)

Aplikasi secara otomatis memilih model terbaik yang tersedia:

1. **Primary**: XGBoost dari PyCaret (`models/best_pycaret_model.pkl`)
   - Jika tersedia, akan digunakan (performa tertinggi)
   - Preprocessing sudah built-in dalam pipeline PyCaret
   
2. **Fallback**: Random Forest manual
   - Dilatih on-the-fly jika PyCaret tidak tersedia
   - Menggunakan `src/preprocessing.py` dan `src/modeling.py`

### Decision Logic

- **APPROVED**: PD < 15% (Risiko rendah)
- **CONDITIONAL APPROVAL**: 15% ≤ PD < 30% (Risiko sedang)
- **REJECTED**: PD ≥ 30% (Risiko tinggi)

## 📈 Model Comparison

| Model | AUC | Accuracy | Recall | Precision |
|-------|-----|----------|--------|-----------|
| **XGBoost (PyCaret)** | **0.9785** | **93.36%** | **80.30%** | **88.75%** |
| CatBoost | 0.9780 | 93.32% | 79.12% | 89.63% |
| LightGBM | 0.9777 | 93.22% | 78.79% | 89.45% |
| Random Forest (Manual) | ~0.92 | ~91% | ~75% | ~85% |

## 🧪 Re-train Model

Untuk melatih ulang model dengan dataset baru:

```bash
# Jalankan notebook PyCaret
jupyter notebook isengisengPyCaret.ipynb
```

Model baru akan tersimpan di `models/best_pycaret_model.pkl`

## 📊 Features Used

Dataset menggunakan 13 fitur prediksi:
- `person_age`: Umur peminjam
- `person_income`: Pendapatan tahunan
- `person_emp_length`: Lama bekerja
- `person_home_ownership`: Status kepemilikan rumah
- `loan_amnt`: Jumlah pinjaman
- `loan_intent`: Tujuan pinjaman
- `loan_int_rate`: Suku bunga
- `loan_percent_income`: Rasio pinjaman terhadap pendapatan (DTI)
- `cb_person_cred_hist_length`: Durasi histori kredit
- `credit_score`: Skor kredit
- `cb_person_default_on_file`: Riwayat gagal bayar

## ⚠️ Troubleshooting

### PyCaret tidak terinstall

Jika aplikasi menampilkan "Menggunakan Random Forest manual", artinya PyCaret belum terinstall:

```bash
conda install -c conda-forge pycaret -y
```

### Error saat load model

Pastikan file `models/best_pycaret_model.pkl` ada dan PyCaret terinstall dengan versi yang sama.

## 👨‍💻 Development

Dibuat untuk tugas Bu Alfi - Semester 4

---

**Model Powered by**: PyCaret AutoML + XGBoost  
**Web Framework**: Streamlit  
**Visualization**: Plotly
