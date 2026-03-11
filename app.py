"""
NexBank Credit Decision Engine
Arsitektur Clean Code: Dataset diload di UI, diproses oleh modul src/.
Visualisasi disesuaikan dengan standar Dashboard interaktif (Plotly Gauge Chart).
"""

import streamlit as st
import pandas as pd
import os
import time
import plotly.graph_objects as go

# Memanggil modul-modul dari folder src/
from .src.preprocessor import Preprocessor as dp
from .src.modeling import CreditRiskModel as crm
from .src.rules import hitung_kolektibilitas_ojk

st.set_page_config(page_title="Credit Risk Analysis System", page_icon="🏦", layout="wide")

# --- 1. TAHAP LOAD DATASET & TRAINING ---
@st.cache_resource
def load_and_train_system():
    file_path = "loan_data.csv"
    if not os.path.exists(file_path) and os.path.exists("data/loan_data.csv"):
        file_path = "data/loan_data.csv"
        
    try:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['loan_status']) 
    except FileNotFoundError:
        return None, None, "File loan_data.csv tidak ditemukan!"

    preprocessor = dp()
    X_processed, y = preprocessor.fit_transform(df)

    model = crm()
    model.train(X_processed, y)

    return preprocessor, model, "Sistem Siap!"

preprocessor, ml_model, status_msg = load_and_train_system()

# --- 2. TAHAP UI & INPUT DASHBOARD ---
st.title("🏦 Credit Risk Analysis System")
st.markdown("Sistem Penilaian Risiko Pinjaman Berbasis Machine Learning")
st.divider()

if preprocessor is None:
    st.error(status_msg)
    st.stop()

# Sidebar untuk Input agar tampilan utama fokus ke Hasil (seperti di gambar)
with st.sidebar:
    st.header("📝 Form Input Data")
    app_name = st.text_input("Nama Aplikan", value="Andi")
    age = st.number_input("Umur (Tahun)", min_value=18, max_value=100, value=25)
    income = st.number_input("Pendapatan Tahunan ($)", min_value=1000, value=50000, step=1000)
    loan_intent = st.selectbox("Tujuan Pinjaman", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_amount = st.number_input("Jumlah Pinjaman ($)", min_value=1000, value=20000, step=1000)
    loan_int_rate = st.number_input("Suku Bunga (%)", min_value=1.0, value=15.0, step=0.1)
    
    st.markdown("---")
    st.subheader("Data Tambahan")
    emp_length = st.number_input("Lama Bekerja (Tahun)", min_value=0, max_value=50, value=3)
    home_ownership = st.selectbox("Status Kepemilikan Rumah", ["RENT", "OWN", "MORTGAGE"])
    credit_score = st.number_input("Skor Kredit", min_value=300, max_value=850, value=650)
    hari_tunggakan = st.number_input("Riwayat Tunggakan (Hari)", min_value=0, value=0)
    durasi_kredit = st.number_input("Durasi Histori Kredit (Tahun)", min_value=0, value=4)
    
    analyze_btn = st.button("🚀 Jalankan Analisis", type="primary", use_container_width=True)

# --- 3. TAHAP DEPLOYMENT / PREDIKSI ---
if analyze_btn:
    with st.spinner("Memproses Analisis Risiko..."):
        time.sleep(0.8)

        dti_ratio = loan_amount / income if income > 0 else 1.0
        
        input_raw = pd.DataFrame({
            'person_age': [age], 
            'person_gender': ['male'], 
            'person_education': ['Bachelor'],
            'person_income': [income], 
            'person_emp_exp': [emp_length], 
            'person_home_ownership': [home_ownership],
            'loan_amnt': [loan_amount], 
            'loan_intent': [loan_intent], 
            'loan_int_rate': [loan_int_rate], 
            'loan_percent_income': [dti_ratio], 
            'cb_person_cred_hist_length': [durasi_kredit],
            'credit_score': [credit_score], 
            'previous_loan_defaults_on_file': ['Yes' if hari_tunggakan > 0 else 'No']
        })

        input_processed = preprocessor.transform(input_raw)
        pd_value = ml_model.predict_default_prob(input_processed)
        pd_value = max(0.01, min(pd_value, 0.99)) 
        pd_percent = pd_value * 100

        # Menghitung Expected Loss (Asumsi LGD 45%)
        lgd_rate = 0.45
        expected_loss = loan_amount * pd_value * lgd_rate

        # Logika Keputusan berdasarkan Ambang Batas (Threshold 15% seperti di gambar)
        if pd_percent < 15.0:
            decision = "APPROVED"
            decision_color = "success" # Hijau
        elif pd_percent < 30.0:
            decision = "CONDITIONAL APPROVAL"
            decision_color = "warning" # Kuning
        else:
            decision = "REJECTED"
            decision_color = "error" # Merah

        # --- 4. TATA LETAK HASIL (Sesuai Referensi Gambar) ---
        
        # A. Kotak Detail Aplikan (Desain Gelap)
        st.markdown("### Detail Aplikan")
        st.markdown(f"""
        <div style='background-color: #2c3e50; padding: 20px; border-radius: 8px; color: white; margin-bottom: 25px;'>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 10px;'>
                <div><strong>Nama:</strong> {app_name}</div>
                <div><strong>Tujuan Pinjaman:</strong> {loan_intent}</div>
                <div><strong>Umur:</strong> {age}</div>
                <div><strong>Jumlah Pinjaman:</strong> ${loan_amount:,.2f}</div>
                <div><strong>Pendapatan:</strong> ${income:,.2f}</div>
                <div><strong>Suku Bunga Pinjaman:</strong> {loan_int_rate}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # B. Rekomendasi
        if decision_color == "success":
            st.success(f"**Rekomendasi: {decision}**")
        elif decision_color == "warning":
            st.warning(f"**Rekomendasi: {decision}**")
        else:
            st.error(f"**Rekomendasi: {decision}**")

        # C. Metrik & Grafik (Dibagi jadi 2 Kolom)
        col_text, col_chart = st.columns([1, 1])
        
        with col_text:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color: #c0392b;'>Potensi Kerugian (Expected Loss): ${expected_loss:,.2f}</h3>", unsafe_allow_html=True)
            st.markdown(f"**Probabilitas Gagal Bayar (PD): {pd_percent:.2f}%**")

        with col_chart:
            # Gauge Chart menggunakan Plotly
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = pd_percent,
                number = {'suffix': "%", 'font': {'size': 40, 'color': '#2c3e50'}},
                domain = {'x': [0, 1], 'y': [0, 1]},
                gauge = {
                    'axis': {'range': [0, 100], 'tickwidth': 1},
                    'bar': {'color': "rgba(41, 128, 185, 0.8)"}, # Bar penunjuk berwarna biru
                    'steps': [
                        {'range': [0, 15], 'color': "rgba(46, 204, 113, 0.4)"},  # Hijau
                        {'range': [15, 30], 'color': "rgba(241, 196, 15, 0.4)"}, # Kuning
                        {'range': [30, 100], 'color': "rgba(231, 76, 60, 0.4)"}  # Merah
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 4},
                        'thickness': 0.75,
                        'value': pd_percent
                    }
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # D. Teks Kesimpulan (Di Bawah)
        ambang_batas = 15.0
        status_ambang = "di bawah" if pd_percent < ambang_batas else "di atas"
        keputusan_akhir = "Disetujui (Approved)" if pd_percent < ambang_batas else "Ditolak / Syarat Khusus"
        
        st.info(f"Aplikan bernama **{app_name}** memiliki probabilitas gagal bayar sebesar **{pd_percent:.2f}%**, yang berada {status_ambang} ambang batas {ambang_batas}%. Dengan demikian, pinjaman **{keputusan_akhir}**.")
        
else:
    # Tampilan awal jika tombol belum ditekan
    st.write("Silakan isi form di *Sidebar* sebelah kiri dan klik **Jalankan Analisis** untuk melihat detail profil risiko nasabah.")