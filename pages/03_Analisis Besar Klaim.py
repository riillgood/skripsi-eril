import streamlit as st
import pandas as pd
import numpy as np
import rumus as r

st.set_page_config(page_title="Analisis Besar Klaim", layout="wide")
st.title("Estimasi Besar Klaim")
st.subheader("Pemodelan Besar Klaim")

# 1) Pastikan data ter-upload
if 'uploaded_data' not in st.session_state:
    st.warning("Silakan unggah file CSV terlebih dahulu.")
    st.stop()
data = st.session_state['uploaded_data']

# 2) Inisialisasi tipe_list & idx_besar
if 'tipe_list' not in st.session_state:
    st.session_state.tipe_list = sorted(data['tipe_klasifikasi'].unique())
if 'idx_besar' not in st.session_state:
    st.session_state.idx_besar = 0

tipe_list = st.session_state.tipe_list
idx_besar  = st.session_state.idx_besar

# 4) Ambil tipe sekarang setelah sentinel
# Pastikan idx_besar tidak keluar dari rentang yang valid
if idx_besar >= len(tipe_list):
    idx_besar = len(tipe_list) - 1
    st.session_state.idx_besar = idx_besar

selected_tipe = tipe_list[idx_besar]

st.sidebar.markdown(f"### Analisis Besar Klaim — {selected_tipe}")

# 3) Hitung hasil
results2_dict = r.analisis_besar_klaim(data, selected_tipe)

# 4) Ambil hasil untuk tipe yang dipilih
results2 = results2_dict.get(selected_tipe, [])
if not results2:
    st.warning(f"Tidak ada hasil analisis yang dapat ditampilkan untuk {selected_tipe}.")
    # Jangan hentikan aplikasi sepenuhnya, tapi sediakan tombol navigasi
else:
    # 5) Tampilkan histogram
    st.subheader(f"Ringkasan Hasil Estimasi Parameter Distribusi Model Besar Klaim untuk {selected_tipe}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{results2[0]['Distribusi']}**") # Weibull 3P
        st.image(results2[0]["Histogram"], use_container_width=True)
    with col2:
        st.markdown(f"**{results2[1]['Distribusi']}**") # Weibull 2P
        st.image(results2[1]["Histogram"], use_container_width=True)

    # Urutkan hasil berdasarkan nilai KS terkecil (terbaik)
    sorted_results2 = sorted(results2, key=lambda x: x['Kolmogorov-Smirnov'])

    df_summary = pd.DataFrame(sorted_results2)

    # 1. Ambil dan format rentang tanggal dari session_state
    if 'analysis_dates' in st.session_state:
        start_date, end_date = st.session_state.analysis_dates
        rentang_tanggal_str = f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}"
    else:
        # Fallback jika halaman ini diakses langsung tanpa memilih tanggal
        rentang_tanggal_str = "Tidak Ditetapkan"

    # 2. Buat DataFrame HANYA untuk kolom utama yang ingin ditampilkan
    df_summary_simple = pd.DataFrame({
        "Distribusi yang digunakan": [r["Distribusi"] for r in sorted_results2],
        "Rentang Tanggal Analisis": rentang_tanggal_str,
        "Ekspektasi Besar Klaim": [r["Ekspektasi"] for r in sorted_results2],
        "Standar Deviasi Besar Klaim": [r["Standar Deviasi"] for r in sorted_results2]
    })

    # 3. Buat expander untuk menampilkan detail tambahan
    for result in sorted_results2:
        with st.expander(f"Detail untuk Distribusi {result['Distribusi']}"):
            st.write(f"**Parameter:** {result['Parameter']}")
            st.write(f"**Kolmogorov Smirnov:** {result['Kolmogorov-Smirnov']:.4f}")
            st.write(f"**Critical Value:** {result['Critical Value']:.4f}")
            st.write(f"**H0 Ditolak:** {result['H0 Ditolak']}")

    # 2. Tampilkan tabel utama yang lebih sederhana
    st.subheader(f"Ringkasan Hasil Estimasi Besar Klaim untuk {selected_tipe}")
    st.table(df_summary_simple.style.format({
        "Ekspektasi Besar Klaim": "{:,.2f}",
        "Standar Deviasi Besar Klaim": "{:,.2f}"
    }))

    # Simpan hasil ke session state untuk halaman berikutnya
    if 'tabel2' not in st.session_state or not isinstance(st.session_state.tabel2, dict):
        st.session_state.tabel2 = {}
    
    # Ganti nama kolom di DataFrame yang akan disimpan ke session state
    df_summary = df_summary.rename(columns={
        'Distribusi': 'Distribusi yang digunakan',
        'Ekspektasi': 'Ekspektasi Besar Klaim',
        'Standar Deviasi': 'Standar Deviasi Besar Klaim'
    })
    
    st.session_state.tabel2[selected_tipe] = df_summary # Simpan seluruh summary

# --- LOGIKA TOMBOL BARU YANG LEBIH STABIL ---
last = len(tipe_list) - 1

# Fungsi callback sekarang HANYA mengubah state atau menyetel "bendera" navigasi
def go_back_besar():
    if st.session_state.idx_besar == 0:
        # Setel bendera untuk pindah ke halaman sebelumnya
        st.session_state.navigate_to_page_2 = True
    else:
        st.session_state.idx_besar -= 1

def go_next_besar():
    if st.session_state.idx_besar == last:
        # Setel bendera untuk pindah ke halaman berikutnya
        st.session_state.navigate_to_page_4 = True
    else:
        st.session_state.idx_besar += 1

col1, col2 = st.columns(2)
col1.button("Back", on_click=go_back_besar, key="btn_back_besar")
col2.button("Next", on_click=go_next_besar, key="btn_next_besar")

# --- KONTROL NAVIGASI (DI LUAR CALLBACK) ---
# Blok ini akan memeriksa bendera setelah tombol diklik dan script berjalan kembali

if st.session_state.get("navigate_to_page_2", False):
    st.session_state.navigate_to_page_2 = False # Reset bendera
    # Atur state untuk halaman tujuan
    st.session_state.idx_banyak = len(st.session_state.tipe_list) - 1
    st.switch_page("pages/02_Analisis Banyak Klaim.py")

if st.session_state.get("navigate_to_page_4", False):
    st.session_state.navigate_to_page_4 = False # Reset bendera
    st.switch_page("pages/04_Total Klaim.py")

st.stop()