# file: pages/Analisis Deskriptif.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Analisis Deskriptif", layout="wide")
st.title("Analisis Deskriptif")
st.write("Halaman ini menampilkan ringkasan statistik dan tabel frekuensi secara sederhana.")

# Ambil data hasil upload dari session state
data = st.session_state.get('uploaded_data')
if data is None:
    st.warning("Silakan unggah file CSV terlebih dahulu di halaman utama.")
    st.stop()

# Definisikan variabel target
target_vars = [
    ("Lama Pengajuan Klaim", "lama_pengajuan_klaim"),
    ("Besar Klaim", "besar_klaim")
]

# Loop untuk menampilkan statistik deskriptif per Tipe Klasifikasi
for title, col in target_vars:
    st.markdown(f"#### Statistik Deskriptif `{title}` per Tipe Klasifikasi")
    stats = (
        data
        .groupby("tipe_klasifikasi")[col]
        .agg(
            Jumlah_Data="count",
            Rata_rata="mean",
            Standar_Deviasi="std",
            Nilai_Minimum="min",
            Median="median",
            Nilai_Maksimum="max",
        )
        .rename_axis("Tipe Klasifikasi")
        .reset_index()
    )
    # Ubah nama kolom
    stats = stats.rename(columns={
        "Jumlah_Data": "Jumlah Data",
        "Rata_rata": "Rata-rata",
        "Standar_Deviasi": "Standar Deviasi",
        "Nilai_Minimum": "Nilai Minimum",
        "Nilai_Maksimum": "Nilai Maksimum",
    })
    # Bulatkan numeric untuk konsistensi
    for num_col in ["Rata-rata", "Standar Deviasi", "Nilai Minimum", "Median", "Nilai Maksimum"]:
        stats[num_col] = stats[num_col].round(2) if stats[num_col].dtype == float else stats[num_col]

    # Format angka: ribuan pakai titik, desimal pakai koma
    def fmt_int(x):
        s = f"{int(x):,}"       # "1,234,567"
        return s.replace(",", ".")  # "1.234.567"

    def fmt_float(x):
        s = f"{x:,.2f}"          # "1,234,567.89"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")  # "1.234.567,89"

    fmt = {
        "Jumlah Data":    lambda x: fmt_int(x),
        "Rata-rata":      lambda x: fmt_float(x),
        "Standar Deviasi":lambda x: fmt_float(x),
        "Nilai Minimum":  lambda x: fmt_int(x),
        "Median":         lambda x: fmt_float(x),
        "Nilai Maksimum": lambda x: fmt_int(x),
    }
    st.table(stats.style.format(fmt))

# Tombol Next dengan key unik

if st.button("Next"):
    st.switch_page("pages/02_Analisis Banyak Klaim.py")
