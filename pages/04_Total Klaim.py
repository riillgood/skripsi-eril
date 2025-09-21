import streamlit as st
import pandas as pd
import numpy as np

# ==============================================================================
# KONFIGURASI DAN PEMERIKSAAN DATA
# ==============================================================================

st.set_page_config(page_title="Total Klaim", layout="wide")
st.title("Estimasi Total Klaim Agregat")
st.markdown("---")

if "tabel1" not in st.session_state or "tabel2" not in st.session_state:
    st.error("Data analisis tidak ditemukan. Silakan jalankan halaman 'Analisis Banyak Klaim' dan 'Analisis Besar Klaim' terlebih dahulu.")
    st.stop()

# ==============================================================================
# FUNGSI PERHITUNGAN
# ==============================================================================

def calculate_total_claim(pareto_row, besar_row, tipe_ks):
    mb = pareto_row["Ekspektasi Banyak Klaim"]
    vb = pareto_row["Standar Deviasi Banyak Klaim"]**2
    mx = besar_row["Ekspektasi Besar Klaim"]
    vx = besar_row["Standar Deviasi Besar Klaim"]**2
    mean_tk = mb * mx
    var_tk  = (vx * mb) + ((mx**2) * vb)
    std_tk  = np.sqrt(var_tk)
    k = 1.645
    batas_atas = mean_tk + k * std_tk
    return {
        "Tipe Klasifikasi": tipe_ks,
        "Distribusi Banyak Klaim": "Pareto",
        "Distribusi Besar Klaim":  besar_row["Distribusi yang digunakan"],
        "Rentang Tanggal Analisis": pareto_row["Rentang Tanggal Analisis"],
        "Ekspektasi Total Klaim":  mean_tk,
        "Standar Deviasi Total Klaim": std_tk,
        "Batas Atas Premi (95%)": batas_atas,
    }

# ==============================================================================
# PROSES UTAMA
# ==============================================================================

df_banyak_dict = st.session_state["tabel1"]
df_besar_dict  = st.session_state["tabel2"]

all_combinations = []
tipe_list = sorted(df_banyak_dict.keys())

for tipe_ks in tipe_list:
    df_banyak = df_banyak_dict.get(tipe_ks)
    df_besar  = df_besar_dict.get(tipe_ks)
    if df_banyak is not None and not df_banyak.empty and df_besar is not None and not df_besar.empty:
        pareto_row = df_banyak.iloc[0]
        for index, besar_row in df_besar.iterrows():
            combination_result = calculate_total_claim(pareto_row, besar_row, tipe_ks)
            all_combinations.append(combination_result)

# ==============================================================================
# TABEL RINGKASAN AKHIR
# ==============================================================================

st.header("Total Klaim per Tipe Klasifikasi")

if all_combinations:
    df_summary = pd.DataFrame(all_combinations)
    tipe_ks_unik = sorted(df_summary["Tipe Klasifikasi"].unique())

    for tipe in tipe_ks_unik:
        st.subheader(f"Tabel Total Klaim untuk {tipe}")
        df_tipe = df_summary[df_summary["Tipe Klasifikasi"] == tipe]
        df_tipe_sorted = df_tipe.sort_values(by="Distribusi Besar Klaim", ascending=True).reset_index(drop=True)
        
        cols_ordered = [
            "Distribusi Banyak Klaim",
            "Distribusi Besar Klaim",
            "Rentang Tanggal Analisis",
            "Ekspektasi Total Klaim",
            "Standar Deviasi Total Klaim", 
            "Batas Atas Premi (95%)"
        ]
        
        st.table(df_tipe_sorted[cols_ordered].style.format({
            "Ekspektasi Total Klaim": "{:,.2f}",
            "Standar Deviasi Total Klaim": "{:,.2f}",
            "Batas Atas Premi (95%)": "{:,.2f}"
        }))

    st.header("Total Klaim Keseluruhan")

    def calculate_grand_total(df, distribution_name):
        if df.empty:
            return None
        
        rentang_tanggal = df["Rentang Tanggal Analisis"].iloc[0]
        total_ekspektasi = df["Ekspektasi Total Klaim"].sum()
        total_variansi = (df["Standar Deviasi Total Klaim"] ** 2).sum()
        total_std = np.sqrt(total_variansi)
        total_batas_atas = total_ekspektasi + 1.645 * total_std
        
        return pd.DataFrame([{
            "Distribusi yang digunakan": f"Pareto + {distribution_name}",
            "Rentang Tanggal Analisis": rentang_tanggal,
            "Ekspektasi Total Klaim Keseluruhan": total_ekspektasi,
            "Standar Deviasi Total Klaim Keseluruhan": total_std,
            "Batas Atas Total Klaim Keseluruhan (95%)": total_batas_atas
        }])

    df_2p = df_summary[df_summary["Distribusi Besar Klaim"] == "Weibull Min (2P)"]
    df_3p = df_summary[df_summary["Distribusi Besar Klaim"] == "Weibull Min (3P)"]

    total_2p_df = calculate_grand_total(df_2p, "Weibull Min (2P)")
    total_3p_df = calculate_grand_total(df_3p, "Weibull Min (3P)")

    if total_2p_df is not None and total_3p_df is not None:
        final_total_df = pd.concat([total_2p_df, total_3p_df]).reset_index(drop=True)
        st.subheader("Ringkasan Total Klaim Keseluruhan")
        st.table(final_total_df.style.format({
            "Ekspektasi Total Klaim Keseluruhan": "{:,.2f}",
            "Standar Deviasi Total Klaim Keseluruhan": "{:,.2f}",
            "Batas Atas Total Klaim Keseluruhan (95%)": "{:,.2f}"
        }))

else:
    st.info("Tidak ada data valid yang dapat ditotalkan. Pastikan analisis di halaman sebelumnya sudah dijalankan untuk semua tipe RS.")

st.markdown("---")

# --- TOMBOL NAVIGASI BARU YANG SUDAH DIPERBAIKI ---
# Menggunakan st.page_link adalah cara paling bersih dan modern untuk navigasi.
# Ini tidak memerlukan callback dan akan berfungsi dengan benar sekarang
# karena halaman tujuannya (03_Analisis Besar Klaim.py) sudah diperbaiki.

st.page_link(
    "pages/03_Analisis Besar Klaim.py",
    label="Back"
)