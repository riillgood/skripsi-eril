import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Total Klaim", layout="wide")
st.header("Estimasi Total Klaim BPJS di Indonesia")

# 1. Ambil data dari session
if "uploaded_data" not in st.session_state:
    st.warning("Silakan unggah file CSV atau lakukan Analisis Banyak/Besar Klaim terlebih dahulu.")
    st.stop()
data = st.session_state["uploaded_data"]

if "tabel1" not in st.session_state or "tabel2" not in st.session_state:
    st.error("Silakan jalankan halaman Analisis Banyak Klaim dan Analisis Besar Klaim terlebih dahulu.")
    st.stop()

top5_banyak = st.session_state["tabel1"]  # dict: kelas → df_top5_banyak
top5_besar  = st.session_state["tabel2"]  # dict: kelas → df_top5_besar

# 2. Buat tab per kelas
kelas_list = sorted(top5_banyak.keys())
tabs = st.tabs([f"Kelas {k}" for k in kelas_list])

for tab, kelas in zip(tabs, kelas_list):
    with tab:
        df_banyak = top5_banyak[kelas].copy()
        df_besar  = top5_besar[kelas].copy()

        # Pastikan kolom numeric
        for df in (df_banyak, df_besar):
            for col in ["Ekspektasi", "Standar Deviasi"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

        # — Buat salinan untuk tampilan dengan kolom fmt —
        disp_banyak = df_banyak.copy()
        disp_banyak["Ekspektasi"] = disp_banyak["Ekspektasi"].map(lambda x: f"{x:,.2f}")
        disp_banyak["Standar Deviasi"]   = disp_banyak["Standar Deviasi"].map(lambda x: f"{x:,.2f}")

        disp_besar = df_besar.copy()
        disp_besar["Ekspektasi"] = disp_besar["Ekspektasi"].map(lambda x: f"{x:,.4f}")
        disp_besar["Standar Deviasi"]   = disp_besar["Standar Deviasi"].map(lambda x: f"{x:,.4f}")

        # Tabel 1: Banyak Klaim
        st.subheader(f"1. Top-5 Banyak Klaim (Kelas {kelas})")
        st.table(disp_banyak[[
            "Distribusi",
            "Kolmogorov Smirnov",
            "Ekspektasi",
            "Standar Deviasi"
        ]])

        # Tabel 2: Besar Klaim
        st.subheader(f"2. Top-5 Besar Klaim (Kelas {kelas})")
        st.table(disp_besar[[
            "Distribusi",
            "Kolmogorov Smirnov",
            "Ekspektasi",
            "Standar Deviasi"
        ]])

        # Tabel 3: Kombinasi
        combis = []
        for _, n in df_banyak.iterrows():
            for _, x in df_besar.iterrows():
                mb, vb = n["Ekspektasi"], n["Standar Deviasi"]
                mx, vx = x["Ekspektasi"], x["Standar Deviasi"]
                mean_tk = mb * mx
                var_tk  = vx * mb + (mx**2) * vb
                std_tk  = np.sqrt(var_tk)
                combis.append({
                    "Distribusi Banyak Klaim": n["Distribusi"],
                    "Distribusi Besar Klaim":  x["Distribusi"],
                    "Ekspektasi Total Klaim":  mean_tk,
                    "Variansi Total Klaim":    var_tk,
                    "Standar Deviasi Total Klaim": std_tk
                })
        df_total = (
            pd.DataFrame(combis)
              .sort_values("Ekspektasi Total Klaim")
              .head(5)
              .reset_index(drop=True)
        )

        # Hitung batas atas Chebyshev pada α = 0.1
        alpha = 0.10
        k = 1 / np.sqrt(alpha)

        # Tambahkan kolom UpperBound = Ekspektasi Total + k * sqrt(Standar Deviasi Total)
        df_total["Batas Atas"] = (
            df_total["Ekspektasi Total Klaim"]
        + k * np.sqrt(df_total["Variansi Total Klaim"])
        )

        # Buat salinan untuk tampilan
        disp_total = df_total.copy()
        disp_total["Ekspektasi Total Klaim"] = disp_total["Ekspektasi Total Klaim"].map(lambda x: f"{x:,.4f}")
        disp_total["Standar Deviasi Total Klaim"] = disp_total["Standar Deviasi Total Klaim"].map(lambda x: f"{x:,.4f}")
        disp_total["Batas Atas"] = disp_total["Batas Atas"].map(lambda x: f"{x:,.4f}")

        st.subheader(f"3. Top-5 Kombinasi Total Klaim (Kelas {kelas}) dengan Batas Atas 90%")
        st.table(disp_total[[
            "Distribusi Banyak Klaim",
            "Distribusi Besar Klaim",
            "Ekspektasi Total Klaim",
            "Standar Deviasi Total Klaim",
            "Batas Atas"
        ]])
# try:
#     if 'uploaded_data' in st.session_state:
#         data = st.session_state['uploaded_data']
#         table1 = st.session_state['tabel1']
#         table2 = st.session_state['tabel2']

#         # Step 9: Ekspektasi, Variansi Total Klaim, dan Simpangan Baku Total Klaim
#         mean_tk = table1['Ekspektasi Banyak Klaim'] * table2['Ekspektasi Besar Klaim']
#         variance_tk = (table2['Variansi Besar Klaim'] * table1['Ekspektasi Banyak Klaim']) + (table2['Ekspektasi Besar Klaim']**2 * table1['Variansi Banyak Klaim'])
#         std_dev_tk = np.sqrt(variance_tk)

#         # Step 10: Batas Atas Total Klaim
#         confidence_level = 0.90
#         significance_level = 1 - confidence_level
#         k = np.sqrt(1 / significance_level)
#         upper_bound = mean_tk + (k * std_dev_tk)

#         # Create the DataFrame
#         table3 = {
#             'Kelas': [1, 2, 3, 4],
#             'Total Klaim': mean_tk,
#             'Variansi Total Klaim': variance_tk,
#             'Simpangan Baku Total Klaim': std_dev_tk,
#             'Batas Atas': upper_bound
#         }
#         table3 = pd.DataFrame(table3)
#         st.subheader("Estimasi Total Klaim BPJS per Kelas")
#         st.table(table3)

#         # Calculate sums outside the loop
#         total_klaim = sum(table3['Total Klaim'])
#         total_simpangan_baku = sum(table3['Simpangan Baku Total Klaim'])
#         total_batas_atas = sum(table3["Batas Atas"])

#         # Membuat tabel keempat
#         table4 = pd.DataFrame([{
#             "Total Klaim": total_klaim,
#             "Total Simpangan Baku": total_simpangan_baku,
#             "Total Batas Atas": total_batas_atas
#         }])
#         st.subheader("Estimasi Total Klaim BPJS di Indonesia")
#         st.table(table4)
#     else:
#         st.warning("Silakan unggah file CSV atau lakukan analisis Pareto dan Weibull terlebih dahulu.")
# except:
#     st.warning("Silakan unggah file CSV atau lakukan analisis Pareto dan Weibull terlebih dahulu.")
