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

# 2) Inisialisasi kelas_list & idx_besar
#    — pastikan ini **sebelum** kamu cek idx_besar di sentinel
if 'kelas_list' not in st.session_state:
    st.session_state.kelas_list = sorted(data['kelas'].unique())
if 'idx_besar' not in st.session_state:
    st.session_state.idx_besar = 0

kelas_list = st.session_state.kelas_list
last       = len(kelas_list) - 1
idx_besar  = st.session_state.idx_besar

# 3) Sentinel check (navigasi)
if idx_besar == -1:
    # Back dari besar kelas 1 → Banyak Klaim kelas terakhir
    st.session_state.idx_banyak = last
    st.session_state.idx_besar  = 0
    st.switch_page("pages/02_Analisis Banyak Klaim.py")
elif idx_besar == last + 1:
    # Next setelah besar kelas terakhir → Total Klaim
    st.session_state.idx_besar = last
    st.switch_page("pages/04_Total Klaim.py")

# 4) Ambil kelas sekarang setelah sentinel
selected_kelas = kelas_list[idx_besar]

st.sidebar.markdown(f"### Analisis Besar Klaim — Kelas {selected_kelas}")
# # 2) Sidebar: pilih kelas
# kelas_opsi     = sorted(data['kelas'].unique())
# selected_kelas = st.sidebar.selectbox("Pilih Kelas", kelas_opsi, format_func=lambda x: f"Kelas {x}")

# 3) Hitung hasil
results2_dict = r.analisis_besar_klaim(data, selected_kelas)

# 4) Ambil hasil untuk kelas yang dipilih
results2 = results2_dict.get(selected_kelas, [])
if not results2:
    st.warning(f"Tidak ada hasil untuk Kelas {selected_kelas}.")
    st.stop()

# 5) Tampilkan histogram **sekali**
st.subheader(f"Histogram Estimasi Parameter untuk Kelas {selected_kelas}")
for r in results2:
    st.markdown(f"**Distribusi {r['Distribusi']}**")
    st.image(r["Histogram"], use_container_width=True)

# 6) Tampilkan tabel Kolmogorov–Smirnov
st.subheader(f"Nilai Kolmogorov–Smirnov Kelas {selected_kelas}")
sorted_results2 = sorted(results2, key=lambda x: x['Kolmogorov-Smirnov'])
df_results2 = pd.DataFrame({
        'Distribusi': [r['Distribusi'] for r in sorted_results2],
        'Kolmogorov Smirnov': [f"{r['Kolmogorov-Smirnov']:.4f}" for r in sorted_results2],
        'Ekspektasi': [f"{r['Ekspektasi']:,.4f}" for r in sorted_results2],
        'Standar Deviasi': [f"{r['Standar Deviasi']:,.4f}" for r in sorted_results2]
    })
st.table(df_results2)

# 7) Expander hanya berisi teks summary, tanpa plot
st.subheader(f"Detail Ringkasan untuk Kelas {selected_kelas}")
for r in results2:
    with st.expander(r["Distribusi"]):
        st.write(f"**Parameter:** {r['Parameter']}")
        st.write(f"**Kolmogorov-Smirnov:** {r['Kolmogorov-Smirnov']:.4f}")
        st.write(f"**Critical Value:** {r['Critical Value']:.4f}")
        st.write(f"**H0 Ditolak:** {r['H0 Ditolak']}")

top5 = sorted_results2[:5]

# BANGUN DataFrame DENGAN ANGKA FLOAT, tanpa string formatting apapun
df_top5_besar = pd.DataFrame({
    "Distribusi":        [r["Distribusi"]        for r in top5],
    "Kolmogorov Smirnov":[r["Kolmogorov-Smirnov"] for r in top5],
    "Ekspektasi":        [r["Ekspektasi"]         for r in top5],
    "Standar Deviasi":  [r["Standar Deviasi"]    for r in top5],
}).reset_index(drop=True)

# Simpan ke session sebagai numeric murni
if 'tabel2' not in st.session_state or not isinstance(st.session_state.tabel2, dict):
    st.session_state.tabel2 = {}
st.session_state.tabel2[selected_kelas] = df_top5_besar

# Tampilkan (boleh styling di sini pakai st.dataframe)
st.subheader(f"5 Distribusi Dengan Hasil Terbaik (Kelas {selected_kelas})")
st.table(
    df_top5_besar.style.format({
        "Kolmogorov Smirnov":"{:.4f}",
        "Ekspektasi": "{:,.4f}",
        "Standar Deviasi": "{:,.4f}"
    })
)
# # 8) Tabel 5 Hasil Terbaik Setiap Kelas — simpan per kelas
# df_top5_besar = df_results2.head(5).reset_index(drop=True)

# # Pastikan tabel2 adalah dict
# if 'tabel2' not in st.session_state or not isinstance(st.session_state['tabel2'], dict):
#     st.session_state['tabel2'] = {}

# st.session_state['tabel2'][selected_kelas] = df_top5_besar

# st.subheader(f"5 Distribusi Dengan Hasil Terbaik (Kelas {selected_kelas})")
# st.table(df_top5_besar)

# 5) Tombol Back / Next
def go_back_besar():
    st.session_state.idx_besar = idx_besar - 1

def go_next_besar():
    st.session_state.idx_besar = idx_besar + 1

col1, col2 = st.columns(2)
col1.button("Back",   on_click=go_back_besar, key="btn_back_besar")
col2.button("Next",    on_click=go_next_besar, key="btn_next_besar")

# 6) Hentikan render di sini agar Streamlit langsung rerun dengan idx_besar baru
st.stop()

# # 8) Tabel 5 Hasil Terbaik Setiap Kelas
# df_top5_besar = df_results2.head(5).reset_index(drop=True)
# st.session_state['tabel2'] = df_top5_besar
# st.subheader("5 Distribusi Dengan Hasil Terbaik")
# st.table(df_top5_besar)

# # 6) Tombol Next
# if st.button("Next"):
#     st.switch_page("pages/04_Total Klaim.py")

# if 'uploaded_data' in st.session_state:
#     data = st.session_state['uploaded_data']
#     abk_results2 = r.analisis_besar_klaim(data)
    
#     # Ambil list dari tuple
#     if isinstance(abk_results2, tuple) and len(abk_results2) > 0:
#         dict_list = abk_results2[0]
#     else:
#         dict_list = abk_results2
    
#     # Pastikan dict_list adalah list
#     if not isinstance(dict_list, list):
#         dict_list = [dict_list]
    
#     # Buat DataFrame
#     abk2_table = pd.DataFrame(dict_list)
    
#     # Format angka untuk tampilan yang lebih baik - gunakan notasi yang lebih readable
#     for col in abk2_table.columns:
#         if col in ['Ekspektasi Besar Klaim', 'Variansi Besar Klaim', 'Simpangan Baku Besar Klaim']:
#             # Format angka dalam miliar atau juta untuk keterbacaan
#             abk2_table[col] = abk2_table[col].astype(float)
            
#     # Tampilkan tabel dengan format yang lebih sederhana
#     st.header("Hasil Terbaik Analisis Besar Klaim")
    
#     # Gunakan st.table() untuk tampilan yang lebih statis (tidak ada tanda seru)
#     st.table(abk2_table.style.format({
#         'Ekspektasi Besar Klaim': '{:,.2f}',
#         'Variansi Besar Klaim': '{:,.2f}',
#         'Simpangan Baku Besar Klaim': '{:,.2f}'
#     }))
    
#     # Simpan tabel yang sudah diformat
#     st.session_state['tabel2'] = abk2_table
    
#     if st.button("Next"):
#         st.switch_page("pages/04_Total Klaim.py")
# else:
#     st.warning("Silakan unggah file CSV terlebih dahulu.")