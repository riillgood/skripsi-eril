import streamlit as st
import pandas as pd
import rumus as r

st.set_page_config(page_title="Analisis Banyak Klaim", layout="wide")
st.title("Estimasi Banyak Klaim")
st.subheader("Pemodelan Waktu Antar Kedatangan")

#debug_mode = st.checkbox("Tampilkan log debug")

# 1) Pastikan data ter-upload
if 'uploaded_data' not in st.session_state:
    st.warning("Silakan unggah file CSV terlebih dahulu.")
    st.stop()
data = st.session_state['uploaded_data']
data["tanggal_klaim_diajukan"] = pd.to_datetime(
    data["tanggal_klaim_diajukan"], errors="coerce"
)

# — Sidebar: input tanggal dulu —
st.sidebar.markdown("### Rentang Tanggal Analisis")
# — Sidebar: Rentang Tanggal Analisis (persist & selalu tampil) —
min_date = data["tanggal_klaim_diajukan"].min().date()
max_date = data["tanggal_klaim_diajukan"].max().date()

# Ambil value dari session jika sudah ada, atau default ke (min,max)
default_dates = st.session_state.get("analysis_dates", (min_date, max_date))
tanggal = st.sidebar.date_input(
    "Masukkan tanggal yang akan dianalisis:",
    value=default_dates,
    min_value=min_date,
    max_value=max_date,
    key="date_input"  # beri key supaya streamlit simpan state
)

# Validasi: paksa user memilih rentang non-default
if not (isinstance(tanggal, (list, tuple)) and len(tanggal)==2):
    st.warning("Silakan pilih rentang tanggal yang lengkap—dua tanggal, bukan satu.")
    st.stop()
if tanggal == (min_date, max_date):
    st.info("Default tanggal masih dari data ter‑upload. Silakan ubah rentang di sidebar untuk menjalankan analisis.")
    st.stop()

# Simpan ke session (disimpan otomatis karena `key="date_input"` tetapi kita juga simpan eksplisit)
st.session_state.analysis_dates = tanggal
start_date, end_date = tanggal

# 1) Inisialisasi kelas_list & idx_banyak
if 'kelas_list' not in st.session_state:
    st.session_state.kelas_list = sorted(data['kelas'].unique())
if 'idx_banyak' not in st.session_state:
    st.session_state.idx_banyak = 0

kelas_list = st.session_state.kelas_list
last       = len(kelas_list) - 1
idx_banyak = st.session_state.idx_banyak

# 2) Sentinel check (navigasi) — **harus** sebelum pakai idx_banyak
if idx_banyak == -1:
    # Back dari kelas pertama → Analisis Deskriptif
    st.session_state.idx_banyak = 0
    st.switch_page("pages/01_Analisis Deskriptif.py")
elif idx_banyak == last + 1:
    # Next setelah kelas terakhir → Analisis Besar Klaim
    st.session_state.idx_banyak = last
    st.session_state.idx_besar  = 0
    st.switch_page("pages/03_Analisis Besar Klaim.py")

# 3) Sekarang idx_banyak pasti 0…last, ambil kelas:
selected_kelas = kelas_list[idx_banyak]
st.sidebar.markdown(f"### Analisis Banyak Klaim — Kelas {selected_kelas}")

# 3) Hitung hasil
results_dict = r.analisis_banyak_klaim(data, selected_kelas=selected_kelas, analysis_start=start_date, analysis_end=end_date)

# 4) Ambil hasil untuk kelas yang dipilih
results = results_dict.get(selected_kelas, [])
if not results:
    st.warning(f"Tidak ada hasil untuk Kelas {selected_kelas}.")
    st.stop()

# 5) Tampilkan histogram **sekali**
st.subheader(f"Histogram Estimasi Parameter untuk Kelas {selected_kelas}")
for r in results:
    st.markdown(f"**Distribusi {r['Distribusi']}**")
    st.image(r["Histogram"], use_container_width=True)

# 6) Tampilkan tabel Kolmogorov–Smirnov
st.subheader(f"Nilai Kolmogorov–Smirnov Kelas {selected_kelas}")
sorted_results = sorted(results, key=lambda x: x['Kolmogorov-Smirnov'])
df_results = pd.DataFrame({
        'Distribusi': [r['Distribusi'] for r in sorted_results],
        'Kolmogorov Smirnov': [f"{r['Kolmogorov-Smirnov']:.4f}" for r in sorted_results],
        'Fungsi Intensitas': [r['Fungsi Intensitas'] for r in sorted_results],
        'Ekspektasi': [r['Ekspektasi'] for r in sorted_results],
        'Standar Deviasi': [r['Standar Deviasi'] for r in sorted_results]
        })
st.table(df_results)

# 7) Expander hanya berisi teks summary, tanpa plot
st.subheader(f"Detail Ringkasan untuk Kelas {selected_kelas}")
for r in results:
    with st.expander(r["Distribusi"]):
        st.image(r["Fungsi Hazard"], use_container_width=True)
        st.write(f"**Parameter:** {r['Parameter']}")
        st.write(f"**Kolmogorov-Smirnov:** {r['Kolmogorov-Smirnov']:.4f}")
        st.write(f"**Critical Value:** {r['Critical Value']:.4f}")
        st.write(f"**H0 Ditolak:** {r['H0 Ditolak']}")

# 8) Tabel 5 Hasil Terbaik Setiap Kelas — simpan per kelas
df_top5_banyak = df_results.head(5).reset_index(drop=True)

# Pastikan tabel1 adalah dict
if 'tabel1' not in st.session_state or not isinstance(st.session_state['tabel1'], dict):
    st.session_state['tabel1'] = {}

st.session_state['tabel1'][selected_kelas] = df_top5_banyak

st.subheader(f"5 Distribusi Dengan Hasil Terbaik (Kelas {selected_kelas})")
st.table(df_top5_banyak)

# 4) Tombol navigasi:
def go_back():
    st.session_state.idx_banyak = idx_banyak - 1

def go_next():
    st.session_state.idx_banyak = idx_banyak + 1

col1, col2 = st.columns(2)
col1.button("Back", on_click=go_back, key="back_banyak")
col2.button("Next", on_click=go_next, key="next_banyak")

# **Stop** eksekusi agar page tidak melanjutkan render kedua tombol
st.stop()

# # 8) Tabel 5 Hasil Terbaik Setiap Kelas
# df_top5_banyak = df_results.head(5).reset_index(drop=True)
# st.session_state['tabel1'] = df_top5_banyak
# st.subheader("5 Distribusi Dengan Hasil Terbaik")
# st.table(df_top5_banyak)

# # 6) Tombol Next
# if st.button("Next"):
#     st.switch_page("pages/03_Analisis Besar Klaim.py")

# Cek apakah data ada di session state
# if 'uploaded_data' in st.session_state:
#     data = st.session_state['uploaded_data']
#     abk_results = r.analisis_banyak_klaim(data)
    
    # # Ambil list dari tuple
    # if isinstance(abk_results, tuple) and len(abk_results) > 0:
    #     dict_list = abk_results[0]
    # else:
    #     dict_list = abk_results
    
    # # Pastikan dict_list adalah list
    # if not isinstance(dict_list, list):
    #     dict_list = [dict_list]
    
    # # Buat DataFrame
    # abk_table = pd.DataFrame(dict_list)
    
    # # Format angka untuk tampilan yang lebih baik - gunakan notasi yang lebih readable
    # for col in abk_table.columns:
    #     if col in ['Lamda', 'Ekspektasi Banyak Klaim', 'Variansi Banyak Klaim', 'Simpangan Baku Banyak Klaim']:
    #         # Format angka dalam miliar atau juta untuk keterbacaan
    #         abk_table[col] = abk_table[col].astype(float)
            
    # # Tampilkan tabel dengan format yang lebih sederhana
    # st.header("Hasil Terbaik Analisis Banyak Klaim")
    
    # # Gunakan st.table() untuk tampilan yang lebih statis (tidak ada tanda seru)
    # st.table(abk_table.style.format({
    #     'Lamda': '{:,.2f}',
    #     'Ekspektasi Banyak Klaim': '{:,.2f}',
    #     'Variansi Banyak Klaim': '{:,.2f}',
    #     'Simpangan Baku Banyak Klaim': '{:,.2f}'
    # }))
    
    # # Simpan tabel yang sudah diformat
    # st.session_state['tabel2'] = abk_table
    
#     if st.button("Next"):
#         st.switch_page("pages/03_Analisis Besar Klaim.py")
# else:
#     st.warning("Silakan unggah file CSV terlebih dahulu.")
