import streamlit as st
import pandas as pd
import rumus as r
import datetime

st.set_page_config(page_title="Analisis Banyak Klaim", layout="wide")
st.title("Estimasi Banyak Klaim")
st.subheader("Pemodelan Waktu Antar Kedatangan (Distribusi Pareto)")

# --- Pastikan data ada ---
if 'uploaded_data' not in st.session_state:
    st.warning("Silakan unggah file CSV terlebih dahulu.")
    st.stop()

data = st.session_state['uploaded_data']
data["tanggal_klaim_diajukan"] = pd.to_datetime(
    data["tanggal_klaim_diajukan"], errors="coerce"
)

# --- Sidebar Rentang Tanggal ---
st.sidebar.markdown("### Rentang Tanggal Analisis")
min_date = data["tanggal_klaim_diajukan"].min().date()
#max_date = data["tanggal_klaim_diajukan"].max().date()
default_dates = st.session_state.get("analysis_dates", (min_date, datetime.date.today()))


tanggal = st.sidebar.date_input(
    "Masukkan tanggal yang akan dianalisis:",
    value=default_dates,
    min_value=min_date,
    #max_value=max_date,
    key="date_input"
)

if not (isinstance(tanggal, (list, tuple)) and len(tanggal) == 2):
    st.warning("Silakan pilih rentang tanggal yang lengkap (dua tanggal).")
    st.stop()
if tanggal == (min_date, datetime.date.today()):
    st.info("Default tanggal masih dari data ter-upload. Silakan ubah rentang di sidebar untuk menjalankan analisis.")
    st.stop()

st.session_state.analysis_dates = tanggal
start_date, end_date = tanggal

# --- Navigasi antar tipe rumah sakit ---
if 'tipe_list' not in st.session_state:
    st.session_state.tipe_list = sorted(data['tipe_klasifikasi'].unique())
if 'idx_banyak' not in st.session_state:
    st.session_state.idx_banyak = 0

tipe_list = st.session_state.tipe_list
idx_banyak = st.session_state.idx_banyak
last = len(tipe_list) - 1

if idx_banyak == -1:
    st.session_state.idx_banyak = 0
    st.switch_page("pages/01_Analisis Deskriptif.py")
elif idx_banyak == last + 1:
    st.session_state.idx_banyak = last
    st.session_state.idx_besar = 0
    st.switch_page("pages/03_Analisis Besar Klaim.py")

selected_tipe = tipe_list[idx_banyak]
st.sidebar.markdown(f"### Analisis Banyak Klaim — {selected_tipe}")

# --- Jalankan analisis Pareto saja ---
all_results = r.analisis_banyak_klaim( # Ganti nama variabel agar lebih jelas
    data, selected_tipe=selected_tipe,
    analysis_start=start_date, analysis_end=end_date
)

# --- Tampilkan hasil ---
if selected_tipe in all_results:
    # Ambil dictionary hasil
    result = all_results[selected_tipe][0]

    # 1. Tampilkan plot utama (Fungsi Hazard & Histogram)
    st.subheader(f"Ringkasan Hasil Estimasi Parameter Distribusi Pareto dari Waktu Antar Kedatangan untuk {selected_tipe}")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Histogram**")
        st.image(result.get("Histogram"), use_container_width=True)
    with col2:
        st.markdown("**Fungsi Hazard**")
        st.image(result.get("Fungsi Hazard"), use_container_width=True)

    # 2. Tampilkan tabel ringkasan utama
    # Format rentang tanggal menjadi string untuk ditampilkan
    rentang_tanggal_str = f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}"

    df_summary = pd.DataFrame([{
        'Analisis': 'Banyak Klaim',
        'Rentang Tanggal Analisis': rentang_tanggal_str,  # <-- Kolom baru ditambahkan di sini
        'Parameter Proses Poisson': result.get('Fungsi Intensitas'),
        'Ekspektasi Banyak Klaim': result.get('Ekspektasi'),
        'Standar Deviasi Banyak Klaim': result.get('Standar Deviasi')
    }])
    
    # 3. Buat expander untuk detail tambahan
    with st.expander("Detail untuk Distribusi Pareto"):
        st.write(f"**Parameter:** {result.get('Parameter')}")
        st.write(f"**Kolmogorov Smirnov:** {result.get('Kolmogorov-Smirnov'):.4f}")
        st.write(f"**Critical Value:** {result.get('Critical Value'):.4f}")
        st.write(f"**H0 Ditolak:** {result.get('H0 Ditolak')}")

    st.subheader(f"Ringkasan Hasil Estimasi Banyak Klaim untuk {selected_tipe}")
    st.table(df_summary.style.format({
        'Parameter Proses Poisson': '{:,.4f}',
        'Ekspektasi Banyak Klaim': '{:,.4f}',
        'Standar Deviasi Banyak Klaim': '{:,.4f}'
    }))

    # --- PERBAIKAN: Simpan hasil ke session state untuk halaman berikutnya ---
    if 'tabel1' not in st.session_state or not isinstance(st.session_state.tabel1, dict):
        st.session_state.tabel1 = {}
    st.session_state.tabel1[selected_tipe] = df_summary # Simpan summary Pareto

else:
    st.warning(f"Analisis untuk '{selected_tipe}' tidak dapat diselesaikan atau tidak menghasilkan data.")

# --- Tombol navigasi (tidak perlu diubah) ---
def go_back():
    st.session_state.idx_banyak = st.session_state.idx_banyak - 1

def go_next():
    st.session_state.idx_banyak = st.session_state.idx_banyak + 1

col1, col2 = st.columns(2)
col1.button("Back", on_click=go_back, key="back_banyak")
col2.button("Next", on_click=go_next, key="next_banyak")

st.stop()

# # 8) Tabel 5 Hasil Terbaik Setiap tipe
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
