import streamlit as st
import pandas as pd
import rumus as r # Import rumus.py

st.set_page_config(page_title="Estimasi Total Klaim BPJS", layout="wide")

# Initialize session state variables
if 'uploaded_data' not in st.session_state:
    st.session_state['uploaded_data'] = None
if 'tabel1' not in st.session_state:
    st.session_state['tabel1'] = pd.DataFrame()
if 'tabel2' not in st.session_state:
    st.session_state['tabel2'] = pd.DataFrame()

# Halaman utama
st.title("Estimasi Total Klaim BPJS di Indonesia")
st.subheader("Panduan dan Upload File CSV")
st.write("""
**Tata Cara Mengunggah File CSV:**
1. Siapkan file CSV Anda dengan kolom **`kelas`**, **`lama_pengajuan_klaim`**, dan **`besar_klaim`**.
   - **`kelas`**: Berisi kategori data, seperti tipe klaim atau klasifikasi lainnya.
   - **`lama_pengajuan_klaim`**: Berisi durasi waktu pengajuan klaim dalam satuan tertentu (misalnya hari).
   - **`besar_klaim`**: Berisi nilai atau jumlah klaim.
2. Pastikan nama kolom sesuai dengan ketentuan, **tidak boleh ada spasi atau karakter khusus**.
3. Unggah file dalam format **CSV (Comma-Separated Values)** melalui tombol di bawah.
4. Setelah berhasil diunggah, data akan divalidasi secara otomatis untuk memastikan kolom yang diperlukan ada.

**Catatan Penting**:
- Jika file tidak sesuai format, Anda akan mendapatkan pesan peringatan.
- Format file harus UTF-8 tanpa karakter tersembunyi.

Silakan unggah file CSV Anda di bawah ini dengan mengklik tombol "Browse files":
""")

# File upload
uploaded_file = st.file_uploader("Upload file CSV", type="csv")

if uploaded_file is not None:
    try:
        # Membaca file CSV
        data = pd.read_csv(uploaded_file)
        
        # Validasi kolom yang diperlukan
        required_columns = {"kelas", "lama_pengajuan_klaim", "besar_klaim"}
        missing_columns = required_columns - set(data.columns)
        
        if missing_columns:
            st.error(f"Kolom berikut tidak ditemukan dalam file: {', '.join(missing_columns)}. Silakan unggah file yang sesuai.")
        else:
            st.success("File berhasil diunggah dan memenuhi syarat!")
            st.write("Pratinjau data:")
            st.dataframe(data.head())  # Menampilkan pratinjau data
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")
    st.session_state['uploaded_data'] = data

    if st.button("Mulai Analisis"):
        st.switch_page("pages/01_Analisis Deskriptif.py")
        