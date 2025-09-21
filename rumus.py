import numpy as np
import pandas as pd
import streamlit as st
import sklearn as sk
from scipy import stats
from scipy.stats import kstest, pareto, weibull_min
from scipy.special import gamma
# from scipy.integrate import trapezoid
from scipy.integrate import quad
from scipy.interpolate import interp1d
from collections import namedtuple
from pandas.tseries.offsets import CustomBusinessDay
import holidays
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from io import BytesIO



def read_data(file_path):
    """
    Membaca data dari file CSV.
    """
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Error: {e}")
        return None

# # --- Struktur spesifikasi distribusi ---
# DistSpec = namedtuple("DistSpec", ["dist", "param_names"])

# distributions = {
#     # Distribusi baru dengan penamaan simbol sesuai referensi:
#     "Alpha":                DistSpec(stats.alpha,          ["α", "μ", "σ"]),        # a->alpha, loc->mu, scale->sigma
#     "Beta Prime":      DistSpec(stats.betaprime,  ["α","β","μ","σ"]),
#     "Burr":                 DistSpec(stats.burr,           ["k", "α", "γ", "β"]),  # c->k, d->alpha, loc->gamma, scale->beta
#     "Burr Type XII":   DistSpec(stats.burr12,     ["k", "α", "γ", "β"]),
#     "Exponential":          DistSpec(stats.expon,          ["γ", "λ"]),               # loc=gamma, inv scale=lambda
#     "Fatigue Life":    DistSpec(stats.fatiguelife, ["α", "γ", "β"]),               # a->alpha, loc->gamma, scale->beta
#     "Fisk":            DistSpec(stats.fisk,       ["α","μ","σ"]),
#     "Gamma":                DistSpec(stats.gamma,          ["α", "γ", "β"]),         # a->alpha, loc->gamma, scale->beta
#     "Generalized Extreme Value": DistSpec(stats.genextreme, ["k", "μ", "σ"]),   # c->k, loc->mu, scale->sigma
#     "Generalized Gamma":    DistSpec(stats.gengamma,       ["k", "α", "γ", "β"]), # a->k, c->alpha, loc->gamma, scale->beta
#     "Generalized Pareto":   DistSpec(stats.genpareto,      ["k", "μ", "σ"]),         # c->k, loc->mu, scale->sigma
#     "Gumbel Right":         DistSpec(stats.gumbel_r,       ["μ", "σ"]),             # loc=mu, scale=sigma
#     "Gompertz":        DistSpec(stats.gompertz,   ["α","β","μ"]),
#     "Inverse Gamma":        DistSpec(stats.invgamma,       ["α", "μ", "σ"]),          # shape->alpha, loc->mu, scale->sigma
#     "Inverse Gaussian":     DistSpec(stats.invgauss,       ["λ", "μ", "γ"]),        # shape invgauss->lambda, scale->mu, loc->gamma
#     "Laplace":         DistSpec(stats.laplace,    ["μ","σ"]),               # loc=mu, scale=sigma
#     "Levy":                 DistSpec(stats.levy,           ["γ", "σ"]),               # loc->gamma, scale->sigma
#     "Log-Gamma":            DistSpec(stats.loggamma,       ["α", "β"]),               # c->alpha, scale->beta
#     "Log-Normal":           DistSpec(stats.lognorm,        ["σ", "γ", "μ"]),         # s->sigma, loc->gamma, scale->mu
#     "Logistic":        DistSpec(stats.logistic,   ["μ","σ"]),
#     "Lomax":           DistSpec(stats.lomax,      ["α","σ"]),
#     "Pareto":               DistSpec(stats.pareto,         ["α", "β"]),               # shape=alpha, scale=beta (loc=0)
#     "Rayleigh":             DistSpec(stats.rayleigh,       ["γ", "σ"]),               # loc->gamma, scale->sigma
#     "Wald":            DistSpec(stats.wald,       ["μ","λ","γ"]),
#     "Weibull Min":          DistSpec(stats.weibull_min,    ["α", "γ", "β"]),        # c->alpha, loc->gamma, scale->beta
#     "Weibull Max":          DistSpec(stats.weibull_max,    ["α", "γ", "β"]),        # c->alpha, loc->gamma, scale->beta
# }

def estimasi_pareto(data):
    """
    Estimasi distribusi Pareto (hanya α dan β, dengan loc=0).

    Returns:
        params: (alpha, beta)
        buffer: BytesIO berisi PNG plot histogram + PDF
        param_str: ringkasan 'α=.., β=..'
    """
    # Filter data agar hanya positif
    data = data[data > 0]

    # Fit parameter Pareto dengan loc=0 (fix)
    alpha, loc, beta = stats.pareto.fit(data, floc=0)

    # Rentang data untuk PDF
    x = np.linspace(np.percentile(data, 10), np.percentile(data, 90), 1000)
    y = stats.pareto.pdf(x, alpha, loc=0, scale=beta)

    # Pastikan nilai y valid
    y = np.nan_to_num(y, nan=0.0, posinf=np.max(y[np.isfinite(y)]), neginf=0.0)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Histogram data
    bin_edges = np.histogram_bin_edges(data, bins="auto")
    ax.hist(data, bins=bin_edges, density=True, alpha=0.6,
            label="Data", color='skyblue', edgecolor='black')

    # Plot PDF Pareto
    ax.plot(x, y, label="PDF Pareto", color="red", linewidth=2)
    ax.set_title("Estimasi Distribusi Pareto", fontsize=12)
    ax.set_xlabel("Nilai")
    ax.set_ylabel("Kepadatan")
    ax.set_yscale("log")

    counts, _ = np.histogram(data, bins=bin_edges, density=True)
    ymax = max(counts.max(), np.max(y))
    ax.set_ylim(1e-5, ymax * 1.1)

    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)

    # Simpan ke buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)

    # Return 3 hal
    params = (alpha, beta)
    param_str = f"α={alpha:.4f}, β={beta:.4f}"

    return params, buffer, param_str

def kolmogorov_smirnov_pareto(data, params):
    """
    Uji Kolmogorov–Smirnov khusus distribusi Pareto (loc=0).
    
    Args:
        data   : array data sampel
        params : tuple (alpha, beta) hasil fit
    
    Returns:
        ks_stat   : statistik KS
        ks_pval   : p-value KS
        crit_val  : nilai kritis KS pada α=0.05
    """
    alpha, beta = params
    
    # Pastikan data hanya positif
    data = data[data > 0]
    
    # Buat distribusi Pareto dengan α dan β (loc=0)
    frozen = pareto(alpha, loc=0, scale=beta)
    
    # Hitung statistik KS
    ks_stat, ks_pval = kstest(data, frozen.cdf)
    
    # Nilai kritis (signifikansi 5%)
    crit_val = 1.36 / np.sqrt(len(data))
    
    return ks_stat, ks_pval, crit_val

def hazard_pareto(alpha, beta, t_max=100, n_points=200):
    """
    Fungsi hazard distribusi Pareto tipe I + plot.
    
    h(t) = α / t, untuk t >= beta
    
    Args:
        alpha : parameter shape (α)
        beta  : parameter skala (β)
        t_max : batas maksimum t untuk plotting
        n_points : jumlah titik pada sumbu t
    
    Returns:
        t        : array waktu
        h        : array nilai hazard
        buffer   : BytesIO berisi plot PNG
    """
    # Buat range t mulai dari beta sampai t_max
    t = np.linspace(beta, t_max, n_points)
    
    # Hitung hazard function
    hazard = alpha / t
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, hazard, label=f"Hazard Pareto (α={alpha:.2f}, β={beta:.2f})", color="red")
    ax.set_xlabel("t")
    ax.set_ylabel("h(t)")
    ax.set_title("Fungsi Hazard Distribusi Pareto", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    
    # Simpan ke buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    
    return t, hazard, buffer

def hitung_hari_kerja(start_date, end_date):
    """
    Menghitung semua hari kerja (Senin-Jumat) dalam rentang tanggal,
    dan mengecualikan hari libur nasional Indonesia.
    """
    # Membuat daftar hari libur nasional Indonesia.
    # Rentang tahun dibuat dinamis berdasarkan tanggal input.
    id_holidays = holidays.country_holidays('ID', years=range(start_date.year, end_date.year + 1))
    
    # Membuat rentang semua hari dalam kalender.
    all_days = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Filter untuk hari yang BUKAN Sabtu/Minggu DAN BUKAN hari libur.
    workdays = [
        d for d in all_days 
        if d.weekday() < 5 and d.normalize() not in id_holidays
    ]
    
    return workdays

# 2. Fungsi t1 dan t2
def tentukan_t1_t2(global_start, start_date, end_date, holidays=None): # Anda bisa biarkan 'holidays=None' di sini, tidak masalah
    
    # --- PERBAIKAN DI SINI ---
    # Hapus argumen 'holidays' saat memanggil fungsi
    workdays_before = hitung_hari_kerja(global_start, start_date - pd.Timedelta(days=1))
    count_before = len(workdays_before)
    
    # Hapus argumen 'holidays' di sini juga
    workdays_month = hitung_hari_kerja(start_date, end_date)
    count_month = len(workdays_month)
    # --- AKHIR PERBAIKAN ---

    if count_month == 0:
        return None, None
        
    t1 = count_before + 1
    t2 = count_before + count_month
    
    return t1, t2

# 3. Fungsi intensitas λ Pareto + Np_t
def intensitas_pareto(global_start, start_date, end_date, alpha, full_data, holidays=None):
    """
    Menghitung λ(t) Pareto.
    Np_t dihitung dari train_data yang diberikan.
    """
    nama_kolom_tanggal = "tanggal_klaim_diajukan"
    # Hitung t1 dan t2 (tidak berubah)
    t1, t2 = tentukan_t1_t2(global_start, start_date, end_date, holidays)
    if t1 is None or t2 is None:
        return None, None, None

    # --- PERUBAHAN UTAMA DI SINI ---
    # Pastikan kolom tanggal dalam format datetime untuk perbandingan yang akurat
    # Anda mungkin sudah melakukan ini di fungsi pemanggil, tapi aman untuk memastikan lagi
    full_data[nama_kolom_tanggal] = pd.to_datetime(full_data[nama_kolom_tanggal])
    
    # Filter seluruh data dari awal hingga tanggal akhir periode analisis
    data_hingga_akhir_analisis = full_data[full_data[nama_kolom_tanggal] <= pd.to_datetime(end_date)]
    
    # Np_t adalah jumlah baris dari data yang telah difilter sesuai permintaan
    np_t = data_hingga_akhir_analisis.shape[0]
    # --- AKHIR PERUBAHAN ---

    # Hitung λ(t) menggunakan Np_t yang baru dihitung
    lamda = np_t * alpha * (np.log(t2) - np.log(t1))
    
    return lamda, t1, t2, np_t

def eks_var_pareto(lamda):
    """
    Menghitung ekspektasi E[N(t)] dan simpangan baku σ[N(t)] 
    berdasarkan nilai λ(t).

    Params:
        lamda : float
            Nilai intensitas λ(t) hasil dari fungsi intensitas_pareto.
    
    Returns:
        E_Nt : ekspektasi
        std_Nt : simpangan baku
        var_Nt : variansi
    """
    if lamda <= 0:
        return None, None, None

    exp_lam = np.exp(lamda)

    # Ekspektasi
    mean_nt = (lamda * exp_lam) / (exp_lam - 1)

    # Variansi
    var_nt = (lamda * exp_lam / (exp_lam - 1)) * (1 - lamda / (exp_lam - 1))

    # Simpangan baku
    std_nt = np.sqrt(var_nt)

    return mean_nt, std_nt, var_nt

def analisis_banyak_klaim(data: pd.DataFrame, selected_tipe=None, analysis_start=None, analysis_end=None):
    # — 0) Parse tanggal analisis jadi Timestamp —
    analysis_start = pd.to_datetime(analysis_start).normalize()
    analysis_end   = pd.to_datetime(analysis_end).normalize()

    # — 1) Siapkan data & kolom datetime —
    df = data.copy()
    df["tanggal_klaim_diajukan"] = pd.to_datetime(
        df["tanggal_klaim_diajukan"], errors="coerce"
    ).dt.normalize()

    # — 2) Tentukan list tipe yang mau diproses —
    if selected_tipe:
        tipe_list = [selected_tipe]
    else:
        tipe_list = df["tipe_klasifikasi"].unique()

    all_results = {}

    for tipe in tipe_list:
        class_data = df[df["tipe_klasifikasi"] == tipe].sort_values("tanggal_klaim_diajukan")
        if class_data.empty:
            st.warning(f"Tidak ada data untuk tipe {tipe}")
            continue

        # Tentukan periode training di fungsi pemanggil
        class_data["bulan"] = class_data["tanggal_klaim_diajukan"].dt.to_period("M")
        bulan_unik    = class_data["bulan"].unique()
        n_bulan_train = max(1, int(0.8 * len(bulan_unik)))
        bulan_train   = bulan_unik[:n_bulan_train]
        train_data    = class_data[class_data["bulan"].isin(bulan_train)]

        # Siapkan sampel untuk estimasi parameter dari data training
        sampel = train_data["lama_pengajuan_klaim"].dropna().astype(float)
        sampel = sampel[sampel > 0].values
        if len(sampel) < 10:
            st.warning(f"Data training terlalu sedikit untuk tipe {tipe} ({len(sampel)} obs)")
            continue

        try:
            # a) Estimasi parameter dari sampel data training
            params, hist_buf, param_str = estimasi_pareto(sampel)
            alpha, beta = params

            # b) Uji Kolmogorov-Smirnov
            ks_stat, ks_pval, crit = kolmogorov_smirnov_pareto(sampel, params)

            # c) Fungsi Hazard
            _, hazard_values, hazard_buf = hazard_pareto(alpha, beta, t_max=100, n_points=200)

            # d) Hitung Fungsi Intensitas (lamda)
            # --- LOGIKA OPSI 1 ---
            # Tanggal referensi diambil langsung dari data paling awal yang ada di file.
            start_for_t_calc = class_data["tanggal_klaim_diajukan"].min()
            
            # Panggil fungsi intensitas dan kirimkan 'train_data' ke dalamnya
            lamda, t1, t2, np_t_value = intensitas_pareto(
                global_start=start_for_t_calc,
                start_date=analysis_start,
                end_date=analysis_end,
                alpha=alpha,
                full_data=class_data
            )

                        # --- TAMBAHKAN BARIS PRINT DI SINI ---
            print(f"--- DEBUG INFO UNTUK TIPE: {tipe} ---")
            print(f"t1: {t1}, t2: {t2}, Np_t: {np_t_value}")
            print("--------------------------------------")


            if lamda is None:
                st.warning(f"Tidak dapat menghitung fungsi intensitas untuk rentang tanggal yang dipilih pada tipe {tipe}.")
                continue

            # e) Hitung Ekspektasi dan Variansi
            mean_nt, std_nt, var_nt = eks_var_pareto(lamda)

            # f) Kumpulkan semua hasil
            results = [{
                "Distribusi":         "Pareto",
                "Parameter":          param_str,
                "Kolmogorov-Smirnov": ks_stat,
                "Critical Value":     crit,
                "H0 Ditolak":         "Ya" if ks_stat > crit else "Tidak",
                "Histogram":          hist_buf,
                "Hazard":             hazard_values,
                "Fungsi Hazard":      hazard_buf,
                "Fungsi Intensitas":  lamda,
                "Ekspektasi":         mean_nt,
                "Variansi":           var_nt,
                "Standar Deviasi":    std_nt,
            }]

            all_results[tipe] = results

        except Exception as e:
            st.error(f"Gagal analisis Pareto untuk {tipe}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return all_results

def estimasi_weibull_3p(data):
    """
    Estimasi distribusi Weibull Min 3 parameter (α, γ, β).
    Fungsi ini ditambahkan karena sebelumnya hilang.
    """
    data = data[data > 0]
    # Fit 3 parameter: shape (alpha), loc (gamma), scale (beta)
    alpha, gamma_val, beta = stats.weibull_min.fit(data)
    params = (alpha, gamma_val, beta)
    param_str = f"α={alpha:.4f}, γ={gamma_val:.4f}, β={beta:.4f}"
    
    # Plotting
    x = np.linspace(np.percentile(data, 1), np.percentile(data, 99), 1000)
    y = stats.weibull_min.pdf(x, alpha, loc=gamma_val, scale=beta)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins="auto", density=True, alpha=0.6, color="lightcoral", edgecolor="black")
    ax.plot(x, y, "r-", lw=2, label="Weibull 3p")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)

    return params, buffer, param_str

def estimasi_weibull_2p(data):
    """
    Estimasi distribusi Weibull Min 2 parameter (α, β) dengan γ=0.
    """
    data = data[data > 0]
    # Paksa loc (gamma) menjadi 0
    alpha, gamma_val, beta = stats.weibull_min.fit(data, floc=0)
    params = (alpha, beta)
    param_str = f"α={alpha:.4f}, β={beta:.4f}"
    
    # Plotting
    x = np.linspace(np.percentile(data, 1), np.percentile(data, 99), 1000)
    y = stats.weibull_min.pdf(x, alpha, loc=0, scale=beta)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(data, bins="auto", density=True, alpha=0.6, color="skyblue", edgecolor="black")
    ax.plot(x, y, "g-", lw=2, label="Weibull 2p")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)

    return params, buffer, param_str

# --- FUNGSI UJI STATISTIK ---

def kolmogorov_smirnov_weibull_3p(data, params):
    alpha, gamma_val, beta = params
    data = data[data > 0]
    frozen = weibull_min(alpha, loc=gamma_val, scale=beta)
    ks_stat, ks_pval = kstest(data, frozen.cdf)
    crit_val = 1.36 / np.sqrt(len(data))
    return ks_stat, ks_pval, crit_val

def kolmogorov_smirnov_weibull_2p(data, params):
    alpha, beta = params
    data = data[data > 0]
    frozen = weibull_min(alpha, loc=0, scale=beta)
    ks_stat, ks_pval = kstest(data, frozen.cdf)
    crit_val = 1.36 / np.sqrt(len(data))
    return ks_stat, ks_pval, crit_val

# --- FUNGSI MEAN, VAR, STD ---

def mean_var_std_weibull_3p(params):
    alpha, gamma_val, beta = params
    mean = gamma_val + beta * gamma(1 + 1/alpha)
    var  = beta**2 * (gamma(1 + 2/alpha) - (gamma(1 + 1/alpha))**2)
    std  = np.sqrt(var)
    return mean, var, std

def mean_var_std_weibull_2p(params):
    alpha, beta = params
    mean = beta * gamma(1 + 1/alpha)
    var  = beta**2 * (gamma(1 + 2/alpha) - (gamma(1 + 1/alpha))**2)
    std  = np.sqrt(var)
    return mean, var, std

# --- FUNGSI UTAMA ANALISIS BESAR KLAIM ---

def analisis_besar_klaim(data, selected_tipe=None):
    data = data.copy()
    data["tanggal_klaim_diajukan"] = pd.to_datetime(
        data["tanggal_klaim_diajukan"], errors="coerce"
    )

    if selected_tipe is not None:
        tipe_list = [selected_tipe]
    else:
        tipe_list = data["tipe_klasifikasi"].unique()

    all_results2 = {}
    for tipe in tipe_list:
        class_data = data[data["tipe_klasifikasi"] == tipe].sort_values(by="tanggal_klaim_diajukan")
        if class_data.empty:
            st.warning(f"Tidak ada data untuk tipe {tipe}")
            continue

        class_data["bulan"] = class_data["tanggal_klaim_diajukan"].dt.to_period("M")
        bulan_unik    = class_data["bulan"].unique()
        n_bulan_train = max(1, int(0.8 * len(bulan_unik)))
        bulan_train   = bulan_unik[:n_bulan_train]
        train_data    = class_data[class_data["bulan"].isin(bulan_train)]

        # PERBAIKAN: Ambil sampel dari kolom "besar_klaim"
        sampel = train_data["besar_klaim"].dropna().astype(float)
        sampel = sampel[sampel > 0] # Filter nilai non-positif
        
        if len(sampel) < 10:
            st.warning(f"Data training untuk tipe {tipe} terlalu sedikit ({len(sampel)} obs)")
            continue
        
        tipe_results = []
        try:
            # --- Analisis Weibull 3P ---
            params_3p, buffer_3p, param_str_3p = estimasi_weibull_3p(sampel)
            ks_stat_3p, _, crit_3p = kolmogorov_smirnov_weibull_3p(sampel, params_3p)
            mean_3p, var_3p, std_3p = mean_var_std_weibull_3p(params_3p)
            result_3p = {
                "Distribusi": "Weibull Min (3P)", "Parameter": param_str_3p, "Histogram": buffer_3p,
                "Kolmogorov-Smirnov": ks_stat_3p, "Critical Value": crit_3p, "H0 Ditolak": 'Ya' if ks_stat_3p > crit_3p else 'Tidak',
                "Ekspektasi": mean_3p, "Variansi": var_3p, "Standar Deviasi": std_3p
            }
            tipe_results.append(result_3p)

            # --- Analisis Weibull 2P ---
            params_2p, buffer_2p, param_str_2p = estimasi_weibull_2p(sampel)
            ks_stat_2p, _, crit_2p = kolmogorov_smirnov_weibull_2p(sampel, params_2p)
            mean_2p, var_2p, std_2p = mean_var_std_weibull_2p(params_2p)
            result_2p = {
                "Distribusi": "Weibull Min (2P)", "Parameter": param_str_2p, "Histogram": buffer_2p,
                "Kolmogorov-Smirnov": ks_stat_2p, "Critical Value": crit_2p, "H0 Ditolak": 'Ya' if ks_stat_2p > crit_2p else 'Tidak',
                "Ekspektasi": mean_2p, "Variansi": var_2p, "Standar Deviasi": std_2p
            }
            tipe_results.append(result_2p)

        except Exception as e:
            st.error(f"Gagal melakukan analisis besar klaim untuk tipe {tipe}: {e}")
            continue # Lanjutkan ke tipe berikutnya jika ada error

        # PERBAIKAN: Simpan hasil ke dictionary utama
        if tipe_results:
            all_results2[tipe] = tipe_results

    return all_results2