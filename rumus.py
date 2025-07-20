import numpy as np
import pandas as pd
import streamlit as st
import sklearn as sk
from scipy import stats
from scipy.stats import kstest
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

# --- Struktur spesifikasi distribusi ---
DistSpec = namedtuple("DistSpec", ["dist", "param_names"])

distributions = {
    # Distribusi baru dengan penamaan simbol sesuai referensi:
    "Alpha":                DistSpec(stats.alpha,          ["α", "μ", "σ"]),        # a->alpha, loc->mu, scale->sigma
    "Beta Prime":      DistSpec(stats.betaprime,  ["α","β","μ","σ"]),
    "Burr":                 DistSpec(stats.burr,           ["k", "α", "γ", "β"]),  # c->k, d->alpha, loc->gamma, scale->beta
    "Burr Type XII":   DistSpec(stats.burr12,     ["k", "α", "γ", "β"]),
    "Exponential":          DistSpec(stats.expon,          ["γ", "λ"]),               # loc=gamma, inv scale=lambda
    "Fatigue Life":    DistSpec(stats.fatiguelife, ["α", "γ", "β"]),               # a->alpha, loc->gamma, scale->beta
    "Fisk":            DistSpec(stats.fisk,       ["α","μ","σ"]),
    "Gamma":                DistSpec(stats.gamma,          ["α", "γ", "β"]),         # a->alpha, loc->gamma, scale->beta
    "Generalized Extreme Value": DistSpec(stats.genextreme, ["k", "μ", "σ"]),   # c->k, loc->mu, scale->sigma
    "Generalized Gamma":    DistSpec(stats.gengamma,       ["k", "α", "γ", "β"]), # a->k, c->alpha, loc->gamma, scale->beta
    "Generalized Pareto":   DistSpec(stats.genpareto,      ["k", "μ", "σ"]),         # c->k, loc->mu, scale->sigma
    "Gumbel Right":         DistSpec(stats.gumbel_r,       ["μ", "σ"]),             # loc=mu, scale=sigma
    "Gompertz":        DistSpec(stats.gompertz,   ["α","β","μ"]),
    "Inverse Gamma":        DistSpec(stats.invgamma,       ["α", "μ", "σ"]),          # shape->alpha, loc->mu, scale->sigma
    "Inverse Gaussian":     DistSpec(stats.invgauss,       ["λ", "μ", "γ"]),        # shape invgauss->lambda, scale->mu, loc->gamma
    "Laplace":         DistSpec(stats.laplace,    ["μ","σ"]),               # loc=mu, scale=sigma
    "Levy":                 DistSpec(stats.levy,           ["γ", "σ"]),               # loc->gamma, scale->sigma
    "Log-Gamma":            DistSpec(stats.loggamma,       ["α", "β"]),               # c->alpha, scale->beta
    "Log-Normal":           DistSpec(stats.lognorm,        ["σ", "γ", "μ"]),         # s->sigma, loc->gamma, scale->mu
    "Logistic":        DistSpec(stats.logistic,   ["μ","σ"]),
    "Lomax":           DistSpec(stats.lomax,      ["α","σ"]),
    "Pareto":               DistSpec(stats.pareto,         ["α", "β"]),               # shape=alpha, scale=beta (loc=0)
    "Rayleigh":             DistSpec(stats.rayleigh,       ["γ", "σ"]),               # loc->gamma, scale->sigma
    "Wald":            DistSpec(stats.wald,       ["μ","λ","γ"]),
    "Weibull Min":          DistSpec(stats.weibull_min,    ["α", "γ", "β"]),        # c->alpha, loc->gamma, scale->beta
    "Weibull Max":          DistSpec(stats.weibull_max,    ["α", "γ", "β"]),        # c->alpha, loc->gamma, scale->beta
}

def estimasi_parameter(data, dist_name):
    """
    Fit distribusi dan tampilkan histogram dengan overlay PDF dalam satu fungsi,
    meniru struktur estimate_distribution Anda.
    
    Returns:
        params: tuple hasil fit MLE
        buffer: BytesIO berisi PNG plot
        param_str: ringkasan 'nama=nilai' dari parameter
    """
    # Ambil spec
    spec = distributions[dist_name]
    dist = spec.dist
    params = dist.fit(data)
    param_names = spec.param_names

    # Filter data agar hanya positif untuk distribusi tertentu
    if dist_name in ["Pareto", "Burr", "Log-Normal", "Generalized Gamma"]:
        data = data[data > 0]
    
    # Gunakan rentang persentil 10%-90% untuk menghindari outlier mendominasi
    x = np.linspace(np.percentile(data, 10), np.percentile(data, 90), 1000)
    y = dist.pdf(x, *params)
    
    # Pastikan nilai y tidak mengandung NaN atau Inf
    y = np.nan_to_num(y, nan=0.0, posinf=np.max(y[np.isfinite(y)]), neginf=0.0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Terapkan transformasi log jika diperlukan
    if dist_name in ["Pareto", "Burr", "Log-Normal", "Generalized Gamma"]:
        data = np.log1p(data)
        x = np.log1p(x)
        ax.set_xscale("log")
        ax.set_yscale("log")  # Terapkan skala log pada sumbu Y untuk distribusi skewed

    # Menentukan jumlah bins secara otomatis
    bin_edges = np.histogram_bin_edges(data, bins="auto")
    ax.hist(data, bins=bin_edges, density=True, alpha=0.6, label="Data", color='skyblue', edgecolor='black')
    ax.plot(x, y, label=f"PDF {dist_name}", color="red", linewidth=2)
    ax.set_title(f"Estimasi Distribusi {dist_name}", fontsize=12)
    ax.set_xlabel("Nilai")
    ax.set_ylabel("Kepadatan")
    
    # 1) Hitung tinggi histogram:
    counts, _ = np.histogram(data, bins=bin_edges, density=True)
    max_hist = counts.max()
    # 2) Cari maksimum PDF:
    max_pdf = np.max(y)
    # 3) Atur ulang sumbu Y:
    ymax = max(max_hist, max_pdf)
    ax.set_ylim(1e-5, ymax * 1.1)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    
    param_str = ", ".join(f"{name}={value:.4f}" for name, value in zip(param_names, params))
    
    return params, buffer, param_str

def kolmogorov_smirnov(data, dist_name, params):
    """
    Melakukan uji Kolmogorov–Smirnov pada `data` 
    terhadap distribusi dist_name dengan parameter `params`.
    `params` di sini adalah tuple hasil dist.fit(data).
    """
    # 1) Ambil spec dan objek distribusi
    spec = distributions[dist_name]
    dist = spec.dist

    # 2) Tentukan berapa banyak parameter shape
    #    (anggap selalu ada 2 terakhir: loc & scale)
    n_shape = len(params) - 2
    shape_args = params[:n_shape]
    loc, scale = params[n_shape], params[n_shape + 1]

    # 3) Buat “frozen” distribution dengan shape, loc, scale
    frozen = dist(*shape_args, loc=loc, scale=scale)

    # 4) Hitung statistik KS
    ks_stat, ks_pval = kstest(data, frozen.cdf)

    # 5) Hitung critical value (α=0.05) untuk sampel n
    crit_val = 1.36 / np.sqrt(len(data))

    return ks_stat, ks_pval, crit_val

def estimasi_hazard(data_interarrival, dist_name):
    """
    Fit distribusi ke data_interarrival,
    kembalikan fungsi hazard h(t) yang vectorized.
    """
    spec    = distributions[dist_name]
    dist    = spec.dist
    params  = dist.fit(data_interarrival)
    n_shape = len(params) - 2
    frozen  = dist(
        *params[:n_shape],
        loc=params[n_shape],
        scale=params[n_shape+1]
    )

    def hazard_fn(t):
        # pdf / [1 - cdf], clip untuk hindari div/0
        num   = frozen.pdf(t)
        denom = np.clip(1 - frozen.cdf(t), a_min=1e-8, a_max=None)
        hazard = num / denom
        return hazard

    return hazard_fn

def plot_hazard(data, hazard_fn, dist_name):
    # data = np.asarray(data, dtype=float)
    x_plot = np.linspace(data.min(), data.max(), 1000)
    hz_plot = hazard_fn(x_plot)

    # Plot hanya kurva hazard function
    fig, ax = plt.subplots(figsize=(8, 5))
    bin_edges = np.histogram_bin_edges(data, bins="auto")
    ax.hist(data, bins=bin_edges, density=True, alpha=0.6, label="Data", color='skyblue', edgecolor='black')
    ax.plot(x_plot, hz_plot, label="Fungsi Hazard", linewidth=2)
    if dist_name in ["Weibull Max", "Pareto", "Log-Normal", "Generalized Pareto", "Burr", "Generalized Gamma"]:
        ax.set_yscale('log')

    # Mengubah judul dan font size, label plot
    ax.set_title(f"Fungsi Hazard Distribusi {dist_name}", fontsize=14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Hazard rate")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Simpan gambar ke buffer
    buffer = BytesIO()
    plt.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    plt.close(fig)

    return buffer

def menghitung_hari_kerja(start_date, end_date, include_saturday: bool = True):
    """
    Kembalikan objek CustomBusinessDay yang:
      • weekmask = Mon–Sat (jika include_saturday=True) atau Mon–Fri
      • mengecualikan libur nasional Indonesia tahun start_date…end_date
    """
    years    = range(start_date.year, end_date.year + 1)
    id_hols  = holidays.Indonesia(years=years)
    weekmask = "Mon Tue Wed Thu Fri Sat" if include_saturday else "Mon Tue Wed Thu Fri"
    return CustomBusinessDay(weekmask=weekmask, holidays=id_hols)


def menghitung_batas(
    data,
    date_col: str,
    analysis_start,   # str atau datetime
    analysis_end,     # str atau datetime
    include_saturday: bool = False
):
    """
    Hitung t1, t2 sebagai:
      t1 = hari kerja ke‑berapa (1‑indexed) sampai analysis_start
      t2 = hari kerja ke‑berapa         sampai analysis_end
    menggunakan kalender Indonesia + (opsional) Sabtu kerja.
    """
    # 1) Normalize dan parse
    #first = pd.to_datetime(data[date_col].min()).normalize()
    start = pd.to_datetime(analysis_start).normalize()
    end   = pd.to_datetime(analysis_end).normalize()

    # 2) Buat freq business day custom
    cbd   = menghitung_hari_kerja(start, end, include_saturday)

    # 3) Hitung jumlah hari kerja inklusif
    t1 = 1
    t2 = len(pd.date_range(start=start, end=end,   freq=cbd))
    return t1, t2

def fungsi_intensitas(
    data, hazard_fn,
    date_col: str,
    analysis_start,  # str atau datetime
    analysis_end,    # str atau datetime
    include_saturday: bool = False
) -> float:
    """
    λ = Np * ∫[t1..t2] h(t) dt
      • Np = jumlah klaim di [start..end] dari seluruh data kelas
      • h(t) di‑fit dari seluruh sampel lama_pengajuan_klaim pada train_data
        (Anda bebas memisahkan train_data vs data di level atas)
      • t1, t2 dihitung dengan menghitung_batas (termasuk libur nasional)
    """
    # — 1) Hitung t1, t2 —
    t1, t2 = menghitung_batas(
        data, date_col, analysis_start, analysis_end,
        include_saturday=include_saturday
    )

    # — 2) Hitung Np (semua klaim di periode analisis) —
    start = pd.to_datetime(analysis_start).normalize()
    end   = pd.to_datetime(analysis_end).normalize()
    mask  = (data[date_col] >= start) & (data[date_col] <= end)
    np_t    = int(mask.sum())
    if np_t == 0:
        return 0.0
    
    # Debug: tampilkan t1, t2, Np
    # st.write(f"🔍 {hazard_fn.__name__ if hasattr(hazard_fn,'__name__') else ''}  t1={t1}, t2={t2}, Np={np_t}")

    # — 5) Integrasi adaptif h(t) dari t1..t2 —
    I, err = quad(hazard_fn, t1, t2, limit=200)
    lambda_val = np_t * I
    # Debug: tampilkan hasil integral dan lambda
    # st.write(f"   ∫ₜ₁^{t2} h(t)dt = {I:.6e}  (err≈{err:.6e})")
    # st.write(f"   λ = Np×I = {np_t}×{I:.6e} = {lambda_val:.6e}")

    return lambda_val

def eks_var_banyak_klaim(data, hazard_fn, date_col: str, analysis_start, analysis_end, include_saturday: bool = False):
    """
    Menghitung ekspektasi E[N(t)] dan variansi Var[N(t)]
    untuk proses Poisson terpancung, dengan λ(t) dihitung
    oleh fungsi_intensitas.

    Returns:
      (mean, variance) sebagai tuple float.
    """
    # 1) Dapatkan λ(t)
    lambda_val = fungsi_intensitas(data, hazard_fn, date_col, analysis_start, 
                                   analysis_end, include_saturday=include_saturday)
    # Jika tidak ada klaim, langsung return 0,0
    if lambda_val == 0:
        return 0.0, 0.0
    # 2) Hitung momen Poisson terpancung
    p0    = np.exp(-lambda_val)           # P(N=0)
    denom = 1 - p0                        # normalisasi (k>0)
    mean_1  = lambda_val / denom            # E[N(t)]
    eks2_1 = (lambda_val**2 + lambda_val) / denom  # E[N^2]
    var_1   = eks2_1 - mean_1**2                  # Var[N(t)]
    std_1   = np.sqrt(var_1)                       # SD[N(t)]

    return mean_1, var_1, std_1

def analisis_banyak_klaim(data: pd.DataFrame, selected_kelas=None, analysis_start=None, analysis_end=None):
    # — 0) Parse tanggal analisis jadi Timestamp —
    analysis_start = pd.to_datetime(analysis_start).normalize()
    analysis_end   = pd.to_datetime(analysis_end).normalize()

    # — 1) Siapkan data & kolom datetime —
    df = data.copy()
    df["tanggal_klaim_diajukan"] = pd.to_datetime(
        df["tanggal_klaim_diajukan"], errors="coerce"
    ).dt.normalize()

    # — 2) Tentukan list kelas yang mau diproses —
    if selected_kelas:
        kelas_list = [selected_kelas]
    else:
        kelas_list = df["kelas"].unique()

    all_results = {}

    for kelas in kelas_list:
        class_data = df[df["kelas"] == kelas].sort_values("tanggal_klaim_diajukan")
        if class_data.empty:
            st.warning(f"Tidak ada data untuk Kelas {kelas}")
            continue

        # — 3) Split train/test berdasarkan bulan —
        class_data["bulan"] = class_data["tanggal_klaim_diajukan"].dt.to_period("M")
        bulan_unik    = class_data["bulan"].unique()
        n_bulan_train = max(1, int(0.8 * len(bulan_unik)))
        bulan_train   = bulan_unik[:n_bulan_train]
        train_data    = class_data[class_data["bulan"].isin(bulan_train)]

        # — 4) Siapkan sampel interarrival dari train_data —
        sampel = train_data["lama_pengajuan_klaim"].dropna().astype(float)
        sampel = sampel[sampel > 0].values
        if len(sampel) < 10:
            st.warning(f"Data training terlalu sedikit untuk Kelas {kelas} ({len(sampel)} obs)")
            continue

        # — 5) Loop distribusi —
        results = []
        dist_name = "Pareto"
        spec = distributions[dist_name]
        try:
            sampel_train = (train_data["lama_pengajuan_klaim"].dropna().astype(float).values)
            # a) Estimasi parameter & KS
            params, hist_buf, param_str = estimasi_parameter(sampel_train, dist_name)
            ks_stat, ks_pval, crit       = kolmogorov_smirnov(sampel_train, dist_name, params)

            # b) Hazard & plot
            # 2) build sekali hazard analytic dari sampeL_train
            frozen = spec.dist(*params[:-2], loc=params[-2], scale=params[-1])
            hazard_fn = lambda t: frozen.pdf(t)/np.clip(1 - frozen.cdf(t), 1e-8, None)

            hazard_buf  = plot_hazard(sampel_train, hazard_fn, dist_name)

            # c) Intensity dari seluruh class_data
            lambda_val = fungsi_intensitas(class_data, hazard_fn, date_col="tanggal_klaim_diajukan",
                analysis_start=analysis_start, analysis_end=analysis_end,
                include_saturday=False   # atau True jika Sabtu dihitung kerja
            )
            mean_1, var_1, std_1 = eks_var_banyak_klaim(class_data, hazard_fn, date_col="tanggal_klaim_diajukan",
                analysis_start=analysis_start, analysis_end=analysis_end,
                include_saturday=False   # atau True jika Sabtu dihitung kerja
            )

            results.append({
                "Distribusi":         dist_name,
                "Parameter":          param_str,
                "Kolmogorov-Smirnov": ks_stat,
                "Critical Value":     crit,
                "H0 Ditolak":         "Ya" if ks_stat > crit else "Tidak",
                "Histogram":          hist_buf,
                "Fungsi Hazard":      hazard_buf,
                "Fungsi Intensitas":  lambda_val,
                "Ekspektasi":         mean_1,
                "Variansi":           var_1,
                "Standar Deviasi":    std_1
            })

        except Exception as e:
            st.error(f"Gagal memproses {dist_name} di Kelas {kelas}: {e}")

        all_results[kelas] = results

    return all_results

def estimasi_parameter2(data, dist_name):
    """
    Fit distribusi dan tampilkan histogram dengan overlay PDF dalam satu fungsi,
    meniru struktur estimate_distribution Anda.
    
    Returns:
        params: tuple hasil fit MLE
        buffer: BytesIO berisi PNG plot
        param_str: ringkasan 'nama=nilai' dari parameter
    """
    # Ambil spec
    spec = distributions[dist_name]
    dist = spec.dist
    params = dist.fit(data)
    param_names = spec.param_names

    # Filter data agar hanya positif untuk distribusi tertentu
    # if dist_name in ["Weibull Max", "Truncated Normal", "Nakagami"]:
    #     data = data[data > 0]
    
    # Gunakan rentang persentil 10%-90% untuk menghindari outlier mendominasi
    x = np.linspace(np.percentile(data, 10), np.percentile(data, 90), 1000)
    y = dist.pdf(x, *params)
    
    # Pastikan nilai y tidak mengandung NaN atau Inf
    y = np.nan_to_num(y, nan=0.0, posinf=np.max(y[np.isfinite(y)]), neginf=0.0)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Terapkan transformasi log jika diperlukan
    # if dist_name in ["Weibull Max", "Truncated Normal", "Nakagami"]:
    #     data = np.log1p(data)
    #     x = np.log1p(x)
    #     ax.set_xscale("log")
    #     ax.set_yscale("log")  # Terapkan skala log pada sumbu Y untuk distribusi skewed

    # Menentukan jumlah bins secara otomatis
    bin_edges = np.histogram_bin_edges(data, bins="auto")
    ax.hist(data, bins=bin_edges, density=True, alpha=0.6, label="Data", color='skyblue', edgecolor='black')
    ax.plot(x, y, label=f"PDF {dist_name}", color="red", linewidth=2, zorder=5)
    ax.set_title(f"Estimasi Distribusi {dist_name}", fontsize=12)
    ax.set_xlabel("Nilai")
    ax.set_ylabel("Kepadatan")
    
    # 1) Hitung tinggi histogram:
    counts, _ = np.histogram(data, bins=bin_edges, density=True)
    max_hist = counts.max()
    # 2) Cari maksimum PDF:
    max_pdf = np.max(y)
    # 3) Atur ulang sumbu Y:
    ymax = max(max_hist, max_pdf)
    ax.set_ylim(0, ymax * 1.1)
    
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close(fig)
    
    param_str = ", ".join(f"{name}={value:.4f}" for name, value in zip(param_names, params))
    
    return params, buffer, param_str

# def eks_var_besar_klaim(data, dist_name):
#     """
#     Menghitung ekspektasi E[X] dan variansi Var[X]
#     untuk distribusi yang diberikan.
#     """
#     # 1) Ambil spec dan objek distribusi
#     spec = distributions[dist_name]
#     dist = spec.dist

#     # 2) Fit distribusi ke data
#     params = dist.fit(data)
    
#     # 3) Hitung ekspektasi dan variansi
#     mean_2 = dist.mean(*params[:-2], loc=params[-2], scale=params[-1])
#     var_2  = dist.var(*params[:-2], loc=params[-2], scale=params[-1])

#     return mean_2, var_2
def eks_var_besar_klaim(data, dist_name):
    """
    Hitung ekspektasi dan variansi, fallback ke data empiris
    jika teoretisnya inf/nan.
    """
    spec = distributions[dist_name]
    dist = spec.dist

    # 1) Fit distribusi
    params = dist.fit(data)

    # 2) Hitung momen teoretis, tangani error
    try:
        mean_t = dist.mean(*params[:-2],
                           loc=params[-2],
                           scale=params[-1])
        var_t  = dist.var(*params[:-2],
                          loc=params[-2],
                          scale=params[-1])
    except Exception:
        mean_t, var_t = np.nan, np.nan

    # 3) Fallback ke data empiris jika tidak finite
    if not np.isfinite(mean_t):
        mean_2 = np.mean(data)
    else:
        mean_2 = mean_t

    if not np.isfinite(var_t):
        var_2 = np.var(data)
    else:
        var_2 = var_t
    
    std_2 = np.sqrt(var_2)

    return mean_2, var_2, std_2

def analisis_besar_klaim(data, selected_kelas=None):
    # 1) Pastikan kolom tanggal sudah datetime
    data = data.copy()
    data["tanggal_klaim_diajukan"] = pd.to_datetime(
        data["tanggal_klaim_diajukan"], errors="coerce"
    )

    # 2) Tentukan kelas mana yang akan diproses
    if selected_kelas is not None:
        kelas_list = [selected_kelas]
    else:
        kelas_list = data["kelas"].unique()

    all_results2 = {}
    for kelas in kelas_list:
        class_data = (
            data[data["kelas"] == kelas]
                .sort_values(by="tanggal_klaim_diajukan")
        )
        if class_data.empty:
            st.warning(f"Tidak ada data untuk Kelas {kelas}")
            continue

        # 3) Buat kolom 'bulan' dan split train/test
        class_data["bulan"] = class_data["tanggal_klaim_diajukan"].dt.to_period("M")
        bulan_unik    = class_data["bulan"].unique()
        n_bulan_train = max(1, int(0.8 * len(bulan_unik)))
        bulan_train   = bulan_unik[:n_bulan_train]
        train_data    = class_data[class_data["bulan"].isin(bulan_train)]

        # 4) Ambil hanya kolom lama_pengajuan_klaim
        sampel = train_data["besar_klaim"].dropna()
        if len(sampel) < 10:
            st.warning(f"Data training terlalu sedikit ({len(sampel)} obs)")
            continue

        # 5) Estimasi untuk setiap distribusi
        results2 = []
        for dist_name, spec in distributions.items():
            try:
                params, buffer, param_str = estimasi_parameter2(sampel, dist_name)
                ks_stat, ks_pval, critical_value = kolmogorov_smirnov(sampel, dist_name, params)
                mean_2, var_2, std_2 = eks_var_besar_klaim(sampel, dist_name)
                results2.append({
                    "Distribusi": dist_name,
                    "Parameter":   param_str,
                    "Histogram":   buffer,
                    "Kolmogorov-Smirnov": ks_stat,
                    "Critical Value": critical_value,
                    "H0 Ditolak": 'Ya' if ks_stat > critical_value else 'Tidak',
                    "Ekspektasi":  mean_2,
                    "Variansi":    var_2,
                    "Standar Deviasi": std_2
                })
            except Exception as e:
                st.error(f"Gagal memproses {dist_name}: {e}")
        all_results2[kelas] = results2

    return all_results2

