import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Forecasting Produksi Padi Jawa Barat",
    page_icon="🌾",
    layout="wide"
)

# Fungsi helper
@st.cache_data
def load_data():
    """Load dan preprocessing data"""
    df = pd.read_csv('produksi_padi_jabar.csv', delimiter=';')
    bulan_map = {'JANUARI': 1, 'FEBRUARI': 2, 'MARET': 3, 'APRIL': 4, 'MEI': 5, 'JUNI': 6,
                 'JULI': 7, 'AGUSTUS': 8, 'SEPTEMBER': 9, 'OKTOBER': 10, 'NOVEMBER': 11, 'DESEMBER': 12}
    df['date'] = pd.to_datetime(df['tahun'].astype(str) + '-' + df['bulan'].map(bulan_map).astype(str) + '-01')
    df_final = df[['date', 'nama_kabupaten_kota', 'produksi_padi']].sort_values('date')
    return df_final

def hitung_mape_kustom(y_true, y_pred, threshold=10):
    """Calculate MAPE excluding values below threshold"""
    mask = np.array(y_true) > threshold
    if not mask.sum(): 
        return None
    y_true_filtered = np.array(y_true)[mask]
    y_pred_filtered = np.array(y_pred)[mask]
    return np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered))

def auto_tune_holtwinters(series_data, use_log=False):
    """Find best Holt-Winters configuration"""
    train, test = series_data[:-12], series_data[-12:]
    train_prepared = np.log(train.replace(0, 0.01)) if use_log else train.replace(0, 0.01)
    
    best_mape = float('inf')
    best_config = ""
    
    configs = [
        {'t': 'add', 's': 'add', 'd': False},
        {'t': 'add', 's': 'mul', 'd': False},
        {'t': 'add', 's': 'add', 'd': True},
        {'t': 'mul', 's': 'mul', 'd': True}
    ]
    
    for cfg in configs:
        try:
            model = ExponentialSmoothing(
                train_prepared, 
                trend=cfg['t'], 
                seasonal=cfg['s'],
                damped_trend=cfg['d'], 
                seasonal_periods=12
            ).fit(optimized=True)
            
            forecast = np.exp(model.forecast(len(test))) if use_log else model.forecast(len(test))
            mape = hitung_mape_kustom(test, forecast)
            
            if mape and mape < best_mape:
                best_mape = mape
                best_config = f"T={cfg['t']}, S={cfg['s']}, D={cfg['d']}, Log={use_log}"
                
        except Exception as e:
            continue
            
    return best_mape, best_config

def get_future_forecast(ts):
    """Generate forecast untuk 12 bulan ke depan"""
    mape_n, cfg_n = auto_tune_holtwinters(ts, use_log=False)
    mape_l, cfg_l = auto_tune_holtwinters(ts, use_log=True)
    use_log = mape_l < mape_n
    cfg = cfg_l if use_log else cfg_n
    parts = cfg.split(', ')
    trend, seasonal = parts[0].split('=')[1], parts[1].split('=')[1]
    damped = 'true' in parts[2].lower()
    ts = ts.replace(0, 0.01)
    ts_model = np.log(ts) if use_log else ts
    model = ExponentialSmoothing(ts_model, trend=trend, seasonal=seasonal, damped_trend=damped, seasonal_periods=12).fit(optimized=True)
    forecast = np.exp(model.forecast(12)) if use_log else model.forecast(12)
    return forecast, use_log, (mape_l if use_log else mape_n), model

def categorize(val, bl, bh):
    """Kategorisasi hasil panen"""
    if val <= 10: return "Paceklik"
    elif val < bl: return "Panen Rendah"
    elif val < bh: return "Panen Sedang"
    elif val > (bh * 1.5): return "Panen Raya Ekstrem"
    else: return "Panen Raya"

# Load data
df_final = load_data()

# Header
st.title("🌾 Forecasting Produksi Padi Jawa Barat")
st.markdown("### Prediksi Produksi Padi Menggunakan Holt-Winters Exponential Smoothing")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Pengaturan")
menu = st.sidebar.radio("Pilih Menu:", ["📊 Dashboard", "🔮 Forecasting", "📈 Analisis Data"])

# Menu Dashboard
if menu == "📊 Dashboard":
    st.header("Dashboard Produksi Padi Jawa Barat")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_produksi = df_final['produksi_padi'].sum()
        st.metric("Total Produksi (2018-2023)", f"{total_produksi:,.0f} Ton")
    with col2:
        avg_produksi = df_final['produksi_padi'].mean()
        st.metric("Rata-rata Produksi/Bulan", f"{avg_produksi:,.0f} Ton")
    with col3:
        jumlah_kota = df_final['nama_kabupaten_kota'].nunique()
        st.metric("Jumlah Kabupaten/Kota", jumlah_kota)
    
    st.markdown("---")
    
    # Tren Total Produksi
    st.subheader("📈 Tren Total Produksi Padi Jawa Barat (2018-2023)")
    df_agg = df_final.groupby('date')['produksi_padi'].sum()
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_agg.index, df_agg.values, color='green', linewidth=2, marker='o')
    ax1.set_xlabel('Tahun')
    ax1.set_ylabel('Total Produksi (Ton)')
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)
    
    # Top 5 Kabupaten/Kota
    st.subheader("🏆 Top 5 Kabupaten/Kota dengan Produksi Tertinggi")
    top_5 = df_final.groupby('nama_kabupaten_kota')['produksi_padi'].sum().nlargest(5)
    
    col1, col2 = st.columns(2)
    with col1:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        top_5.plot(kind='barh', ax=ax2, color='skyblue')
        ax2.set_xlabel('Total Produksi (Ton)')
        ax2.set_ylabel('Kabupaten/Kota')
        st.pyplot(fig2)
    
    with col2:
        st.dataframe(
            pd.DataFrame({
                'Kabupaten/Kota': top_5.index,
                'Total Produksi (Ton)': top_5.values.astype(int)
            }).reset_index(drop=True),
            use_container_width=True
        )

# Menu Forecasting
elif menu == "🔮 Forecasting":
    st.header("Prediksi Produksi Padi 12 Bulan Ke Depan")
    
    # Pilih Kabupaten/Kota
    kota_list = sorted(df_final['nama_kabupaten_kota'].unique())
    selected_kota = st.selectbox("Pilih Kabupaten/Kota:", kota_list, index=kota_list.index('KABUPATEN BANDUNG') if 'KABUPATEN BANDUNG' in kota_list else 0)
    
    if st.button("🚀 Jalankan Forecasting", type="primary"):
        with st.spinner("Memproses forecasting..."):
            # Ambil data time series
            ts_kota = df_final[df_final['nama_kabupaten_kota'] == selected_kota].set_index('date')['produksi_padi'].asfreq('MS').fillna(0)
            
            # Generate forecast
            forecast_2024, use_log, error_val, model = get_future_forecast(ts_kota)
            forecast_2024[forecast_2024 < 0] = 0
            
            # Metrics
            st.success("✅ Forecasting berhasil!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Akurasi Model (MAPE)", f"{(1-error_val)*100:.2f}%")
            with col2:
                mae = mean_absolute_error(ts_kota[-12:], forecast_2024)
                st.metric("MAE", f"{mae:,.0f} Ton")
            with col3:
                rmse = np.sqrt(mean_squared_error(ts_kota[-12:], forecast_2024))
                st.metric("RMSE", f"{rmse:,.0f} Ton")
            
            st.markdown("---")
            
            # Visualisasi
            st.subheader("📊 Visualisasi Hasil Forecasting")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(ts_kota.index, ts_kota.values, label='Data Historis (2018-2023)', color='black', linewidth=2)
            ax.plot(forecast_2024.index, forecast_2024.values, label='Forecast 2024', color='red', linestyle='--', marker='o', linewidth=2)
            ax.set_title(f'Forecast Produksi Padi: {selected_kota}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Tahun')
            ax.set_ylabel('Produksi (Ton)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Tabel Prediksi
            st.subheader("📋 Detail Prediksi per Bulan")
            b_low, b_high = ts_kota.quantile(0.25), ts_kota.quantile(0.75)
            df_pred = pd.DataFrame({
                'Bulan': forecast_2024.index.strftime('%B %Y'),
                'Prediksi (Ton)': forecast_2024.values.round(0).astype(int)
            })
            df_pred['Kategori'] = df_pred['Prediksi (Ton)'].apply(lambda x: categorize(x, b_low, b_high))
            
            st.dataframe(df_pred, use_container_width=True)
            
            # Download hasil
            csv = df_pred.to_csv(index=False)
            st.download_button(
                label="📥 Download Hasil Prediksi (CSV)",
                data=csv,
                file_name=f"prediksi_{selected_kota.replace(' ', '_')}.csv",
                mime="text/csv"
            )

# Menu Analisis Data
elif menu == "📈 Analisis Data":
    st.header("Analisis Data Produksi Padi")
    
    # Filter Kabupaten/Kota
    kota_filter = st.multiselect(
        "Pilih Kabupaten/Kota untuk Perbandingan:",
        df_final['nama_kabupaten_kota'].unique(),
        default=df_final.groupby('nama_kabupaten_kota')['produksi_padi'].sum().nlargest(3).index.tolist()
    )
    
    if kota_filter:
        df_filtered = df_final[df_final['nama_kabupaten_kota'].isin(kota_filter)]
        
        # Tren per Kota
        st.subheader("📊 Tren Produksi per Kabupaten/Kota")
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        for kota in kota_filter:
            kota_data = df_filtered[df_filtered['nama_kabupaten_kota'] == kota]
            ax1.plot(kota_data['date'], kota_data['produksi_padi'], label=kota, linewidth=2, marker='o', markersize=3)
        ax1.set_xlabel('Tahun')
        ax1.set_ylabel('Produksi (Ton)')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        st.pyplot(fig1)
        
        # Statistik Deskriptif
        st.subheader("📊 Statistik Deskriptif")
        stats = df_filtered.groupby('nama_kabupaten_kota')['produksi_padi'].agg([
            ('Total', 'sum'),
            ('Rata-rata', 'mean'),
            ('Median', 'median'),
            ('Min', 'min'),
            ('Max', 'max'),
            ('Std Dev', 'std')
        ]).round(0).astype(int)
        st.dataframe(stats, use_container_width=True)
        
        # Distribusi
        st.subheader("📊 Distribusi Produksi")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        for kota in kota_filter:
            kota_data = df_filtered[df_filtered['nama_kabupaten_kota'] == kota]['produksi_padi']
            ax2.hist(kota_data, alpha=0.5, label=kota, bins=30)
        ax2.set_xlabel('Produksi (Ton)')
        ax2.set_ylabel('Frekuensi')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2)
    else:
        st.warning("⚠️ Silakan pilih minimal satu kabupaten/kota untuk analisis")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🌾 Forecasting Produksi Padi Jawa Barat | Data: 2018-2023</p>
        <p>Model: Holt-Winters Exponential Smoothing</p>
    </div>
    """,
    unsafe_allow_html=True
)
