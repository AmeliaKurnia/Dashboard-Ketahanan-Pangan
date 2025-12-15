import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import st_folium
import plotly.express as px
import os
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="GeoAI Ketahanan Pangan",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container {padding-top: 1rem; padding-bottom: 2rem; padding-left: 2rem; padding-right: 2rem;}
    div[data-testid="stMetric"] {background-color: #ffffff; border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}
    h1 {color: #2c3e50; margin-bottom: 0.5rem;}
    h2, h3 {color: #34495e;}
    
    /* Styling Tabs - PERBAIKAN: Background transparan/menyatu */
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px; 
        background-color: transparent; /* Ubah dari warna solid ke transparan */
        border: 1px solid #ddd; /* Tambah border tipis agar terlihat rapi */
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd; 
        color: #0d47a1;
        border-color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. KONFIGURASI DATA & METADATA (LENGKAP SESUAI SKRIPSI)
# -----------------------------------------------------------------------------
VAR_MAPPING = {
    "X1": "Indeks Ketahanan Pangan (IKP)",
    "X2": "Produksi Padi",
    "X3": "Produksi Jagung",
    "X4": "Pendapatan per Kapita",
    "X5": "Persentase Pengeluaran per Kapita Sebulan Makanan",
    "X6": "Realisasi Penerima Bansos Pangan",
    "X7": "Harga Komoditas Beras",
    "X8": "Harga Komoditas Jagung",
    "X9": "Akses Air Minum Layak",
    "X10": "Akses Sanitasi Layak",
    "X11": "Prevalensi Balita Wasting",
    "X12": "Prevalensi Balita Underweight",
    "X13": "Kepadatan Penduduk",
    "X14": "Indeks Risiko Bencana"
}

# Metadata Detail (Sumber: Bab 3 Skripsi)
VAR_METADATA = {
    "X1": {
        "Unit": "Skor (0-100)", 
        "Def": "Indikator komposit yang digunakan untuk mengukur kondisi ketahanan pangan suatu wilayah berdasarkan dimensi ketersediaan, akses, dan pemanfaatan pangan (BPN, 2023)."
    },
    "X2": {
        "Unit": "Ton", 
        "Def": "Jumlah total padi yang dipanen, diukur dalam ton gabah kering panen. Dihitung dari luas panen dikali hasil per hektar (BPS, 2024)."
    },
    "X3": {
        "Unit": "Ton", 
        "Def": "Jumlah total jagung yang dipanen dalam satu musim tanam. Mencerminkan output fisik kinerja pertanian jagung (BPS, 2024)."
    },
    "X4": {
        "Unit": "Rupiah", 
        "Def": "Pendapatan rata-rata setiap individu dalam suatu wilayah. Mencerminkan kemampuan ekonomi/daya beli masyarakat terhadap pangan (Eliezer, 2024)."
    },
    "X5": {
        "Unit": "Persen (%)", 
        "Def": "Persentase rata-rata pengeluaran individu per bulan untuk makanan. Semakin tinggi persentasenya, semakin besar beban ekonomi rumah tangga (BPN, 2023)."
    },
    "X6": {
        "Unit": "Keluarga (KPM)", 
        "Def": "Jumlah bantuan sosial yang telah disalurkan dan diterima oleh masyarakat untuk menjaga akses pangan saat terjadi guncangan ekonomi (Dalias & Wisana, 2023)."
    },
    "X7": {
        "Unit": "Rupiah/Kg", 
        "Def": "Harga rata-rata beras kualitas medium di tingkat konsumen. Kestabilan harga beras krusial untuk kepastian akses pangan (Widarso & Djamaluddin, 2024)."
    },
    "X8": {
        "Unit": "Rupiah/Kg", 
        "Def": "Nilai jual jagung di pasar pada periode tertentu. Dipengaruhi oleh kualitas, lokasi, dan kondisi pasar (BPS, 2024)."
    },
    "X9": {
        "Unit": "Persen (%)", 
        "Def": "Persentase penduduk yang menggunakan sumber air minum yang memenuhi syarat teknis dan kesehatan (FAO)."
    },
    "X10": {
        "Unit": "Persen (%)", 
        "Def": "Persentase penduduk yang memiliki akses terhadap fasilitas sanitasi yang aman, layak, dan tidak mencemari lingkungan (FAO, 2024)."
    },
    "X11": {
        "Unit": "Persen (%)", 
        "Def": "Proporsi balita dengan berat badan terlalu rendah dibandingkan tinggi badan (kurus). Menandakan masalah gizi akut jangka pendek (FAO, 2024)."
    },
    "X12": {
        "Unit": "Persen (%)", 
        "Def": "Persentase balita dengan berat badan kurang dari standar usianya (BB/U). Mencerminkan akumulasi masalah gizi kronis dan akut (WHO)."
    },
    "X13": {
        "Unit": "Jiwa/km²", 
        "Def": "Jumlah penduduk per satuan luas wilayah. Tekanan demografis dapat mengganggu stabilitas ketersediaan pangan (FAO)."
    },
    "X14": {
        "Unit": "Skor Indeks", 
        "Def": "Potensi terjadinya kehilangan nyawa atau kerusakan aset akibat bencana. Dinilai berdasarkan bahaya, kerentanan, dan kapasitas (UNDRR, 2017)."
    }
}

CLEAN_VARS_LIST = list(VAR_MAPPING.values())

INDIKATOR_NEGATIF = [
    "Persentase Pengeluaran per Kapita Sebulan Makanan",
    "Harga Komoditas Beras", 
    "Harga Komoditas Jagung",
    "Prevalensi Balita Wasting",
    "Prevalensi Balita Underweight",
    "Kepadatan Penduduk",
    "Indeks Risiko Bencana"
]

DIMENSI_DICT = {
    "Indikator Umum": ["Indeks Ketahanan Pangan (IKP)"],
    "Ketersediaan (Availability)": ["Produksi Padi", "Produksi Jagung"],
    "Aksesibilitas (Accessibility)": ["Pendapatan per Kapita", "Persentase Pengeluaran per Kapita Sebulan Makanan", "Realisasi Penerima Bansos Pangan", "Harga Komoditas Beras", "Harga Komoditas Jagung"],
    "Pemanfaatan (Utilization)": ["Akses Air Minum Layak", "Akses Sanitasi Layak", "Prevalensi Balita Wasting", "Prevalensi Balita Underweight"],
    "Stabilitas (Stability)": ["Kepadatan Penduduk", "Indeks Risiko Bencana"]
}

# -----------------------------------------------------------------------------
# 3. FUNGSI UTAMA (NORMALISASI & ANALISIS)
# -----------------------------------------------------------------------------
def normalize_name(name):
    """Normalisasi Nama Provinsi agar Excel match dengan GeoJSON"""
    if not isinstance(name, str): return str(name)
    name = name.upper().strip()
    
    corrections = {
        # --- PERBAIKAN NAMA PROVINSI (SESUAI REQUEST) ---
        "DI. ACEH": "ACEH", 
        "NANGGROE ACEH DARUSSALAM": "ACEH",
        
        "DI YOGYAKARTA": "DI YOGYAKARTA", "DIY": "DI YOGYAKARTA", "DAERAH ISTIMEWA YOGYAKARTA": "DI YOGYAKARTA",
        "DKI JAKARTA": "DKI JAKARTA", "JAKARTA": "DKI JAKARTA", "JAKARTA RAYA": "DKI JAKARTA",
        
        # --- UPDATE FIX: Menambahkan variasi tanpa titik ---
        "BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG", 
        "KEP. BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG",
        "KEP BANGKA BELITUNG": "KEPULAUAN BANGKA BELITUNG", # Tambahan
        
        "KEPULAUAN RIAU": "KEPULAUAN RIAU", 
        "KEP. RIAU": "KEPULAUAN RIAU",
        "KEP RIAU": "KEPULAUAN RIAU", # Tambahan
        
        "NUSATENGGARA BARAT": "NUSA TENGGARA BARAT", 
        "NUSA TENGGARA BARAT": "NUSA TENGGARA BARAT", "NTB": "NUSA TENGGARA BARAT",
        
        "NUSATENGGARA TIMUR": "NUSA TENGGARA TIMUR", 
        "NUSA TENGGARA TIMUR": "NUSA TENGGARA TIMUR", "NTT": "NUSA TENGGARA TIMUR",
        
        "PAPUA BARAT DAYA": "PAPUA BARAT DAYA", "PAPUA SELATAN": "PAPUA SELATAN",
        "PAPUA TENGAH": "PAPUA TENGAH", "PAPUA PEGUNUNGAN": "PAPUA PEGUNUNGAN"
    }
    return corrections.get(name, name)

def generate_emoji_analysis(df_input):
    """Fungsi Analisis Z-Score Simbolik"""
    df = df_input.copy()
    var_cols = [c for c in CLEAN_VARS_LIST if c in df.columns]
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[var_cols])
    df_scaled = pd.DataFrame(scaled_data, columns=var_cols)
    df_scaled['Cluster'] = df['Cluster'].values
    df_scaled['Provinsi'] = df['Provinsi'].values
    
    def get_emoji(val, mean_val=0, is_negative=False):
        upper, lower = mean_val + 0.3, mean_val - 0.3
        if is_negative: return "❌" if val >= upper else ("✅" if val <= lower else "⚠️")
        else: return "✅" if val >= upper else ("❌" if val <= lower else "⚠️")

    def get_dimension_emoji(row, vars_list):
        valid_vars = [v for v in vars_list if v in row.index]
        if not valid_vars: return "-"
        emojis = row[valid_vars].tolist()
        if emojis.count("✅") > emojis.count("❌"): return "✅"
        elif emojis.count("❌") > emojis.count("✅"): return "❌"
        else: return "⚠️"

    # Analisis Klaster Normal & Noise
    results = []
    # 1. Klaster Normal (Rata-rata)
    df_normal = df_scaled[df_scaled['Cluster'] != -1]
    if not df_normal.empty:
        grouped = df_normal.groupby('Cluster')[var_cols].mean()
        for cluster_id, row in grouped.iterrows():
            res = {"Klaster": f"Klaster {cluster_id}", "Tipe": "Kelompok"}
            temp_emojis = {col: get_emoji(row[col], 0, col in INDIKATOR_NEGATIF) for col in var_cols}
            for dim, vars_ in DIMENSI_DICT.items(): res[dim] = get_dimension_emoji(pd.Series(temp_emojis), vars_)
            res['Anggota'] = ", ".join(sorted(df_normal[df_normal['Cluster'] == cluster_id]['Provinsi'].tolist()))
            results.append(res)
    
    # 2. Noise (Individu)
    df_noise = df_scaled[df_scaled['Cluster'] == -1]
    if not df_noise.empty:
        for _, row in df_noise.iterrows():
            res = {"Klaster": "Noise (Outlier)", "Tipe": f"Provinsi: {row['Provinsi']}"}
            temp_emojis = {col: get_emoji(row[col], 0, col in INDIKATOR_NEGATIF) for col in var_cols}
            for dim, vars_ in DIMENSI_DICT.items(): res[dim] = get_dimension_emoji(pd.Series(temp_emojis), vars_)
            res['Anggota'] = row['Provinsi']
            results.append(res)
            
    return pd.DataFrame(results)

# -----------------------------------------------------------------------------
# 4. LOAD DATASETS
# -----------------------------------------------------------------------------
@st.cache_data
def load_dataset():
    filename = "Hasil_Clustering_Final.xlsx"
    df = None
    if os.path.exists(filename):
        try:
            df = pd.read_excel(filename)
            if "X1" in df.columns: df = df.rename(columns=VAR_MAPPING)
        except: pass
    
    if df is None: # Dummy
        provs = ["ACEH","SUMATERA UTARA","DKI JAKARTA","JAWA BARAT","JAWA TIMUR","BALI","NUSA TENGGARA TIMUR","PAPUA"]
        np.random.seed(42)
        data = {"Provinsi": provs, "Cluster": np.random.choice([0,1,-1], len(provs))}
        for c in VAR_MAPPING.values(): data[c] = np.random.uniform(10,100, len(provs))
        df = pd.DataFrame(data)

    if 'Cluster' in df.columns and 'Cluster_Label' not in df.columns:
        df['Cluster_Label'] = df['Cluster'].apply(lambda x: f"Klaster {x}" if x != -1 else "Noise (Outlier)")
    
    df['Provinsi_Key'] = df['Provinsi'].apply(normalize_name)
    return df

@st.cache_data
def load_geojson():
    url = "https://raw.githubusercontent.com/ans-4175/peta-indonesia-geojson/master/indonesia-prov.geojson"
    try:
        gdf = gpd.read_file(url)
        
        # --- AUTO-DETECT KOLOM PROVINSI ---
        # Cari kolom yang mungkin berisi nama provinsi
        possible_cols = ['propinsi', 'PROVINSI', 'NAME_1', 'province', 'name', 'NAME']
        target_col = None
        for col in possible_cols:
            if col in gdf.columns:
                target_col = col
                break
        
        if target_col:
            gdf = gdf.rename(columns={target_col: 'Provinsi'})
        else:
            # Fallback jika tidak ketemu, pakai kolom pertama yang tipe string
            obj_cols = gdf.select_dtypes(include=['object']).columns
            if len(obj_cols) > 0:
                gdf = gdf.rename(columns={obj_cols[0]: 'Provinsi'})
        
        # Normalisasi Key
        if 'Provinsi' in gdf.columns:
            gdf['Provinsi_Key'] = gdf['Provinsi'].apply(normalize_name)
            return gdf
        else:
            return None
            
    except Exception as e: 
        print(f"Error GeoJSON: {e}")
        return None

df = load_dataset()
gdf = load_geojson()
available_features = [c for c in CLEAN_VARS_LIST if c in df.columns]

# Merge Data
if gdf is not None and df is not None:
    gdf_final = gdf.merge(df, on="Provinsi_Key", how="left")
    gdf_final['Cluster_Label'] = gdf_final['Cluster_Label'].fillna("Tidak Ada Data")
    
    # Smart Display Name (Prioritas nama dari Excel kalau ada)
    if 'Provinsi_y' in gdf_final.columns:
        gdf_final['Provinsi_Show'] = gdf_final['Provinsi_y'].fillna(gdf_final['Provinsi_x'])
    else:
        gdf_final['Provinsi_Show'] = gdf_final['Provinsi']
else:
    gdf_final = None

# -----------------------------------------------------------------------------
# 5. APLIKASI UTAMA
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2913/2913520.png", width=100)
    st.title("GeoAI Pangan")
    st.markdown("**Monitoring Ketahanan Pangan**\n*Metode: t-SNE & DBSCAN*")
    tile_provider = st.selectbox("Ganti Background Peta:", ["CartoDB positron", "CartoDB dark_matter", "OpenStreetMap"], index=0)
    
    if gdf_final is not None:
        match_c = gdf_final[gdf_final['Cluster_Label'] != "Tidak Ada Data"].shape[0]
        if match_c < 10: st.warning(f"⚠️ Data Match Rendah: {match_c} Provinsi")
            
    st.divider()
    menu = st.radio("Navigasi:", ["🏠 Dashboard Utama", "📊 Analisis Karakteristik", "📚 Metadata & Definisi", "ℹ️ Tentang Metode"])
    st.divider()
    st.caption("© 2025 Amelia Kurnia Fitri")

# HALAMAN 1: DASHBOARD
if menu == "🏠 Dashboard Utama":
    st.title("Peta Klaster Ketahanan Pangan")
    
    c1, c2 = st.columns([1, 2])
    with c1: map_mode = st.radio("Mode Tampilan:", ["🗺️ Hasil Klaster", "📈 Sebaran Variabel"], horizontal=True)
    with c2: var_select = st.selectbox("Pilih Indikator:", available_features) if map_mode == "📈 Sebaran Variabel" else None

    m = folium.Map(location=[-2.5, 118.0], zoom_start=5, tiles=tile_provider)
    
    if gdf_final is not None:
        if map_mode == "🗺️ Hasil Klaster":
            # --- PERBAIKAN WARNA KLASTER & LEGEND ---
            # 1. Warna baru yang lebih cerah dan beda dari abu-abu
            colors = {
                'Klaster 0': '#575fcf', # Biru Tua
                'Klaster 1': '#3498db', # Biru
                'Klaster 2': '#2ecc71', # Hijau
                'Klaster 3': '#f1c40f', # Kuning
                'Klaster 4': '#9b59b6', # Ungu (Baru)
                'Klaster 5': '#e67e22', # Oranye (Baru)
                'Klaster 6': '#1abc9c', # Tosca (Baru)
                'Noise (Outlier)': '#7f8c8d', # TETAP ABU-ABU (sesuai request)
                'Tidak Ada Data': '#ffffff'   # PUTIH (biar beda jauh sama Outlier)
            }
            
            # Render Peta
            folium.GeoJson(
                gdf_final,
                style_function=lambda x: {
                    'fillColor': colors.get(x['properties'].get('Cluster_Label'), 'grey'), 
                    'color': 'black', 
                    'weight': 1, 
                    'fillOpacity': 0.7
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['Provinsi_Show', 'Cluster_Label', VAR_MAPPING["X1"]], 
                    aliases=['Prov:', 'Status:', 'IKP:']
                ),
                popup=folium.GeoJsonPopup(fields=['Provinsi_Show', 'Cluster_Label'])
            ).add_to(m)
            
            # --- GENERATE KONTEN LEGEND DENGAN ANGGOTA ---
            legend_items = []
            
            # Urutkan label agar rapi (Klaster 0, 1, ... lalu Noise)
            sorted_keys = sorted([k for k in colors.keys() if "Klaster" in k]) + ['Noise (Outlier)']
            
            for label in sorted_keys:
                color = colors[label]
                # Cari anggota provinsi berdasarkan label di Dataframe df
                if label == 'Noise (Outlier)':
                    members = df[df['Cluster_Label'] == label]['Provinsi'].tolist()
                else:
                    members = df[df['Cluster_Label'] == label]['Provinsi'].tolist()
                
                # Format text anggota (dipotong jika terlalu panjang untuk visual)
                members_str = ", ".join(members) if members else "-"
                
                item_html = f"""
                <div style="margin-bottom: 8px; border-bottom: 1px solid #ddd; padding-bottom: 5px;">
                    <div style="display: flex; align-items: center; font-weight: bold; font-size: 13px; color: #000000;">
                        <i style="background:{color}; width:12px; height:12px; display:inline-block; margin-right:8px; border:1px solid #333; flex-shrink: 0;"></i>
                        {label} <span style="font-weight:normal; font-size:10px; margin-left:5px; color: #333;">({len(members)} prov)</span>
                    </div>
                    <div style="font-size: 10px; color: #000000; margin-left: 20px; line-height: 1.2; margin-top: 2px;">
                        {members_str}
                    </div>
                </div>
                """
                legend_items.append(item_html)

            # --- PERBAIKAN TAMPILAN LEGEND (SCROLLABLE) ---
            legend_html = f"""
            <div style="
                position: fixed; 
                bottom: 30px; left: 30px; 
                z-index: 9999; 
                width: 250px;
                max-height: 350px;
                overflow-y: auto;
                background: rgba(255, 255, 255, 0.95); /* Background Putih Solid */
                padding: 15px; 
                border: 1px solid #ccc; 
                border-radius: 8px;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                font-family: sans-serif;
                color: #000000;"> <h5 style="margin-top:0; margin-bottom:10px; border-bottom:2px solid #333; padding-bottom:5px; color: #000000;">
                    🗺️ Legenda & Anggota
                </h5>
                {''.join(legend_items)}
                <div style="font-size:9px; color: #333; margin-top:5px;">
                    <i>*Scroll untuk melihat daftar lengkap</i>
                </div>
            </div>
            """
            
            m.get_root().html.add_child(folium.Element(legend_html))
        else:
            folium.Choropleth(geo_data=gdf_final, data=gdf_final, columns=["Provinsi_Key", var_select], key_on="feature.properties.Provinsi_Key", fill_color="YlOrRd", legend_name=var_select).add_to(m)
            folium.GeoJson(gdf_final, style_function=lambda x: {'fillColor': '#00000000', 'color': 'black', 'weight': 1}, tooltip=folium.GeoJsonTooltip(fields=['Provinsi_Show', var_select])).add_to(m)

    st_data = st_folium(m, width="100%", height=500, returned_objects=["last_object_clicked"])

    if st_data and st_data['last_object_clicked']:
        try:
            props = st_data['last_object_clicked']['properties']
            pkey = props.get('Provinsi_Key')
            if pkey:
                pdata = df[df['Provinsi_Key'] == pkey].iloc[0]
                st.divider()
                st.subheader(f"📍 Detail: {pdata['Provinsi']}")
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Status", pdata['Cluster_Label'])
                k2.metric("IKP (X1)", f"{pdata.get(VAR_MAPPING['X1'], 0):.2f}")
                k3.metric("Prod. Padi", f"{pdata.get(VAR_MAPPING['X2'], 0):,.0f}")
                k4.metric("Pendapatan", f"{pdata.get(VAR_MAPPING['X4'], 0):,.0f}")
        except: pass

    st.divider()
    st.subheader("📋 Data Lengkap")
    c_filter = st.selectbox("Filter Klaster:", ["Semua"] + sorted(df['Cluster_Label'].unique()))
    df_show = df[df['Cluster_Label'] == c_filter] if c_filter != "Semua" else df
    
    col_cfg = {"Provinsi": st.column_config.TextColumn("Provinsi", pinned=True)}
    for col in available_features: col_cfg[col] = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(df_show, column_config=col_cfg, use_container_width=True, hide_index=True)

# -----------------------------------------------------------------------------
# HALAMAN 2: ANALISIS KARAKTERISTIK
# -----------------------------------------------------------------------------
elif menu == "📊 Analisis Karakteristik":
    st.title("Analisis Karakteristik Klaster")
    
    tab1, tab2, tab3 = st.tabs(["📈 Distribusi", "📊 Profil Rata-rata", "📝 Interpretasi Simbolik"])
    
    with tab1:
        st.info("Visualisasi sebaran data.")
        var_analisis = st.selectbox("Pilih Variabel:", available_features)
        fig = px.box(
            df, x="Cluster_Label", y=var_analisis, color="Cluster_Label", 
            points="all", hover_data=["Provinsi"], title=f"Distribusi {var_analisis}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.info("Membandingkan rata-rata Klaster Utama vs Noise.")
        df_main = df[df['Cluster'] != -1]
        df_noise = df[df['Cluster'] == -1]
        
        dim_select = st.selectbox("Pilih Dimensi:", list(DIMENSI_DICT.keys()))
        vars_in_dim = [v for v in DIMENSI_DICT[dim_select] if v in df.columns]
        
        outlier_options = df_noise['Provinsi'].unique().tolist()
        selected_outliers = st.multiselect("Pilih Outlier untuk dibandingkan:", outlier_options)
        
        if vars_in_dim:
            avg_df = df_main.groupby("Cluster_Label")[vars_in_dim].mean().reset_index()
            if selected_outliers:
                noise_data = df_noise[df_noise['Provinsi'].isin(selected_outliers)][['Provinsi'] + vars_in_dim]
                noise_data = noise_data.rename(columns={'Provinsi': 'Cluster_Label'})
                final_plot_df = pd.concat([avg_df, noise_data], ignore_index=True)
            else:
                final_plot_df = avg_df

            final_melt = final_plot_df.melt(id_vars="Cluster_Label", var_name="Indikator", value_name="Nilai")
            fig2 = px.bar(
                final_melt, x="Indikator", y="Nilai", color="Cluster_Label", 
                barmode="group", title=f"Profil {dim_select}"
            )
            st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.subheader("Interpretasi Kualitatif (Z-Score)")
        df_emoji = generate_emoji_analysis(df)
        st.dataframe(df_emoji, use_container_width=True, hide_index=True)

# HALAMAN 3: METADATA
elif menu == "📚 Metadata & Definisi":
    st.title("Kamus Data & Definisi Variabel")
    st.markdown("Definisi operasional variabel merujuk pada **Bab 3 Metodologi Penelitian**.")
    
    for dim, vars_ in DIMENSI_DICT.items():
        with st.expander(f"📂 {dim}", expanded=True):
            for v in vars_:
                code = [k for k, val in VAR_MAPPING.items() if val == v][0]
                info = VAR_METADATA.get(code, {})
                st.markdown(f"**{code} - {v}**")
                c1, c2 = st.columns([1, 4])
                with c1:
                    st.info(f"**Satuan:**\n{info.get('Unit')}")
                with c2:
                    st.write(f"**Definisi:** {info.get('Def')}")
                st.divider()

# HALAMAN 4: TENTANG
elif menu == "ℹ️ Tentang Metode":
    st.title("Tentang Metode GeoAI")
        
    st.markdown("""
    ### 1. Geospatial Artificial Intelligence (GeoAI)
    GeoAI adalah pendekatan multidisiplin yang menggabungkan metode dari geografi (khususnya Sistem Informasi Geografis/SIG) dengan Kecerdasan Buatan (AI), terutama *Machine Learning*. 
    
    Dalam penelitian ini, GeoAI digunakan untuk memetakan wilayah ketahanan pangan di Indonesia dengan cara yang lebih adaptif terhadap pola data yang kompleks, melampaui metode klasifikasi statistik konvensional.
    
    ### 2. Tahapan Analisis (Metode Hybrid)
    Penelitian ini menggunakan kombinasi dua algoritma *Unsupervised Learning*:
    
    #### **A. t-SNE (t-Distributed Stochastic Neighbor Embedding)**
    t-SNE digunakan sebagai langkah awal untuk **Reduksi Dimensi**.
    * **Masalah:** Data ketahanan pangan memiliki 14 dimensi (variabel) yang sulit dikelompokkan secara langsung karena fenomena *curse of dimensionality*.
    * **Solusi:** t-SNE memproyeksikan data 14 dimensi tersebut ke dalam ruang 2 dimensi (sumbu X dan Y baru).
    * **Kelebihan:** Sangat unggul dalam mempertahankan struktur lokal, artinya provinsi yang memiliki kemiripan karakteristik akan diletakkan sangat berdekatan dalam peta visualisasi.
    
    #### **B. DBSCAN (Density-Based Spatial Clustering)**
    DBSCAN digunakan untuk melakukan **Clustering** pada hasil reduksi t-SNE.
    * **Konsep:** Mengelompokkan data berdasarkan kepadatan titik. Titik-titik yang berkumpul padat dianggap satu klaster.
    * **Keunggulan vs K-Means:** 1. Tidak perlu menentukan jumlah klaster (K) secara manual.
        2. Bentuk klaster fleksibel (tidak harus bulat).
        3. **Deteksi Noise:** Provinsi yang karakteristiknya sangat unik (berbeda jauh dari provinsi lain) tidak akan dipaksa masuk klaster, melainkan dilabeli sebagai **Noise/Outlier (-1)**. Ini sangat penting untuk mendeteksi wilayah ekstrem (sangat rawan atau sangat tahan).
    
    ### 3. Implementasi Sistem
    Dashboard ini dibangun menggunakan **Python** dengan library:
    * **Streamlit:** Framework antarmuka web interaktif.
    * **Folium:** Visualisasi peta geospasial interaktif.
    * **Scikit-Learn:** Implementasi algoritma Machine Learning (StandardScaler, t-SNE, DBSCAN).
    """)
    st.success("© 2025 Amelia Kurnia Fitri - Proyek Akhir Statistika Bisnis ITS")
