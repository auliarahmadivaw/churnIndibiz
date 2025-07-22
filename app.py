import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt # Import matplotlib
import seaborn as sns # Import seaborn

# =======================
# 2. Load Model & Data
# =======================
st.set_page_config(page_title="Dashboard Churn IndiBiz", layout="wide")
# Load the saved model and preprocessor
try:
    model = joblib.load('xgb_churn_model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
except FileNotFoundError:
    st.error("Model atau preprocessor tidak ditemukan. Pastikan file 'xgb_churn_model.pkl' dan 'preprocessor.pkl' ada di direktori yang sama.")
    st.stop() # Stop the app if files are not found

# Let's define the features that the preprocessor expects:
features_for_preprocessing = ['STO', 'PAKET_DIGI', 'Lama_Berlangganan_Bulan']

# Define the categorical and numerical features used in the preprocessor
categorical_features = ['STO', 'PAKET_DIGI']
numerical_features = ['Lama_Berlangganan_Bulan']

# Get the unique values for categorical features (contoh)
sto_options = ['PBB', 'ARK', 'PKR', 'DUM', 'PPN', 'RGT', 'UBT', 'TBH', 'AMK', 'DRI', 'BKR', 'RBI', 'TAK', 'BGU', 'SLJ', 'BKN', 'BAG', 'SAK', 'PMB', 'BAS', 'MIS', 'PWG', 'SOK', 'KLE', 'SEA', 'SGP']
paket_digi_options = ['1P HSI', '2P', 'sooltanNet F 2P', '3P', '2P INET + VOICE', '1P VOICE', 'sooltanNet E 2P', 'sooltanNet C 3P', '1P INET', '1P']

# Function to assign churn risk category
def assign_churn_risk(prob):
    low_risk_threshold = 0.6
    medium_risk_threshold = 0.85
    if prob < low_risk_threshold:
        return 'Rendah'
    elif prob < medium_risk_threshold:
        return 'Sedang'
    else:
        return 'Tinggi'

# -------------------------------
# Tambahkan Fungsi Rekomendasi
# -------------------------------
def generate_recommendations(df):
    recs = []

    # 1. Intervensi Proaktif pada Paket atau Produk Digital
    if 'PAKET_DIGI' in df.columns and 'Churn_Risk_Category' in df.columns:
        paket_counts = df[df['Churn_Risk_Category'] == 'Tinggi']['PAKET_DIGI'].value_counts()
        if not paket_counts.empty:
            top_paket = paket_counts.idxmax()
            recs.append(
                f"Intervensi proaktif pada pelanggan paket seperti **{top_paket}**, karena menunjukkan konsentrasi risiko churn tinggi."
            )

    # 2. Perbaikan Layanan di STO Tertentu
    if 'STO' in df.columns and 'Churn_Risk_Category' in df.columns:
        sto_counts = df[df['Churn_Risk_Category'] == 'Tinggi']['STO'].value_counts()
        if not sto_counts.empty:
            top_sto = sto_counts.head(2).index.tolist()
            recs.append(
                f"Lakukan audit teknis dan peningkatan kualitas layanan di STO seperti **{', '.join(top_sto)}** yang mencatat churn tertinggi."
            )

    # 3. Edukasi & Engagement di 6–12 Bulan Pertama
    if 'Lama_Berlangganan_Bulan' in df.columns and 'Churn_Risk_Category' in df.columns:
        df_churn = df[df['Churn_Risk_Category'] == 'Tinggi']
        if not df_churn.empty:
            bins = [0, 6, 12, float('inf')]
            labels = ['0–6 bulan', '7–12 bulan', '>12 bulan']
            df_churn['tenure_group'] = pd.cut(df_churn['Lama_Berlangganan_Bulan'], bins=bins, labels=labels, right=True)
            group_counts = df_churn['tenure_group'].value_counts().sort_index()

        # Ambil grup dengan churn terbanyak
        if not group_counts.empty:
            top_group = group_counts.idxmax()
            if top_group == '0–6 bulan':
                recs.append("Tingkatkan onboarding dan komunikasi intensif pada 6 bulan pertama, karena di fase ini banyak pelanggan berisiko churn.")
            elif top_group == '7–12 bulan':
                recs.append("Perkuat engagement dan manfaat tambahan pada bulan ke-7 hingga ke-12, karena fase ini mencatat churn tertinggi.")
            else:
                recs.append("Evaluasi loyalitas pelanggan jangka panjang (>12 bulan) karena mereka masih menunjukkan risiko churn tinggi.")


    # 4. Segmentasi Penawaran Berdasarkan Ekosistem
    if 'EKOSISTEM' in df.columns and 'Churn_Risk_Category' in df.columns:
        ekosistem_risk = df[df['Churn_Risk_Category'] == 'Tinggi']['EKOSISTEM'].value_counts()
        if not ekosistem_risk.empty:
            top_eko = ekosistem_risk.head(2).index.tolist()
            recs.append(
                f"Buat program loyalitas khusus atau referral untuk ekosistem seperti **{', '.join(top_eko)}** yang memiliki tingkat churn tinggi."
            )

    # 5. Evaluasi Berdasarkan Jenis Produk
    if 'L_PRODUK' in df.columns and 'Churn_Risk_Category' in df.columns:
        produk_risk = df[df['Churn_Risk_Category'] == 'Tinggi']['L_PRODUK'].value_counts()
        if not produk_risk.empty:
            top_produk = produk_risk.idxmax()
            recs.append(
                f"Tinjau ulang benefit dan struktur produk **{top_produk}**, karena mendominasi kategori pelanggan dengan risiko churn tinggi."
            )

    return recs


# -------------------------------
# Inisialisasi Session State
# -------------------------------
if 'data_to_predict' not in st.session_state:
    st.session_state.data_to_predict = None

# =======================
# 3. Streamlit App Layout
# =======================

st.title("📊 Dashboard Prediksi Churn Pelanggan IndiBiz")

# Sidebar Navigation
st.sidebar.header("Navigasi 🧭")
page = st.sidebar.radio("Pilih Halaman:", ["Informasi Dashboard", "Unggah Data File", "Input Data Manual", "Analisis Pelanggan Churn"])

# =======================
# Main Content Area based on Sidebar Selection
# =======================

if page == "Informasi Dashboard":
    with st.container():
        st.header("📝 Informasi Dashboard")
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("coba2.jpg", width=500)
        st.markdown("---")
        st.markdown("""
        Selamat datang di Dashboard Prediksi Churn Pelanggan IndiBiz.
        Dashboard ini membantu Anda memahami faktor-faktor yang mempengaruhi churn dan memprediksi pelanggan yang berisiko tinggi menggunakan model machine learning.
        """)

elif page == "Unggah Data File":
    with st.container():
        st.header("📤 Unggah Data Pelanggan")
        st.info("Unggah file Excel (.xlsx) yang berisi data pelanggan untuk prediksi.")
        uploaded_file = st.file_uploader("Upload file Excel data pelanggan (.xlsx) 📂", type=["xlsx"])
        if uploaded_file is not None:
            try:
                data_to_predict_uploaded = pd.read_excel(uploaded_file)
                st.session_state.data_to_predict = data_to_predict_uploaded
                st.success("✅ File berhasil diunggah.")
                st.dataframe(st.session_state.data_to_predict.head())
                required_columns = features_for_preprocessing + ['Customer_Name']
                required_columns.extend(['Order Id', 'SPEEDY', 'L_EKOSISTEM', 'L_PRODUK'])
                required_columns = list(dict.fromkeys(required_columns))
                if not all(col in st.session_state.data_to_predict.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in st.session_state.data_to_predict.columns]
                    st.error(f"❌ File yang diunggah harus memiliki kolom berikut: {', '.join(missing_cols)}")
                    st.session_state.data_to_predict = None
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                st.session_state.data_to_predict = None

        if st.button("🚀 Prediksi Churn"):
            if st.session_state.data_to_predict is not None:
                try:
                    columns_for_prediction = features_for_preprocessing
                    X_predict = st.session_state.data_to_predict[columns_for_prediction]
                    X_predict_processed = preprocessor.transform(X_predict)
                    cat_encoder_full = preprocessor.named_transformers_['cat']
                    all_feature_names_full = list(cat_encoder_full.get_feature_names_out(categorical_features)) + numerical_features
                    X_predict_df = pd.DataFrame(X_predict_processed.toarray() if hasattr(X_predict_processed, 'toarray') else X_predict_processed,
                                                columns=all_feature_names_full)
                    predictions = model.predict(X_predict_df)
                    probabilities = model.predict_proba(X_predict_df)[:, 1]
                    data_with_predictions = st.session_state.data_to_predict.copy()
                    data_with_predictions['Predicted_Status_Churn'] = predictions
                    data_with_predictions['Churn_Probability'] = probabilities
                    data_with_predictions['Churn_Risk_Category'] = data_with_predictions['Churn_Probability'].apply(assign_churn_risk)
                    st.session_state.data_to_predict = data_with_predictions
                    st.subheader("📈 Hasil Prediksi Churn")
                    display_columns = ['Customer_Name', 'STO', 'PAKET_DIGI', 'Lama_Berlangganan_Bulan',
                   'L_PRODUK', 'L_EKOSISTEM', 'Predicted_Status_Churn', 'Churn_Probability', 'Churn_Risk_Category']
                    st.dataframe(st.session_state.data_to_predict[display_columns])
                    st.subheader("📊 Ringkasan Prediksi Churn")
                    with st.container(border=True):
                        st.write("Distribusi Kategori Resiko Churn ⚠️")
                        churn_risk_counts = st.session_state.data_to_predict['Churn_Risk_Category'].value_counts().reindex(['Rendah', 'Sedang', 'Tinggi'])
                        fig2, ax2 = plt.subplots(figsize=(6, 4))
                        sns.barplot(x=churn_risk_counts.index, y=churn_risk_counts.values, ax=ax2, palette='viridis')
                        ax2.set_xlabel('Kategori Resiko Churn')
                        ax2.set_ylabel('Jumlah')
                        st.pyplot(fig2)
                         # Tambahkan Kesimpulan Ringkas setelah grafik distribusi churn
                        with st.container():
                            st.markdown("### 🧾 Kesimpulan Ringkas")
                            total_pelanggan = len(st.session_state.data_to_predict)
                            tinggi = churn_risk_counts.get('Tinggi', 0)
                            sedang = churn_risk_counts.get('Sedang', 0)
                            rendah = churn_risk_counts.get('Rendah', 0)
                    
                            st.markdown(f"""
                            Dari total **{total_pelanggan} pelanggan**, distribusi risiko churn adalah sebagai berikut:
                            - 🔴 **{tinggi} pelanggan** tergolong **resiko Tinggi**
                            - 🟡 **{sedang} pelanggan** tergolong **resiko Sedang**
                            - 🟢 **{rendah} pelanggan** tergolong **resiko Rendah**
                            """)
                    
                            try:
                                top_paket = st.session_state.data_to_predict[st.session_state.data_to_predict['Churn_Risk_Category'] == 'Tinggi']['PAKET_DIGI'].value_counts().idxmax()
                                top_sto = st.session_state.data_to_predict[st.session_state.data_to_predict['Churn_Risk_Category'] == 'Tinggi']['STO'].value_counts().idxmax()
                                top_produk = st.session_state.data_to_predict[st.session_state.data_to_predict['Churn_Risk_Category'] == 'Tinggi']['L_PRODUK'].value_counts().idxmax()
                                top_ekosistem = st.session_state.data_to_predict[st.session_state.data_to_predict['Churn_Risk_Category'] == 'Tinggi']['L_EKOSISTEM'].value_counts().idxmax()
                    
                                st.markdown(f"""
                                📌 Sebagian besar pelanggan dengan risiko churn tinggi berasal dari:
                                - Paket: **{top_paket}**
                                - STO: **{top_sto}**
                                - Produk: **{top_produk}**
                                - Ekosistem: **{top_ekosistem}**
                    
                                Temuan ini menjadi dasar dalam menyusun rekomendasi pada halaman **Analisis Churn** berikutnya.
                                """)
                            except Exception as e:
                                st.warning("Tidak dapat menghitung analisis berdasarkan variabel: beberapa kolom mungkin kosong.")
                        plt.close(fig2)
                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    import traceback
                    st.text(traceback.format_exc())
            else:
                st.warning("⚠️ Silakan unggah file data pelanggan terlebih dahulu.")
                
    
elif page == "Input Data Manual":
    with st.container():
        st.header("⌨️ Input Data Manual")
        st.info("Masukkan detail pelanggan secara manual di bawah ini.")
        manual_sto = st.selectbox("STO", sto_options)
        manual_paket_digi = st.selectbox("PAKET_DIGI", paket_digi_options)
        manual_lama_berlangganan = st.slider("Lama Berlangganan (Bulan)", min_value=0, max_value=100, value=12)
        if st.button("🚀 Prediksi Churn"):
            if manual_sto and manual_paket_digi is not None and manual_lama_berlangganan is not None:
                 try:
                    manual_data = pd.DataFrame({
                        'STO': [manual_sto],
                        'PAKET_DIGI': [manual_paket_digi],
                        'Lama_Berlangganan_Bulan': [manual_lama_berlangganan]
                    })
                    columns_for_prediction = features_for_preprocessing
                    X_predict = manual_data[columns_for_prediction]
                    manual_data_processed = preprocessor.transform(X_predict)
                    cat_encoder_full = preprocessor.named_transformers_['cat']
                    all_feature_names_full = list(cat_encoder_full.get_feature_names_out(categorical_features)) + numerical_features
                    manual_data_processed_df = pd.DataFrame(manual_data_processed.toarray() if hasattr(manual_data_processed, 'toarray') else manual_data_processed,
                                                            columns=all_feature_names_full)
                    manual_prediction = model.predict(manual_data_processed_df)[0]
                    manual_probability = model.predict_proba(manual_data_processed_df)[:, 1][0]
                    manual_churn_risk = assign_churn_risk(manual_probability)
                    st.subheader("📈 Hasil Prediksi Churn (Input Manual)")
                    st.write(f"**Prediksi Status Churn:** {'Churn' if manual_prediction == 1 else 'Tidak Churn'}")
                    st.write(f"**Probabilitas Churn:** {manual_probability:.4f}")
                    st.write(f"**Kategori Resiko Churn:** {manual_churn_risk}")
                 except Exception as e:
                    st.error(f"❌ Error during manual prediction: {e}")
                    import traceback
                    st.text(traceback.format_exc())
            else:
                 st.warning("⚠️ Harap lengkapi semua bidang input manual.")

elif page == "Analisis Pelanggan Churn":
    with st.container():
        st.header("🔍 Analisis Pelanggan Churn")
        st.markdown("Visualisasi persebaran risiko churn berdasarkan faktor layanan pelanggan.")
        if st.session_state.data_to_predict is not None and 'Churn_Risk_Category' in st.session_state.data_to_predict.columns:
            df_mer = st.session_state.data_to_predict.copy()

            # Visualisasi Risiko Churn berdasarkan PAKET_DIGI
            st.subheader("📦 Risiko Churn berdasarkan Jenis Paket")
            paket_risk = df_mer.groupby(['PAKET_DIGI', 'Churn_Risk_Category']).size().unstack().fillna(0)
            fig, ax = plt.subplots(figsize=(10,6))
            paket_risk.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_title('Distribusi Risiko Churn berdasarkan Jenis Paket')
            ax.set_xlabel('Paket Digi')
            ax.set_ylabel('Jumlah Pelanggan')
            ax.legend(title='Kategori Risiko')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Visualisasi Risiko Churn berdasarkan STO
            st.subheader("🧾 Risiko Churn berdasarkan STO")
            fig, ax = plt.subplots(figsize=(10,6))
            pd.crosstab(df_mer['STO'], df_mer['Churn_Risk_Category']).plot(kind='barh', stacked=True, ax=ax)
            ax.set_title("Risiko Churn Berdasarkan STO (Lokasi)")
            ax.set_xlabel("Jumlah Pelanggan")
            ax.set_ylabel("STO")
            st.pyplot(fig)
            plt.close(fig)

            # Visualisasi Risiko Churn berdasarkan Lama Berlangganan
            st.subheader("📊 Risiko Churn berdasarkan Lama Berlangganan")
            df_filtered_lama = df_mer[df_mer['Lama_Berlangganan_Bulan'] > 0].copy()
            bins = [0, 6, 12, 24, 36, 48, df_filtered_lama['Lama_Berlangganan_Bulan'].max()]
            labels = ['0-6 bulan', '7-12 bulan', '13-24 bulan', '25-36 bulan', '37-48 bulan', '>48 bulan']
            if df_filtered_lama['Lama_Berlangganan_Bulan'].max() <= 48:
                for i, bin_val in enumerate(bins):
                    if df_filtered_lama['Lama_Berlangganan_Bulan'].max() <= bin_val:
                        bins = bins[:i+1]
                        labels = labels[:i]
                        break
                if bins[-1] == df_filtered_lama['Lama_Berlangganan_Bulan'].max() and len(labels) == len(bins) - 1:
                    labels[-1] = f'>{bins[-2]} bulan'
            df_filtered_lama['Lama_Berlangganan_Bulan_Binned'] = pd.cut(df_filtered_lama['Lama_Berlangganan_Bulan'], bins=bins, labels=labels, right=True, include_lowest=True, ordered=False)
            df_filtered_lama.dropna(subset=['Lama_Berlangganan_Bulan_Binned'], inplace=True)
            lama_berlangganan_risk = df_filtered_lama.groupby(['Lama_Berlangganan_Bulan_Binned', 'Churn_Risk_Category']).size().unstack().fillna(0)

            fig, ax = plt.subplots(figsize=(10,6))
            lama_berlangganan_risk.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
            ax.set_title('Distribusi Risiko Churn berdasarkan Lama Berlangganan (> 0 bulan)')
            ax.set_xlabel('Lama Berlangganan (bulan)')
            ax.set_ylabel('Jumlah Pelanggan')
            ax.legend(title='Kategori Risiko')
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            # Visualisasi Risiko Churn berdasarkan Produk
            st.subheader("🧾 Risiko Churn berdasarkan Produk")
            fig, ax = plt.subplots(figsize=(10,6))
            pd.crosstab(df_mer['L_PRODUK'], df_mer['Churn_Risk_Category']).plot(kind='barh', stacked=True, ax=ax)
            ax.set_title("Risiko Churn Berdasarkan Produk")
            ax.set_xlabel("Jumlah Pelanggan")
            ax.set_ylabel("Produk")
            st.pyplot(fig)
            plt.close(fig)

            # Visualisasi Risiko Churn berdasarkan Ekosistem
            st.subheader("🌐 Risiko Churn berdasarkan Ekosistem")
            fig, ax = plt.subplots(figsize=(10,8))
            pd.crosstab(df_mer['L_EKOSISTEM'], df_mer['Churn_Risk_Category']).plot(kind='barh', stacked=True, ax=ax)
            ax.set_title("Risiko Churn Berdasarkan Ekosistem")
            ax.set_xlabel("Jumlah Pelanggan")
            ax.set_ylabel("Ekosistem")
            st.pyplot(fig)
            plt.close(fig)

            # -------------------------------
            # Tampilkan Rekomendasi Dinamis
            # -------------------------------
            st.subheader("📌 Rekomendasi Strategis")
            recommendations = generate_recommendations(df_mer)
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")
            else:
                st.info("Tidak ada rekomendasi tambahan berdasarkan data saat ini.")

        else:
            st.warning("Silakan unggah data prediksi churn yang mencakup kolom 'Churn_Risk_Category' terlebih dahulu.")
