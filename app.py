import streamlit as st
import pandas as pd
import joblib
import base64
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt # Import matplotlib
import seaborn as sns # Import seaborn

# =======================
# 1. Load Gambar dan Encode Base64 (Commented out as the file is not available)
# ===============================
# Fungsi untuk mengatur latar belakang
# def set_background(image_path):
#     try:
#         with open(image_path, "rb") as f:
#             data = f.read()
#         encoded = base64.b64encode(data).decode()
#         st.markdown(
#             f"""
#             <style>
#             .stApp {{
#                 background-image: url(data:image/png;base64,{encoded});
#                 background-size: cover;
#                 background-repeat: no-repeat;
#                 background-attachment: fixed; # Optional: Fix background during scroll
#             }}
#             </style>
#             """,
#             unsafe_allow_html=True
#         )
#     except FileNotFoundError:
#         st.warning(f"⚠️ File gambar latar belakang '{image_path}' tidak ditemukan.")

# # Set the background image
# set_background("BG WEB.png")

# # Add custom CSS for reduced opacity of main content area and centering image
# st.markdown(
#     """
#     <style>
#     /* Remove default header background */
#     .stApp > header {
#         background-color: transparent;
#     }

#     /* Style the main content area container with semi-transparent white background */
#     .main .block-container {
#         background-color: rgba(255, 255, 255, 0.9); /* Increased opacity to 90% */
#         padding: 20px; /* Add some padding */
#         border-radius: 10px; /* Rounded corners */
#         margin-top: 20px; /* Add some space from the top */
#         margin-bottom: 20px; /* Add some space at the bottom */
#         position: relative; /* Needed for z-index to work correctly */
#         z-index: 1; /* Ensure the main content is above the background */
#     }

#     /* Center images within the main content area */
#     .main .block-container img {
#         display: block; /* Make the image a block element */
#         margin-left: auto; /* Auto left margin */
#         margin-right: auto; /* Auto right margin */
#     }

#      /* Optional: Adjust opacity of sidebar if needed */
#     /*
#     .stSidebar > div:first-child {
#         background-color: rgba(255, 255, 255, 0.85); # White background with 85% opacity
#         padding: 20px;
#         border-radius: 10px;
#         position: relative;
#         z-index: 1;
#     }
#     */
#     </style>
#     """,
#     unsafe_allow_html=True
# )


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
# Based on cell NamMnwUDMZ9h, these were:
categorical_features = ['STO', 'PAKET_DIGI']
numerical_features = ['Lama_Berlangganan_Bulan']

# Get the unique values for categorical features from the training data (assuming they are representative)
# You would typically load these from a saved list or infer from the preprocessor if possible
# For this example, let's use some plausible options based on the EDA outputs
# Ensure these lists contain all possible values the model was trained on
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

# Initialize session state for data_to_predict if it doesn't exist
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
        st.header("📝 Informasi Dashboard") # Add header with emoji

        # Add space and then the image placeholder
        st.markdown("---") # Add a horizontal rule for separation

        # Use columns to center the image
        col1, col2, col3 = st.columns([1, 2, 1]) # Create three columns, middle one wider

        with col2: # Place the image in the middle column
             st.image("coba2.jpg",  width=500) # Placeholder for company image

        st.markdown("---") # Add another horizontal rule

        st.markdown("""
        Selamat datang di Dashboard Prediksi Churn Pelanggan IndiBiz.
        Dashboard ini membantu Anda memahami faktor-faktor yang mempengaruhi churn dan memprediksi pelanggan yang berisiko tinggi untuk churn menggunakan model machine learning.
        """)


elif page == "Unggah Data File":
    with st.container():
        st.header("📤 Unggah Data Pelanggan") # Add header with emoji
        st.info("Unggah file Excel (.xlsx) yang berisi data pelanggan untuk prediksi.") # Add info message
        # A more robust way is to allow uploading a CSV file
        uploaded_file = st.file_uploader("Upload file Excel data pelanggan (.xlsx) 📂", type=["xlsx"])

        if uploaded_file is not None:
            try:
                data_to_predict_uploaded = pd.read_excel(uploaded_file)
                st.session_state.data_to_predict = data_to_predict_uploaded # Store uploaded data in session state
                st.success("✅ File berhasil diunggah.")
                st.dataframe(st.session_state.data_to_predict.head())

                # Ensure the uploaded data has the necessary columns for preprocessing and display
                # Required columns are those needed by the preprocessor PLUS any columns needed for display (like Customer_Name)
                required_columns = features_for_preprocessing + ['Customer_Name'] # Need Customer_Name for display
                # You might also need 'Order Id', 'SPEEDY', 'L_EKOSISTEM', 'L_PRODUK' for display or other purposes later,
                # but they are not strictly required by the preprocessor itself based on the notebook code.
                # Let's add them to required_columns if they are expected to be in the input file for display.
                # Based on the final output dataframe in the notebook, these columns ('Order Id', 'SPEEDY', 'L_EKOSISTEM', 'L_PRODUK') are present.
                # So, let's include them in the required columns for the input file.
                required_columns.extend(['Order Id', 'SPEEDY', 'L_EKOSISTEM', 'L_PRODUK'])
                # Remove duplicates just in case
                required_columns = list(dict.fromkeys(required_columns))


                if not all(col in st.session_state.data_to_predict.columns for col in required_columns):
                    missing_cols = [col for col in required_columns if col not in st.session_state.data_to_predict.columns]
                    st.error(f"❌ File yang diunggah harus memiliki kolom berikut: {', '.join(missing_cols)}")
                    st.session_state.data_to_predict = None # Reset data if columns is missing


            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                st.session_state.data_to_predict = None

        # =======================
        # 5. Prediction (for File Upload)
        # =======================
        if st.button("🚀 Prediksi Churn"):
            if st.session_state.data_to_predict is not None:
                try:
                    # Columns for prediction must match the features used in the preprocessor
                    columns_for_prediction = features_for_preprocessing

                    # Select only the feature columns needed for prediction that the preprocessor is trained on
                    X_predict = st.session_state.data_to_predict[columns_for_prediction]

                    # Apply the same preprocessing as during training
                    # Use the fitted preprocessor
                    X_predict_processed = preprocessor.transform(X_predict)

                    # Convert processed data back to DataFrame with correct column names for SHAP (optional but good practice)
                    # Get feature names after preprocessing
                    cat_encoder_full = preprocessor.named_transformers_['cat']
                    # Use the original categorical feature names used in preprocessor training
                    all_feature_names_full = list(cat_encoder_full.get_feature_names_out(categorical_features)) + numerical_features

                    X_predict_df = pd.DataFrame(X_predict_processed.toarray() if hasattr(X_predict_processed, 'toarray') else X_predict_processed,
                                                columns=all_feature_names_full)


                    # Make predictions
                    predictions = model.predict(X_predict_df)
                    probabilities = model.predict_proba(X_predict_df)[:, 1] # Probability of churn (class 1)

                    # Add predictions and probabilities to the original dataframe (make a copy to avoid SettingWithCopyWarning)
                    data_with_predictions = st.session_state.data_to_predict.copy()
                    data_with_predictions['Predicted_Status_Churn'] = predictions
                    data_with_predictions['Churn_Probability'] = probabilities

                    # Add Churn Risk Category based on probability
                    data_with_predictions['Churn_Risk_Category'] = data_with_predictions['Churn_Probability'].apply(assign_churn_risk)

                    st.session_state.data_to_predict = data_with_predictions # Update session state with predictions


                    st.subheader("📈 Hasil Prediksi Churn")
                    # Display relevant columns, ensure 'Customer_Name' is included if it exists in the uploaded data
                    display_columns = ['Customer_Name', 'Predicted_Status_Churn', 'Churn_Probability', 'Churn_Risk_Category']
                    st.dataframe(st.session_state.data_to_predict[display_columns])

                    # Display summary of churn predictions using charts
                    st.subheader("📊 Ringkasan Prediksi Churn")

                    # Chart for Churn Risk Category Distribution (Bar Chart)
                    # Using a single container which might help with centering or at least a cleaner layout
                    with st.container(border=True):
                        st.write("Distribusi Kategori Resiko Churn ⚠️")
                        churn_risk_counts = st.session_state.data_to_predict['Churn_Risk_Category'].value_counts().reindex(['Rendah', 'Sedang', 'Tinggi'])
                        fig2, ax2 = plt.subplots(figsize=(6, 4)) # Further adjusted figure size
                        sns.barplot(x=churn_risk_counts.index, y=churn_risk_counts.values, ax=ax2, palette='viridis')
                        ax2.set_xlabel('Kategori Resiko Churn')
                        ax2.set_ylabel('Jumlah')
                        st.pyplot(fig2)
                        plt.close(fig2) # Close the figure to prevent it from being displayed twice


                except Exception as e:
                    st.error(f"❌ Error during prediction: {e}")
                    import traceback
                    st.text(traceback.format_exc())

            else:
                st.warning("⚠️ Silakan unggah file data pelanggan terlebih dahulu.")


elif page == "Input Data Manual":
    with st.container():
        st.header("⌨️ Input Data Manual") # Add header with emoji
        st.info("Masukkan detail pelanggan secara manual di bawah ini.") # Add info message

        # Add input fields for manual data here
        # Removed manual_customer_name input
        manual_sto = st.selectbox("STO", sto_options)
        manual_paket_digi = st.selectbox("PAKET_DIGI", paket_digi_options)
        manual_lama_berlangganan = st.slider("Lama Berlangganan (Bulan)", min_value=0, max_value=100, value=12) # Changed to slider

        # =======================
        # 5. Prediction (for Manual Input)
        # =======================
        if st.button("🚀 Prediksi Churn"):
             # Check if required manual inputs are provided
             if manual_sto and manual_paket_digi is not None and manual_lama_berlangganan is not None:
                 try:
                    # Create a DataFrame from manual input
                    manual_data = pd.DataFrame({
                        'STO': [manual_sto],
                        'PAKET_DIGI': [manual_paket_digi],
                        'Lama_Berlangganan_Bulan': [manual_lama_berlangganan]
                    })

                    # Columns for prediction must match the features used in the preprocessor
                    columns_for_prediction = features_for_preprocessing

                    # Select only the feature columns needed for prediction that the preprocessor is trained on
                    X_predict = manual_data[columns_for_prediction]


                    # Apply preprocessing
                    manual_data_processed = preprocessor.transform(X_predict)

                    # Convert processed data back to DataFrame with correct column names
                    cat_encoder_full = preprocessor.named_transformers_['cat']
                    all_feature_names_full = list(cat_encoder_full.get_feature_names_out(categorical_features)) + numerical_features
                    manual_data_processed_df = pd.DataFrame(manual_data_processed.toarray() if hasattr(manual_data_processed, 'toarray') else manual_data_processed,
                                                            columns=all_feature_names_full)

                    # Make prediction
                    manual_prediction = model.predict(manual_data_processed_df)[0]
                    manual_probability = model.predict_proba(manual_data_processed_df)[:, 1][0]

                    # Determine Churn Risk Category
                    manual_churn_risk = assign_churn_risk(manual_probability)

                    st.subheader("📈 Hasil Prediksi Churn (Input Manual)")
                    # Removed Customer_Name display
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
        st.markdown("Visualisasi faktor-faktor yang terkait dengan pelanggan yang churn.")

        # Use data from session state for visualization if available and has 'Status_Churn'
        if st.session_state.data_to_predict is not None and 'Status_Churn' in st.session_state.data_to_predict.columns:
            df_for_analysis = st.session_state.data_to_predict # Use data from session state
            df_churn = df_for_analysis[df_for_analysis['Status_Churn'] == 1].copy()


            if not df_churn.empty:
                st.subheader("Distribusi Faktor pada Pelanggan Churn")

                # Plot 1: PAKET_DIGI
                st.write("Paket Digi Pelanggan Churn:")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df_churn, y='PAKET_DIGI', ax=ax, order=df_churn['PAKET_DIGI'].value_counts().index)
                ax.set_title('Pelanggan Churn Berdasarkan Paket Digi')
                ax.set_xlabel('Jumlah')
                ax.set_ylabel('Paket')
                st.pyplot(fig)
                plt.close(fig)

                # Plot 2: L_EKOSISTEM
                st.write("Ekosistem Pelanggan Churn:")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(data=df_churn, y='L_EKOSISTEM', ax=ax, order=df_churn['L_EKOSISTEM'].value_counts().index)
                ax.set_title('Pelanggan Churn Berdasarkan Ekosistem')
                ax.set_xlabel('Jumlah')
                ax.set_ylabel('Ekosistem')
                st.pyplot(fig)
                plt.close(fig)

                # Plot 3: L_PRODUK
                st.write("Produk Pelanggan Churn:")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(data=df_churn, y='L_PRODUK', ax=ax, order=df_churn['L_PRODUK'].value_counts().index)
                ax.set_title('Pelanggan Churn Berdasarkan Produk')
                ax.set_xlabel('Jumlah')
                ax.set_ylabel('Produk')
                st.pyplot(fig)
                plt.close(fig)

                # Plot 4: Lama Berlangganan
                st.write("Lama Berlangganan Pelanggan Churn :") # Updated text
                # Filter data for Lama_Berlangganan_Bulan > 0
                df_churn_filtered_lama = df_churn[df_churn['Lama_Berlangganan_Bulan'] > 0].copy()

                if not df_churn_filtered_lama.empty:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.hist(df_churn_filtered_lama['Lama_Berlangganan_Bulan'], bins=10, color='orange', edgecolor='black')
                    ax.set_title('Distribusi Lama Berlangganan Pelanggan Churn') # Updated title
                    ax.set_xlabel('Lama Berlangganan (bulan)')
                    ax.set_ylabel('Jumlah')
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.info("Tidak ada pelanggan churn dengan lama berlangganan di atas 0 bulan dalam file yang diunggah.")


                # Plot 5: STO
                st.write("STO Pelanggan Churn:")
                fig, ax = plt.subplots(figsize=(8, 8)) # Adjust size for potentially many STOs
                sns.countplot(data=df_churn, y='STO', ax=ax, order=df_churn['STO'].value_counts().index)
                ax.set_title('STO Pelanggan yang Churn')
                ax.set_xlabel('Jumlah')
                ax.set_ylabel('STO')
                st.pyplot(fig)
                plt.close(fig)


            else:
                st.info("Tidak ada data pelanggan yang churn dalam file yang diunggah untuk ditampilkan analisisnya.")
        elif st.session_state.data_to_predict is not None and 'Status_Churn' not in st.session_state.data_to_predict.columns:
             st.warning("File yang diunggah tidak memiliki kolom 'Status_Churn' yang diperlukan untuk analisis pelanggan churn.")
        else:
             st.warning("Silakan unggah file data pelanggan terlebih dahulu di halaman 'Unggah Data File' untuk melihat analisis pelanggan churn.")
