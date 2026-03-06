import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# =========================
# FUNGSI KATEGORISASI
# =========================
def kategori_nilai(nilai):
    if nilai >= 90:
        return "Tinggi"
    elif nilai >= 80:
        return "Sedang"
    else:
        return "Rendah"


def kategori_kehadiran(persen):
    if persen >= 95:
        return "Sangat Baik"
    elif persen >= 90:
        return "Baik"
    else:
        return "Cukup"


def kategori_ekskul(jumlah):
    if jumlah >= 3:
        return "Tinggi"
    elif jumlah >= 1:
        return "Sedang"
    else:
        return "Rendah"


def label_prestasi(row):
    nilai_akhir = (row["nilai_smt1"] + row["nilai_smt2"]) / 2

    if nilai_akhir >= 90:
        return "Tinggi"
    elif nilai_akhir >= 80:
        return "Sedang"
    else:
        return "Rendah"


# =========================
# PREPROCESSING
# =========================
def preprocessing(df):
    df = df.copy()

    df["nilai_smt1"] = pd.to_numeric(df["nilai_smt1"], errors="coerce")
    df["nilai_smt2"] = pd.to_numeric(df["nilai_smt2"], errors="coerce")
    df["kehadiran"] = pd.to_numeric(df["kehadiran"], errors="coerce")
    df["jumlah_ekskul"] = pd.to_numeric(df["jumlah_ekskul"], errors="coerce")

    df["kat_smt1"] = df["nilai_smt1"].apply(kategori_nilai)
    df["kat_smt2"] = df["nilai_smt2"].apply(kategori_nilai)
    df["kat_kehadiran"] = df["kehadiran"].apply(kategori_kehadiran)
    df["kat_ekskul"] = df["jumlah_ekskul"].apply(kategori_ekskul)

    df["label_prestasi"] = df.apply(label_prestasi, axis=1)

    return df


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Analisis Prestasi Akademik", layout="wide")

st.title("📊 Aplikasi Analisis Prestasi Akademik Siswa")
st.caption("Penerapan Algoritma Naïve Bayes")

menu = st.sidebar.selectbox(
    "Menu",
    ["Home","Upload Data", "Proses & Analisis", "Hasil Prediksi"]
)

# =========================
# SESSION STATE
# =========================
for key in ["data", "model", "encoder", "X_test", "y_test", "y_pred", "df_prep"]:
    if key not in st.session_state:
        st.session_state[key] = None

# =========================
# MENU HOME
# =========================
if menu == "Home":  

    st.subheader("👋 Selamat Datang")
    st.markdown("""
    Aplikasi ini dikembangkan untuk mendukung proses analisis dan klasifikasi 
    prestasi akademik siswa menggunakan algoritma **Naïve Bayes**.

    Sistem ini membantu mengidentifikasi kategori prestasi siswa berdasarkan:
    - Nilai Semester 1
    - Nilai Semester 2
    - Tingkat Kehadiran
    - Jumlah Kegiatan Ekstrakurikuler

    Hasil analisis digunakan sebagai dasar pendukung dalam 
    peningkatan kualitas prestasi akademik siswa secara objektif dan terukur.
    """)

    st.markdown("### 🧠 Metodologi yang Digunakan")
    st.markdown("""
    Proses analisis dilakukan melalui tahapan data mining sebagai berikut:

    1. **Data Selection** – Pemilihan data siswa yang relevan  
    2. **Preprocessing** – Pembersihan dan kategorisasi data  
    3. **Transformation** – Encoding data kategorikal  
    4. **Modeling** – Penerapan algoritma Categorical Naïve Bayes  
    5. **Evaluation** – Pengujian model menggunakan akurasi dan confusion matrix  

    """)

    st.markdown("### 📌 Navigasi Aplikasi")
    st.markdown("""
    Gunakan menu di sidebar untuk mengakses fitur berikut:

    - 📂 **Upload Data**  
      Mengunggah dataset siswa dalam format Excel (.xlsx)

    - ⚙️ **Proses & Analisis**  
      Menjalankan proses training dan evaluasi model

    - 📈 **Hasil Prediksi**  
      Menampilkan hasil klasifikasi prestasi siswa berdasarkan model
    """)

    st.info("Silakan pilih menu di sidebar untuk memulai proses analisis.")

# =========================
# MENU 1 - UPLOAD DATA
# =========================
elif menu == "Upload Data":
    st.header("📂 Data Selection")

    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        required_cols = [
            "no", "nama", "kelas",
            "nilai_smt1", "nilai_smt2",
            "kehadiran", "jumlah_ekskul"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("Format tidak sesuai.")
            st.stop()

        st.session_state.data = df
        st.success(f"Data berhasil diupload. Total data: {len(df)} siswa")
        st.dataframe(df.head())


# =========================
# MENU 2 - PROSES & ANALISIS
# =========================
elif menu == "Proses & Analisis":

    if st.session_state.data is None:
        st.warning("Upload data terlebih dahulu.")
        st.stop()
    else : 
        st.subheader("📋 Data yang Akan Diproses")
        st.dataframe(st.session_state.data, use_container_width=True)


    st.header("⚙️ Tahapan Proses Data Mining")

    

    if st.button("🔍 Jalankan Proses Lengkap"):

        # 1️⃣ PREPROCESSING
        st.subheader("1️⃣ Preprocessing")
        df_prep = preprocessing(st.session_state.data)
        st.session_state.df_prep = df_prep
        st.dataframe(df_prep.head())

        # 2️⃣ TRANSFORMATION
        st.subheader("2️⃣ Transformation (Encoding)")
        fitur = ["kat_smt1", "kat_smt2", "kat_kehadiran", "kat_ekskul"]
        target = "label_prestasi"

        X = df_prep[fitur]
        y = df_prep[target]

        encoder = OrdinalEncoder()
        X_encoded = encoder.fit_transform(X)

        st.write("Data setelah encoding:")
        st.dataframe(pd.DataFrame(X_encoded, columns=fitur).head())

        # 3️⃣ SPLIT DATA
        st.subheader("3️⃣ Split Data (70% Training, 30% Testing)")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42
        )

        st.write(f"Jumlah Training: {len(X_train)}")
        st.write(f"Jumlah Testing: {len(X_test)}")

        # 4️⃣ TRAINING MODEL
        st.subheader("4️⃣ Training Model Naïve Bayes")
        model = CategoricalNB()
        model.fit(X_train, y_train)

        st.success("Model Naïve Bayes berhasil dilatih menggunakan data training.")
        st.write("Proses training dilakukan menggunakan algoritma Categorical Naïve Bayes dengan pembagian data 70% training dan 30% testing.")

        # 5️⃣ PREDIKSI
        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.encoder = encoder
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

        # 6️⃣ EVALUASI
        st.subheader("5️⃣ Evaluasi Model")

        akurasi = accuracy_score(y_test, y_pred)
        st.write(f"🎯 Akurasi: **{akurasi:.2%}**")

        cm = confusion_matrix(y_test, y_pred, labels=["Tinggi", "Sedang", "Rendah"])
        cm_df = pd.DataFrame(
            cm,
            index=["Actual Tinggi", "Actual Sedang", "Actual Rendah"],
            columns=["Pred Tinggi", "Pred Sedang", "Pred Rendah"]
        )
        st.write("Confusion Matrix:")
        st.dataframe(cm_df)

        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)


# =========================
# MENU 3 - HASIL PREDIKSI
# =========================
elif menu == "Hasil Prediksi":

    if st.session_state.model is None:
        st.warning("Silakan jalankan proses analisis terlebih dahulu.")
        st.stop()

    st.header("📈 Hasil Prediksi Seluruh Data")

    df_prep = st.session_state.df_prep
    fitur = ["kat_smt1", "kat_smt2", "kat_kehadiran", "kat_ekskul"]

    X_all = st.session_state.encoder.transform(df_prep[fitur])
    prediksi = st.session_state.model.predict(X_all)

    df_prep["hasil_prediksi_model"] = prediksi

    st.dataframe(
        df_prep[["nama", "kelas", "label_prestasi", "hasil_prediksi_model"]]
    )


    # =========================
    # EXPORT KE EXCEL
    # =========================
    hasil_export = df_prep[["nama", "kelas", "label_prestasi", "hasil_prediksi_model"]]

    # Simpan ke Excel dalam memory
    from io import BytesIO

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        hasil_export.to_excel(writer, index=False, sheet_name='Hasil_Prediksi')

    output.seek(0)

    st.download_button(
        label="📥 Download Hasil Prediksi (Excel)",
        data=output,
        file_name="hasil_prediksi_naive_bayes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

