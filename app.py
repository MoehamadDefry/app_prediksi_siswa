import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns


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


def kategori_perilaku(skor):
    if skor >= 13:
        return "Baik"
    elif skor >= 10:
        return "Cukup"
    else:
        return "Kurang"


def label_prestasi(row):
    nilai_akhir = (row["nilai_smt1"] + row["nilai_smt2"]) / 2

    if nilai_akhir >= 85:
        return "Tinggi"
    elif nilai_akhir >= 75:
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
    df["skor_perilaku"] = pd.to_numeric(df["skor_perilaku"], errors="coerce")

    df["kat_smt1"] = df["nilai_smt1"].apply(kategori_nilai)
    df["kat_smt2"] = df["nilai_smt2"].apply(kategori_nilai)
    df["kat_kehadiran"] = df["kehadiran"].apply(kategori_kehadiran)
    df["kat_ekskul"] = df["jumlah_ekskul"].apply(kategori_ekskul)
    df["kat_perilaku"] = df["skor_perilaku"].apply(kategori_perilaku)

    df["label_prestasi"] = df.apply(label_prestasi, axis=1)

    return df


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Analisis Prestasi Akademik", layout="wide")

st.title("📊 Aplikasi Analisis Prestasi Akademik Siswa")

menu = st.sidebar.selectbox(
    "Menu",
    ["Home", "Upload Data", "Proses & Analisis", "Hasil Prediksi"]
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
    - Perilaku Belajar Siswa

    Penambahan variabel perilaku belajar bertujuan untuk meningkatkan akurasi model
    serta memberikan pendekatan analisis yang lebih komprehensif terhadap faktor-faktor
    yang mempengaruhi prestasi akademik siswa.
    """)

    st.markdown("### 🧠 Metodologi yang Digunakan")
    st.markdown("""
    1. Data Selection  
    2. Preprocessing  
    3. Transformation  
    4. Modeling  
    5. Evaluation  
    """)

    st.info("Silakan pilih menu di sidebar untuk memulai.")


# =========================
# UPLOAD DATA
# =========================
elif menu == "Upload Data":

    st.header("📂 Data Selection")

    # =========================
    # DOWNLOAD TEMPLATE
    # =========================

    template_df = pd.DataFrame({
        "no": [],
        "nama": [],
        "kelas": [],
        "nilai_smt1": [],
        "nilai_smt2": [],
        "kehadiran": [],
        "jumlah_ekskul": [],
        "skor_perilaku": []
    })

    from io import BytesIO
    output_template = BytesIO()

    with pd.ExcelWriter(output_template, engine='xlsxwriter') as writer:
        template_df.to_excel(writer, index=False, sheet_name='Template')

    output_template.seek(0)

    st.download_button(
        label="📄 Download Template Excel",
        data=output_template,
        file_name="template_dataset_siswa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

    if uploaded_file:
        df = pd.read_excel(uploaded_file)

        required_cols = [
            "no", "nama", "kelas",
            "nilai_smt1", "nilai_smt2",
            "kehadiran", "jumlah_ekskul",
            "skor_perilaku"
        ]

        if not all(col in df.columns for col in required_cols):
            st.error("Format tidak sesuai.")
            st.stop()

        st.session_state.data = df
        st.success(f"Data berhasil diupload. Total data: {len(df)} siswa")
        st.dataframe(df.head())


# =========================
# PROSES & ANALISIS
# =========================
elif menu == "Proses & Analisis":

    if st.session_state.data is None:
        st.warning("Upload data terlebih dahulu.")
        st.stop()

    st.subheader("📋 Data yang Akan Diproses")
    st.dataframe(st.session_state.data)

    if st.button("🔍 Jalankan Proses Lengkap"):

        # PREPROCESSING
        st.subheader("1️⃣ Preprocessing")
        df_prep = preprocessing(st.session_state.data)
        st.session_state.df_prep = df_prep
        st.dataframe(df_prep.head())

        # VISUALISASI
        st.subheader("📊 Visualisasi Data")

        st.write("Distribusi Prestasi Siswa")
        st.bar_chart(df_prep["label_prestasi"].value_counts())

        st.write("Distribusi Perilaku Belajar")
        st.bar_chart(df_prep["kat_perilaku"].value_counts())

        st.write("Perilaku vs Prestasi")
        cross = pd.crosstab(df_prep["kat_perilaku"], df_prep["label_prestasi"])
        st.dataframe(cross)

        # TRANSFORMASI
        st.subheader("2️⃣ Transformation (Encoding)")
        fitur = ["kat_smt1", "kat_smt2", "kat_kehadiran", "kat_ekskul", "kat_perilaku"]
        target = "label_prestasi"

        X = df_prep[fitur]
        y = df_prep[target]

        encoder = OrdinalEncoder()
        X_encoded = encoder.fit_transform(X)

        st.write("Data setelah encoding:")
        st.dataframe(pd.DataFrame(X_encoded, columns=fitur).head())

        # SPLIT
        st.subheader("3️⃣ Split Data")
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=0.3, random_state=42
        )

        st.write(f"Jumlah Training: {len(X_train)}")
        st.write(f"Jumlah Testing: {len(X_test)}")

        # TRAINING
        st.subheader("4️⃣ Training Model")
        model = CategoricalNB()
        model.fit(X_train, y_train)

        st.success("Model berhasil dilatih.")

        y_pred = model.predict(X_test)

        st.session_state.model = model
        st.session_state.encoder = encoder
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

        # EVALUASI
        st.subheader("5️⃣ Evaluasi Model")

        akurasi = accuracy_score(y_test, y_pred)
        st.write(f"🎯 Akurasi: **{akurasi:.2%}**")

        cm = confusion_matrix(y_test, y_pred, labels=["Tinggi", "Sedang", "Rendah"])

        st.write("Confusion Matrix (Visual):")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Tinggi","Sedang","Rendah"],
                    yticklabels=["Tinggi","Sedang","Rendah"])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot(fig)

        st.write("Classification Report:")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())


# =========================
# HASIL PREDIKSI
# =========================
elif menu == "Hasil Prediksi":

    if st.session_state.model is None:
        st.warning("Silakan jalankan proses analisis terlebih dahulu.")
        st.stop()

    st.header("📈 Hasil Prediksi")

    df_prep = st.session_state.df_prep
    fitur = ["kat_smt1", "kat_smt2", "kat_kehadiran", "kat_ekskul", "kat_perilaku"]

    X_all = st.session_state.encoder.transform(df_prep[fitur])
    prediksi = st.session_state.model.predict(X_all)

    df_prep["hasil_prediksi_model"] = prediksi

    st.dataframe(df_prep[["nama", "kelas", "label_prestasi", "hasil_prediksi_model"]])

    from io import BytesIO
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_prep.to_excel(writer, index=False)

    st.download_button(
        label="📥 Download Hasil Prediksi",
        data=output,
        file_name="hasil_prediksi.xlsx"
    )