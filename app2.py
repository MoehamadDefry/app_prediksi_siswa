import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="ASPRI : Aplikasi Analisis Prestasi Siswa",
    layout="wide"
)

# =========================
# FUNGSI LOGIC (DARI app.py)
# =========================

def rekomendasi_siswa(row):
    rekomendasi = []

    # =========================
    # KEHADIRAN
    # =========================
    if row["kat_kehadiran"] == "Cukup":
        rekomendasi.append("Tingkatkan kehadiran di kelas")

    # =========================
    # NILAI AKADEMIK
    # =========================
    if row["kat_smt1"] == "Rendah" or row["kat_smt2"] == "Rendah":
        rekomendasi.append("Perlu remedial dan latihan tambahan")

    # =========================
    # EKSTRAKURIKULER
    # =========================
    if row["kat_ekskul"] == "Rendah":
        rekomendasi.append("Disarankan mengikuti kegiatan ekstrakurikuler")

    # =========================
    # PERILAKU (GLOBAL)
    # =========================
    if row["kat_perilaku"] == "Kurang":
        rekomendasi.append("Perlu pembinaan perilaku belajar secara umum")

    # =========================
    # PERILAKU DETAIL (SPESIFIK 🔥)
    # =========================
    if row["memperhatikan"] == 1:
        rekomendasi.append("Kurang memperhatikan saat guru menjelaskan")

    elif row["memperhatikan"] == 2:
        rekomendasi.append("Perlu meningkatkan fokus saat pembelajaran")

    if row["mengerjakan_tugas"] == 1:
        rekomendasi.append("Kurang mengerjakan tugas dengan baik")

    if row["tepat_waktu"] == 1:
        rekomendasi.append("Sering terlambat mengumpulkan tugas")

    if row["aktif_kelas"] == 1:
        rekomendasi.append("Kurang aktif dalam kegiatan kelas")

    elif row["aktif_kelas"] == 2:
        rekomendasi.append("Perlu lebih aktif dalam bertanya atau menjawab")

    if row["tertib"] == 1:
        rekomendasi.append("Perlu meningkatkan kedisiplinan di kelas")

    # =========================
    # DEFAULT
    # =========================
    if len(rekomendasi) == 0:
        return "Pertahankan performa dan konsistensi belajar"

    return "; ".join(rekomendasi)

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
    if nilai_akhir >= 90:
        return "Tinggi"
    elif nilai_akhir >= 80:
        return "Sedang"
    else:
        return "Rendah"

def hitung_perilaku(row):
    return (
        row["memperhatikan"] +
        row["mengerjakan_tugas"] +
        row["tepat_waktu"] +
        row["aktif_kelas"] +
        row["tertib"]
    )

def preprocessing(df):
    df = df.copy()

    df["nilai_smt1"] = pd.to_numeric(df["nilai_smt1"], errors="coerce")
    df["nilai_smt2"] = pd.to_numeric(df["nilai_smt2"], errors="coerce")
    df["kehadiran"] = pd.to_numeric(df["kehadiran"], errors="coerce")
    df["jumlah_ekskul"] = pd.to_numeric(df["jumlah_ekskul"], errors="coerce")

    df["memperhatikan"] = pd.to_numeric(df["memperhatikan"], errors="coerce")
    df["mengerjakan_tugas"] = pd.to_numeric(df["mengerjakan_tugas"], errors="coerce")
    df["tepat_waktu"] = pd.to_numeric(df["tepat_waktu"], errors="coerce")
    df["aktif_kelas"] = pd.to_numeric(df["aktif_kelas"], errors="coerce")
    df["tertib"] = pd.to_numeric(df["tertib"], errors="coerce")

    df["skor_perilaku"] = df.apply(hitung_perilaku, axis=1)

    df["kat_smt1"] = df["nilai_smt1"].apply(kategori_nilai)
    df["kat_smt2"] = df["nilai_smt2"].apply(kategori_nilai)
    df["kat_kehadiran"] = df["kehadiran"].apply(kategori_kehadiran)
    df["kat_ekskul"] = df["jumlah_ekskul"].apply(kategori_ekskul)
    df["kat_perilaku"] = df["skor_perilaku"].apply(kategori_perilaku)

    df["label_prestasi"] = df.apply(label_prestasi, axis=1)

    return df

# ======================
# CUSTOM CSS DASAR
# ======================
st.markdown("""
<style>
/* Hilangkan menu default */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Background */
.stApp {
    background-color: #F4F1DE;
}

/* Judul */
h1 {
    color: #3D405B;
    font-size: 42px;
}

/* Subjudul */
p {
    color: #555;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

.card h3 {
    margin-bottom: 10px;
    color: #3D405B;
}

.card p {
    font-size: 14px;
    color: #666;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* HERO TITLE */
.hero-title {
    font-size: 48px;
    font-weight: 700;
    color: #3D405B;
    line-height: 1.2;
    transition: 0.3s;
}
.hero-title:hover {
    transform: scale(1.02);
}

/* HIGHLIGHT TEXT */
.highlight {
    color: #E07A5F;
}

/* HERO SUBTEXT */
.hero-sub {
    font-size: 18px;
    color: #555;
    margin-top: 10px;
    margin-bottom: 20px;
}

/* BUTTON PREMIUM */
.btn-primary {
    background: linear-gradient(135deg, #E07A5F, #d6684d);
    color: white;
    padding: 12px 24px;
    border-radius: 12px;
    text-decoration: none;
    display: inline-block;
    font-weight: 500;
    transition: all 0.25s ease;
    box-shadow: 0 6px 15px rgba(224,122,95,0.3);
}

.btn-primary:hover {
    transform: translateY(-3px);
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* CARD PREMIUM */
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    transition: all 0.3s ease;
    height: 100%;
    border: 1px solid rgba(0,0,0,0.03);
    position: relative;
    overflow: hidden;
}
.card::after {
    content: "";
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(
        120deg,
        transparent,
        rgba(255,255,255,0.4),
        transparent
    );
    transition: 0.5s;
}

.card:hover {
    transform: translateY(-8px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.1);
}
.card:hover::after {
    left: 100%;
}

/* ICON */
.card-icon {
    font-size: 30px;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* SPACING */
.section {
    margin-top: 40px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
.navbar {
    background-color: #3D405B;
    padding: 10px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* TOP BAR */
.topbar {
    position: fixed;
    top: 60px;
    left: 0;
    right: 0;
    height: 70px;
    background: #3D405B;
    color: white;

    display: flex;
    align-items: center;
    justify-content: center; /* INI YANG BIKIN TENGAH */

    font-size: 40px; /* DIPERBESAR */
    font-weight: 700;
    letter-spacing: 0.5px;

    z-index: 999;
}

/* CONTENT SHIFT */
.block-container {
    padding-top: 90px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="topbar">
    ASPRI : Aplikasi Analisis Prestasi Siswa
</div>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* PRIMARY EXPLANATION */
.explain-box {
    background:#ffffff;
    padding:15px 20px;
    border-radius:12px;
    border-left:5px solid #E07A5F;
    margin-bottom:20px;
    font-size:15px;
    color:#444;
}

/* SECONDARY TEXT */
.subtext {
    font-size:14px;
    color:#666;
    margin-bottom:10px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR MENU
# =========================

with st.sidebar:
    selected = option_menu(
        menu_title="📊 ASPRI",
        options=[
            "Home",
            "Upload Data",
            "Proses & Analisis",
            "Hasil Prediksi"
        ],
        icons=[
            "house",
            "upload",
            "gear",
            "bar-chart"
        ]
    )


# =========================
# SESSION STATE
# =========================
for key in ["data", "model", "encoder", "df_prep"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================
# HOME (DESAIN BARU)
# =========================
if selected == "Home":

    col1, col2 = st.columns([2,1])

    with col1:
        st.markdown("""
        <div class="hero-title">
        Sistem Prediksi <br>
        <span class="highlight">Performa Siswa</span>
        </div>

        <div class="hero-sub">
        Menganalisis performa siswa berdasarkan nilai, kehadiran,
        dan aktivitas belajar menggunakan machine learning.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## 🚀 Fitur Utama")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card">
        <h3>⚙️ Preprocessing</h3>
        <p>Persiapan data sebelum analisis</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card">
        <h3>📊 Analisis</h3>
        <p>Eksplorasi dan visualisasi data</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card">
        <h3>🤖 Prediksi</h3>
        <p>Klasifikasi performa siswa</p>
        </div>
        """, unsafe_allow_html=True)


# =========================
# UPLOAD (BIKIN LEBIH RAPI)
# =========================
elif selected == "Upload Data":

    st.markdown("## 📂 Upload Dataset")

    # =========================
    # DOWNLOAD TEMPLATE (SIMPLE)
    # =========================
    template = pd.DataFrame(columns=[
        "no","nama","kelas",
        "nilai_smt1","nilai_smt2",
        "kehadiran","jumlah_ekskul",
        "memperhatikan","mengerjakan_tugas",
        "tepat_waktu","aktif_kelas","tertib"
    ])

    from io import BytesIO
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        template.to_excel(writer, index=False)

    st.download_button(
        "⬇️ Download Template",
        data=output,
        file_name="template.xlsx"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # =========================
    # UPLOAD
    # =========================
    file = st.file_uploader("Upload file Excel", type=["xlsx"])

    if file:
        df = pd.read_excel(file)
        st.session_state.data = df

        st.success("Data berhasil diupload")
        st.dataframe(df)

# =========================
# PROSES
# =========================
elif selected == "Proses & Analisis":

    if st.session_state.data is None:
        st.warning("Upload data dulu")
        st.stop()

    st.markdown("## ⚙️ Proses Analisis")

    st.subheader("📋 Data Awal")
    st.caption("Dataset sebelum dilakukan preprocessing")
    st.dataframe(st.session_state.data)

    if st.button("🚀 Jalankan Proses Lengkap"):

        # =========================
        # PREPROCESSING
        # =========================
        st.subheader("1️⃣ Preprocessing")
        st.markdown("""
            <div class="explain-box">
            Tahap ini mengubah data numerik menjadi kategorikal untuk kebutuhan klasifikasi.
            </div>
            """, unsafe_allow_html=True)
        # =========================
        # PROSES
        # =========================
        df_prep = preprocessing(st.session_state.data)
        st.session_state.df_prep = df_prep

        # =========================
        # INSIGHT UTAMA
        # =========================
        total_data = len(df_prep)
        jumlah_fitur = len(df_prep.columns)

        st.info(f"""
        Dataset berhasil diproses dengan total **{total_data} data** dan **{jumlah_fitur} fitur**.

        Pada tahap ini dilakukan:
        - Konversi data numerik menjadi kategori
        - Pembentukan fitur perilaku
        - Penentuan label prestasi siswa

        Transformasi ini bertujuan untuk mempermudah model dalam mengenali pola klasifikasi.
        """)

        # =========================
        # CARD RINGKAS (BIAR GA MONOTON)
        # =========================
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="card">
            <h3>📊 Total Data</h3>
            <p style="font-size:28px;font-weight:bold;">{total_data}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="card">
            <h3>🧩 Jumlah Fitur</h3>
            <p style="font-size:28px;font-weight:bold;">{jumlah_fitur}</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="card">
            <h3>🏷️ Label Output</h3>
            <p style="font-size:20px;">Tinggi / Sedang / Rendah</p>
            </div>
            """, unsafe_allow_html=True)

        # =========================
        # CEK DATA (BUKTI SKRIPSI)
        # =========================
        st.markdown("### 🔍 Hasil Preprocessing (Preview Data)")
        st.dataframe(df_prep.head())

        # =========================
        # DISTRIBUSI KATEGORI (BIAR ADA ANALISIS)
        # =========================
        st.markdown("## 📊 Distribusi Label Prestasi")

        # explain
        st.markdown("""
        <div class="explain-box">
        Visualisasi ini menunjukkan jumlah siswa pada setiap kategori prestasi.
        Distribusi ini penting untuk memahami keseimbangan data sebelum proses modeling.
        </div>
        """, unsafe_allow_html=True)

        # data
        label_counts = df_prep["label_prestasi"].value_counts()

        # chart
        st.bar_chart(label_counts)

        # insight otomatis
        mayoritas = label_counts.idxmax()
        minoritas = label_counts.idxmin()

        st.markdown(f"""
        <div class="subtext">
        Mayoritas siswa berada pada kategori <b>{mayoritas}</b>, 
        sedangkan kategori paling sedikit adalah <b>{minoritas}</b>.
        </div>
        """, unsafe_allow_html=True)

        ratio = label_counts.max() / label_counts.min()

        if ratio > 2:
            st.warning("Distribusi data tidak seimbang (imbalanced), hal ini dapat mempengaruhi performa model.")

        # =========================
        # ANALISIS FAKTOR
        # =========================
        st.subheader("🔎 Analisis Faktor")

        fitur = ["kat_smt1","kat_smt2","kat_kehadiran","kat_ekskul","kat_perilaku"]

        st.markdown("""
        <div class="explain-box">
        Analisis ini bertujuan untuk melihat hubungan antara setiap fitur dengan label prestasi siswa.
        Nilai yang lebih tinggi menunjukkan kecenderungan yang lebih kuat terhadap kategori tertentu.
        </div>
        """, unsafe_allow_html=True)

        tabs = st.tabs(fitur)

        for i, f in enumerate(fitur):
            with tabs[i]:

                st.markdown(f"### Fitur: {f}")

                cross = pd.crosstab(df_prep[f], df_prep["label_prestasi"], normalize='index')
                st.dataframe(cross)

                # insight otomatis
                dominan = cross.idxmax(axis=1)
                st.info(f"""
                Kecenderungan kategori: {dominan.to_dict()}
                """)

                # st.markdown(f"""
                # <div class="subtext">
                # Kecenderungan kategori:
                # {dominan.to_dict()}
                # </div>
                # """, unsafe_allow_html=True)

        interpretasi = ", ".join([f"{k} → {v}" for k,v in dominan.to_dict().items()])

        # st.markdown(f"""
        # <div class="subtext">
        # Secara umum, pola yang terlihat adalah: {interpretasi}.
        # </div>
        # """, unsafe_allow_html=True)
        st.info(f"""
                Secara umum, pola yang terlihat adalah: {interpretasi}.
                """)

        fig, ax = plt.subplots()
        sns.heatmap(cross, annot=True, cmap="Blues")
        st.pyplot(fig)
        # =========================
        # ENCODING
        # =========================
        st.subheader("2️⃣ Encoding")

        st.markdown("""
        <div class="explain-box">
        Tahap encoding mengubah data kategorikal menjadi bentuk numerik agar dapat diproses oleh model machine learning.
        Setiap kategori akan direpresentasikan dalam bentuk angka.
        </div>
        """, unsafe_allow_html=True)

        # =========================
        # PROSES
        # =========================
        X = df_prep[fitur]
        y = df_prep["label_prestasi"]

        encoder = OrdinalEncoder()
        X_enc = encoder.fit_transform(X)

        # =========================
        # MAPPING (INI YANG PENTING BANGET)
        # =========================
        st.markdown("### 🔄 Mapping Kategori ke Angka")

        for i, col in enumerate(fitur):
            kategori = encoder.categories_[i]
            
            mapping_text = ", ".join([f"{k} → {idx}" for idx, k in enumerate(kategori)])
            
            st.info(f"""
            **{col}**: {mapping_text}
                    """)
            # st.markdown(f"""
            # <div class="subtext">
            # <b>{col}</b>: {mapping_text}
            # </div>
            # """, unsafe_allow_html=True)

        # =========================
        # PREVIEW DATA
        # =========================
        st.markdown("### 🔍 Hasil Encoding (Preview Data)")

        df_enc = pd.DataFrame(X_enc, columns=fitur)
        st.dataframe(df_enc.head())

        st.markdown("""
            <div class="subtext">
            Encoding dilakukan menggunakan metode Ordinal Encoding, di mana setiap kategori diberi representasi angka berdasarkan urutan tertentu.
            </div>
            """, unsafe_allow_html=True)
        st.caption("Catatan: Representasi angka tidak selalu menunjukkan urutan tingkat kepentingan.")
        st.markdown("""
        <div class="card">
        <h3>🔢 Total Fitur Ter-encode</h3>
        <p style="font-size:28px;font-weight:bold;">{}</p>
        </div>
        """.format(len(fitur)), unsafe_allow_html=True)

        # =========================
        # SPLIT
        # =========================
        st.subheader("3️⃣ Split Data")

        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=0.3, random_state=42
        )

        # =========================
        # SPLIT DATA ASLI (UNTUK ANALISIS FREKUENSI)
        # =========================

        X_train_asli, X_test_asli, _, _ = train_test_split(
            X,
            y,
            test_size=0.3,
            random_state=42
        )

        # ======================================
        # DISTRIBUSI DATA TRAINING & TESTING
        # ======================================

        train_dist = (
            y_train.value_counts()
            .rename("Training")
        )

        test_dist = (
            y_test.value_counts()
            .rename("Testing")
        )

        dist_df = pd.concat(
            [train_dist, test_dist],
            axis=1
        ).fillna(0).astype(int)

        dist_df["Total"] = (
            dist_df["Training"]
            + dist_df["Testing"]
        )

        dist_df = dist_df.reindex(
            ["Tinggi", "Sedang", "Rendah"]
        )

        train_size = len(X_train)
        test_size = len(X_test)

        split_html = f"""
        <style>
        .split-wrapper {{
            display: flex;
            gap: 30px;
            justify-content: center;
        }}

        .split-card {{
            background: white;
            padding: 25px;
            border-radius: 16px;
            width: 300px;
            text-align: center;
            box-shadow: 0 6px 20px rgba(0,0,0,0.05);
            border-top: 5px solid #3D405B;
        }}

        .split-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
        }}

        .split-value {{
            font-size: 28px;
            font-weight: bold;
            color: #E07A5F;
        }}
        </style>

        <div class="split-wrapper">
            <div class="split-card">
                <div class="split-title">Data Training</div>
                <div class="split-value">{train_size}</div>
            </div>

            <div class="split-card">
                <div class="split-title">Data Testing</div>
                <div class="split-value">{test_size}</div>
            </div>
        </div>
        """

        components.html(split_html, height=200)

        st.markdown("### 📊 Distribusi Data Training dan Testing")

        st.markdown("""
        <div class="explain-box">
        Distribusi berikut menunjukkan jumlah data pada masing-masing kategori prestasi
        setelah proses pembagian dataset menjadi data training dan data testing.
        Data training digunakan untuk membangun model Categorical Naïve Bayes,
        sedangkan data testing digunakan untuk mengevaluasi performa model.
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            dist_df,
            use_container_width=True
        )

        st.bar_chart(
            dist_df[["Training", "Testing"]]
        )

        mayoritas_train = dist_df["Training"].idxmax()
        minoritas_train = dist_df["Training"].idxmin()

        st.info(f"""
        Data training didominasi oleh kategori **{mayoritas_train}**, sedangkan kategori dengan jumlah data paling sedikit adalah **{minoritas_train}**.

        Distribusi ini akan digunakan sebagai dasar dalam proses pelatihan model Categorical Naïve Bayes.
        """)

        # ======================================
        # DISTRIBUSI FREKUENSI ATRIBUT
        # ======================================

        st.subheader("📊 Distribusi Frekuensi Atribut")

        fitur_freq = [
            "kat_smt1",
            "kat_smt2",
            "kat_kehadiran",
            "kat_ekskul",
            "kat_perilaku"
        ]

        st.markdown("""
        <div class="explain-box">
        Distribusi frekuensi atribut menunjukkan jumlah kemunculan setiap kategori atribut
        pada masing-masing label prestasi menggunakan data training. Informasi ini menjadi
        dasar dalam perhitungan probabilitas <i>likelihood</i> pada algoritma Categorical
        Naïve Bayes.
        </div>
        """, unsafe_allow_html=True)

        tabs_freq = st.tabs(fitur_freq)

        for i, f in enumerate(fitur_freq):

            with tabs_freq[i]:

                st.markdown(f"### Fitur : **{f}**")

                freq_df = pd.crosstab(
                    X_train_asli[f],
                    y_train
                )

                freq_df = freq_df.reindex(
                    columns=["Tinggi", "Sedang", "Rendah"],
                    fill_value=0
                )

                freq_df["Total"] = freq_df.sum(axis=1)

                st.dataframe(
                    freq_df,
                    use_container_width=True
                )

        # =========================
        # TRAINING
        # =========================
        st.subheader("4️⃣ Training Model")

        model = CategoricalNB()
        model.fit(X_train, y_train)

        st.success("Model berhasil dilatih")

        y_pred = model.predict(X_test)

        # simpan ke session
        st.session_state.model = model
        st.session_state.encoder = encoder
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred

        # =========================
        # EVALUASI
        # =========================
        st.subheader("5️⃣ Evaluasi Model")

        acc = accuracy_score(y_test, y_pred)
        st.markdown("## 🎯 Performa Model")

        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <h3>📊 Akurasi Model</h3>
            <p style="font-size:40px;font-weight:bold;color:#E07A5F;">
                {acc:.2%}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # insight
        if acc >= 0.8:
            kualitas = "sangat baik"
        elif acc >= 0.6:
            kualitas = "cukup baik"
        else:
            kualitas = "perlu perbaikan"

        st.info(f"""
        Model memiliki performa yang tergolong **{kualitas}** dalam mengklasifikasikan prestasi siswa.
                """
                )
        
        cm = confusion_matrix(y_test, y_pred, labels=["Tinggi","Sedang","Rendah"])

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d')
        st.pyplot(fig)

        st.info(f"""
        Confusion matrix menunjukkan perbandingan antara hasil prediksi model dengan data aktual.
Semakin banyak nilai pada diagonal utama, semakin baik performa model.
        """)

        benar = np.trace(cm)
        total = np.sum(cm)
        st.info(f"""
        Model memprediksi dengan benar sebanyak **{benar}** dari **{total}** data uji.
        """)


        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        report_df = pd.DataFrame(report).transpose()

        avg_precision = report_df.loc["weighted avg", "precision"]
        avg_recall = report_df.loc["weighted avg", "recall"]

        st.info(f"""
        Rata-rata precision: **{avg_precision:.2f}**  
Rata-rata recall: **{avg_recall:.2f}**
        """)


        st.subheader("📊 Evaluasi per Kelas")

        report_df = pd.DataFrame(report).transpose()

        col1, col2, col3 = st.columns(3)

        labels = ["Tinggi", "Sedang", "Rendah"]
        cols = [col1, col2, col3]

        cards_html = ""

        for label in labels:
            if label in report_df.index:
                precision = report_df.loc[label, "precision"]
                recall = report_df.loc[label, "recall"]

                cards_html += f"""
                <div class="eval-card">
                    <div class="eval-title">Kelas {label}</div>

                    <div class="eval-metric">Precision</div>
                    <div class="eval-value">{precision:.2f}</div>

                    <div class="eval-metric">Recall</div>
                    <div class="eval-value">{recall:.2f}</div>
                </div>
                """
        html_full = f"""
        <style>
        body {{
            font-family: sans-serif;
        }}

        .wrapper {{
            display: flex;
            gap: 30px;
            justify-content: center;
            width: 100%;
            margin: auto;
        }}

        .eval-card {{
            background: white;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.05);
            text-align: center;
            border-top: 5px solid #E07A5F;
            width: 400px;
        }}

        .eval-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
        }}

        .eval-metric {{
            font-size: 14px;
            color: #666;
        }}

        .eval-value {{
            font-size: 20px;
            font-weight: bold;
            color: #3D405B;
        }}
        </style>

        <div class="wrapper">
        {cards_html}
        </div>
        """

        components.html(html_full, height=250)
        best_class = report_df["f1-score"].drop("accuracy").idxmax()

        st.info(f"""
           Model memiliki performa terbaik dalam memprediksi kelas **{best_class}**</b>.
Hal ini menunjukkan bahwa pola pada kelas tersebut lebih mudah dikenali oleh model dibandingkan kelas lainnya.         
                """)


# =========================
# HASIL
# =========================
elif selected == "Hasil Prediksi":

    st.markdown("## 📈 Hasil Prediksi")

    # =========================
    # VALIDASI
    # =========================
    if st.session_state.model is None:
        st.warning("Silakan lakukan proses analisis terlebih dahulu")
        st.stop()

    df = st.session_state.df_prep.copy()

    fitur = ["kat_smt1","kat_smt2","kat_kehadiran","kat_ekskul","kat_perilaku"]

    X_all = st.session_state.encoder.transform(df[fitur])
    pred = st.session_state.model.predict(X_all)

    df["hasil_prediksi"] = pred
    df["rekomendasi"] = df.apply(rekomendasi_siswa, axis=1)

    # =========================
    # EXPLAIN
    # =========================
    st.markdown("""
    <div class="explain-box">
    Sistem telah melakukan klasifikasi terhadap seluruh siswa berdasarkan data yang tersedia.
    Hasil ini dapat digunakan untuk mengidentifikasi siswa yang memerlukan perhatian khusus.
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # SUMMARY
    # =========================
    total = len(df)
    tinggi = len(df[df["hasil_prediksi"]=="Tinggi"])
    sedang = len(df[df["hasil_prediksi"]=="Sedang"])
    rendah = len(df[df["hasil_prediksi"]=="Rendah"])

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
        <h3>👥 Total Siswa</h3>
        <p style="font-size:30px;font-weight:bold;">{total}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
        <h3>📈 Prestasi Tinggi</h3>
        <p style="font-size:30px;font-weight:bold;color:#2A9D8F;">{tinggi}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
        <h3>🚨 Perlu Perhatian</h3>
        <p style="font-size:30px;font-weight:bold;color:#E76F51;">{rendah}</p>
        </div>
        """, unsafe_allow_html=True)

    # =========================
    # INSIGHT
    # =========================
    mayoritas = df["hasil_prediksi"].value_counts().idxmax()

    st.markdown(f"""
    <div class="subtext">
    Mayoritas siswa berada pada kategori <b>{mayoritas}</b>. 
    Sistem berhasil mengidentifikasi siswa dengan performa rendah yang memerlukan perhatian lebih.
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # SISWA BERISIKO (PRIORITAS)
    # =========================
    st.markdown("### 🚨 Siswa Berisiko")

    risiko = df[df["hasil_prediksi"]=="Rendah"]

    if len(risiko) > 0:
        st.dataframe(risiko[[
            "nama","kelas",
            "hasil_prediksi",
            "rekomendasi"
        ]])
    else:
        st.success("Tidak ada siswa dalam kategori risiko rendah 🎉")

    # =========================
    # DATA LENGKAP
    # =========================
    st.markdown("### 📋 Seluruh Hasil Prediksi")

    st.dataframe(df[[
        "nama","kelas",
        "label_prestasi",
        "hasil_prediksi",
        "rekomendasi"
    ]])

    # =========================
    # DOWNLOAD
    # =========================
    from io import BytesIO
    output = BytesIO()

    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)

    st.markdown("<br>", unsafe_allow_html=True)

    st.download_button(
        "⬇️ Download Hasil Prediksi",
        data=output,
        file_name="hasil_prediksi.xlsx",
        use_container_width=True
    )