import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Dashboard TPT Jawa Timur",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================


@st.cache_data
def load_data():

    # baca csv dulu
    df = pd.read_csv("Master2020-2022_nc.csv")

    # hitung statistik
    mean_tpt = df["TPT"].mean()
    std_tpt = df["TPT"].std()

    def kategori_tpt(x):

        if x < (mean_tpt - std_tpt):
            return "Rendah"

        elif x > (mean_tpt + std_tpt):
            return "Tinggi"

        else:
            return "Sedang"

    df["Kategori_TPT"] = df["TPT"].apply(kategori_tpt)

    return df

df = load_data()

# =====================================================
# LOAD MODEL
# =====================================================

dt_model = joblib.load("models/decision_tree.pkl")
rf_model = joblib.load("models/random_forest.pkl")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📊 Dashboard TPT")

menu = st.sidebar.selectbox(
    "Pilih Menu",
    [
        "Dashboard",
        "Analisis Deskriptif",
        "Perbandingan Model",
        "Confusion Matrix",
        "Prediksi",
        "Daerah Prioritas",
        "Feature Importance"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.title(
        "Analisis dan Prediksi Tingkat Pengangguran Terbuka Jawa Timur"
    )
    
    mean_tpt = df["TPT"].mean()
    std_tpt = df["TPT"].std()
    
    st.info(f"""
    Kategori Tingkat Pengangguran Terbuka (TPT) dibentuk menggunakan metode
    Mean ± Standar Deviasi.

    • Rendah : TPT < {mean_tpt - std_tpt:.2f}
    • Sedang : {mean_tpt - std_tpt:.2f} ≤ TPT ≤ {mean_tpt + std_tpt:.2f}
    • Tinggi : TPT > {mean_tpt + std_tpt:.2f}
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Data",
        len(df)
    )

    col2.metric(
        "Kabupaten/Kota",
        df["Kabupaten"].nunique()
    )

    col3.metric(
        "Periode",
        f"{df['Tahun'].min()} - {df['Tahun'].max()}"
    )

    st.markdown("---")

    st.subheader("Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

# =====================================================
# ANALISIS DESKRIPTIF
# =====================================================

elif menu == "Analisis Deskriptif":

    st.title("📈 Analisis Deskriptif")
    
    st.info("""
    Menu ini menyajikan gambaran umum kondisi Tingkat Pengangguran Terbuka (TPT) di Jawa Timur berdasarkan data BPS tahun 2020–2022. Analisis dilakukan untuk melihat distribusi kategori pengangguran, perkembangan TPT dari tahun ke tahun, serta mengidentifikasi daerah dengan tingkat pengangguran tertinggi.
    """)

    col1, col2 = st.columns(2)

    with col1:

        kategori = (
            df["Kategori_TPT"]
            .value_counts()
            .reset_index()
        )

        kategori.columns = [
            "Kategori",
            "Jumlah"
        ]

        fig = px.bar(
            kategori,
            x="Kategori",
            y="Jumlah",
            color="Kategori",
            title="Distribusi Kategori TPT"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        fig2 = px.pie(
            df,
            names="Kategori_TPT",
            title="Proporsi Kategori TPT"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    rata_tpt = (
        df.groupby("Tahun")["TPT"]
        .mean()
        .reset_index()
    )

    fig3 = px.line(
        rata_tpt,
        x="Tahun",
        y="TPT",
        markers=True,
        title="Rata-rata TPT per Tahun"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    top10 = (
        df.sort_values(
            "TPT",
            ascending=False
        )
        .head(10)
    )

    fig4 = px.bar(
        top10,
        x="TPT",
        y="Kabupaten",
        orientation="h",
        title="Top 10 Kabupaten/Kota dengan TPT Tertinggi"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )
    
    kategori_terbanyak = df["Kategori_TPT"].value_counts().idxmax()

    st.markdown(f"""
    **Interpretasi:**

    Berdasarkan hasil pengelompokan menggunakan metode tertile, data TPT dibagi menjadi tiga kategori yaitu Rendah, Sedang, dan Tinggi. Kategori yang paling dominan pada dataset adalah **{kategori_terbanyak}**, yang menunjukkan bahwa sebagian besar kabupaten/kota berada pada kelompok tersebut.
    """)
    

# =====================================================
# PERBANDINGAN MODEL
# =====================================================

elif menu == "Perbandingan Model":

    st.title("Perbandingan Model")
    
    st.info("""
    Pada tahap ini dilakukan evaluasi terhadap dua algoritma machine learning yaitu Decision Tree dan Random Forest. Perbandingan dilakukan menggunakan metrik Accuracy, Precision, Recall, dan F1-Score untuk menentukan model terbaik dalam memprediksi kategori Tingkat Pengangguran Terbuka.
    """)

    X = df[
        ["IPM", "TPAK", "PDRB", "Tahun"]
    ]

    y = df["Kategori_TPT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)

    hasil = pd.DataFrame({

        "Model": [
            "Decision Tree",
            "Random Forest"
        ],

        "Accuracy": [
            accuracy_score(y_test, dt_pred),
            accuracy_score(y_test, rf_pred)
        ],

        "Precision": [
            precision_score(y_test, dt_pred, average="weighted"),
            precision_score(y_test, rf_pred, average="weighted")
        ],

        "Recall": [
            recall_score(y_test, dt_pred, average="weighted"),
            recall_score(y_test, rf_pred, average="weighted")
        ],

        "F1 Score": [
            f1_score(y_test, dt_pred, average="weighted"),
            f1_score(y_test, rf_pred, average="weighted")
        ]
    })

    st.dataframe(
        hasil,
        use_container_width=True
    )
    
    best_model = hasil.loc[
    hasil["Accuracy"].idxmax(),
    "Model"
    ]

    best_acc = hasil["Accuracy"].max()

    st.success(
        f"Model terbaik berdasarkan Accuracy adalah {best_model} dengan nilai Accuracy sebesar {best_acc:.2%}."
    )

    fig = px.bar(
        hasil,
        x="Model",
        y="Accuracy",
        color="Model"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================================
# CONFUSION MATRIX
# =====================================================

elif menu == "Confusion Matrix":

    st.title("📋 Confusion Matrix Random Forest")
    
    st.info("""
    Confusion Matrix digunakan untuk mengevaluasi performa model Random Forest dalam mengklasifikasikan kategori TPT. Matriks ini menunjukkan jumlah prediksi yang benar maupun salah pada masing-masing kategori.
    """)

    X = df[
        ["IPM", "TPAK", "PDRB", "Tahun"]
    ]

    y = df["Kategori_TPT"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pred = rf_model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        pred
    )

    fig, ax = plt.subplots(figsize=(6,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax
    )

    st.pyplot(fig)

    report = classification_report(
        y_test,
        pred,
        output_dict=True
    )
    
    st.markdown("""
    **Interpretasi:**

    Semakin besar nilai yang berada pada diagonal utama confusion matrix, maka semakin baik kemampuan model dalam melakukan klasifikasi. Nilai precision, recall, dan F1-score yang tinggi menunjukkan bahwa model memiliki kemampuan yang baik dalam membedakan kategori TPT Rendah, Sedang, dan Tinggi.
    """)

    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

# =====================================================
# PREDIKSI
# =====================================================

elif menu == "Prediksi":

    st.title("Prediksi Kategori TPT")
    
    st.info("""
    Menu ini digunakan untuk melakukan simulasi prediksi kategori Tingkat Pengangguran Terbuka berdasarkan nilai IPM, TPAK, PDRB, dan Tahun. Hasil prediksi diperoleh dari model Decision Tree dan Random Forest yang telah dilatih menggunakan data historis BPS.
    """)

    ipm = st.number_input("IPM", value=70.0)
    tpak = st.number_input("TPAK", value=70.0)
    pdrb = st.number_input("PDRB", value=30000.0)

    tahun = st.selectbox(
        "Tahun",
        sorted(df["Tahun"].unique())
    )

    if st.button("Prediksi"):

        data = pd.DataFrame({

            "IPM": [ipm],
            "TPAK": [tpak],
            "PDRB": [pdrb],
            "Tahun": [tahun]

        })

        hasil_dt = dt_model.predict(data)[0]
        hasil_rf = rf_model.predict(data)[0]

        col1, col2 = st.columns(2)

        col1.success(
            f"Decision Tree : {hasil_dt}"
        )
        

        col2.success(
            f"Random Forest : {hasil_rf}"
        )
        
        st.markdown(f"""
        ### Interpretasi

        Berdasarkan karakteristik data yang dimasukkan, model Random Forest memprediksi bahwa daerah tersebut termasuk dalam kategori **{hasil_rf}**. Hasil ini dapat digunakan sebagai gambaran awal dalam proses pengambilan keputusan terkait kebijakan ketenagakerjaan.
        """)

# =====================================================
# DAERAH PRIORITAS
# =====================================================

elif menu == "Daerah Prioritas":

    st.title("🎯 Daerah Prioritas")
    
    st.info("""
    Daerah prioritas merupakan kabupaten/kota yang diprediksi berada pada kategori Tingkat Pengangguran Terbuka (TPT) Tinggi. Daerah-daerah ini berpotensi membutuhkan perhatian lebih dalam penyusunan program ketenagakerjaan dan penciptaan lapangan kerja.
    """)

    X = df[
        ["IPM", "TPAK", "PDRB", "Tahun"]
    ]

    df["Prediksi"] = rf_model.predict(X)

    prioritas = df[
        df["Prediksi"] == "Tinggi"
    ]

    st.dataframe(
        prioritas[
            [
                "Kabupaten",
                "Tahun",
                "Prediksi"
            ]
        ],
        use_container_width=True
    )
    
    st.markdown(f"""
    ### Kesimpulan

    Berdasarkan hasil prediksi menggunakan algoritma Random Forest, terdapat **{len(prioritas)} data kabupaten/kota** yang termasuk kategori TPT Tinggi. Daerah-daerah tersebut dapat dijadikan prioritas dalam penyusunan program peningkatan kesempatan kerja, pelatihan tenaga kerja, serta pengembangan sektor ekonomi produktif.
    """)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

elif menu == "Feature Importance":

    st.title("Feature Importance Random Forest")

    st.info("""
    Feature Importance digunakan untuk mengetahui seberapa besar pengaruh
    masing-masing variabel terhadap proses prediksi kategori Tingkat
    Pengangguran Terbuka (TPT). Semakin tinggi nilai importance, semakin
    besar kontribusi variabel tersebut dalam menentukan hasil prediksi model.
    """)

    X = df[
        ["IPM", "TPAK", "PDRB", "Tahun"]
    ]

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    })

    importance = importance.sort_values(
        "Importance",
        ascending=False
    )

    # Ubah ke persen
    importance["Persentase (%)"] = (
        importance["Importance"] * 100
    ).round(2)

    st.subheader("Ranking Variabel")

    st.dataframe(
        importance[
            ["Feature", "Persentase (%)"]
        ],
        use_container_width=True
    )

    # Grafik
    fig = px.bar(
        importance,
        x="Persentase (%)",
        y="Feature",
        orientation="h",
        text="Persentase (%)",
        title="Pengaruh Variabel terhadap Prediksi Kategori TPT"
    )

    fig.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # Variabel terpenting
    fitur_utama = importance.iloc[0]["Feature"]
    nilai_utama = importance.iloc[0]["Persentase (%)"]

    st.success(
        f"Variabel yang paling berpengaruh adalah **{fitur_utama}** "
        f"dengan kontribusi sebesar **{nilai_utama:.2f}%**."
    )

    st.markdown("---")

    st.subheader("Interpretasi")

    st.markdown(
        f"""
    Berdasarkan hasil perhitungan Feature Importance menggunakan algoritma
    Random Forest, variabel **{fitur_utama}** memiliki pengaruh paling besar
    dalam proses prediksi kategori Tingkat Pengangguran Terbuka (TPT).

    Hal ini menunjukkan bahwa perubahan pada variabel tersebut memiliki
    kontribusi yang lebih dominan dibandingkan variabel lainnya dalam
    menentukan apakah suatu kabupaten/kota termasuk kategori **Rendah**,
    **Sedang**, atau **Tinggi**.

    Informasi ini dapat digunakan sebagai bahan pertimbangan dalam
    penyusunan kebijakan ketenagakerjaan, terutama dengan memberikan
    perhatian lebih pada faktor-faktor yang memiliki pengaruh terbesar
    terhadap tingkat pengangguran.
    """
        )

    st.markdown("---")

    st.subheader("Kesimpulan")

    st.info(
        f"""
    Hasil analisis menunjukkan bahwa **{fitur_utama}** merupakan faktor
    yang paling berpengaruh terhadap kategori Tingkat Pengangguran Terbuka
    di kabupaten/kota Provinsi Jawa Timur berdasarkan model Random Forest.
    Oleh karena itu, peningkatan kinerja pada aspek tersebut berpotensi
    memberikan dampak yang signifikan terhadap penurunan tingkat pengangguran.
    """
        )