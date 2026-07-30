import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Sistem Deteksi Kematangan 1  Buah Pisang",
    page_icon="🍌",
    layout="wide"
)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/model_pisang.keras"
    )
model = load_model()
kelas = [
    "busuk",
    "matang",
    "mentah",
    "terlalu_matang"
]

nama_kelas = {
    "busuk": "Busuk",
    "matang": "Matang",
    "mentah": "Mentah",
    "terlalu_matang": "Terlalu Matang"
}

emoji = {
    "busuk": "🟤",
    "matang": "🟡",
    "mentah": "🟢",
    "terlalu_matang": "🟠"
}

deskripsi = {
    "busuk": """
### 🟤 Kondisi Pisang: Busuk

Pisang terdeteksi sudah mengalami **pembusukan** dan berada pada
kondisi yang tidak layak untuk dikonsumsi.

Pisang pada tahap ini biasanya menunjukkan perubahan warna yang cukup
ekstrem, seperti bercak cokelat tua hingga hitam yang semakin meluas.
Kulit dapat terlihat rusak, lembek, atau mengalami perubahan tekstur.
Pada kondisi pembusukan yang lebih lanjut, bagian dalam buah juga dapat
menjadi sangat lunak dan berair.

**Ciri-ciri yang umum ditemukan:**
- Banyak bercak cokelat tua atau hitam pada kulit.
- Tekstur buah sangat lembek.
- Kulit mulai rusak atau menghitam.
- Dapat muncul aroma yang tidak normal.
- Kondisi buah sudah melewati tingkat kematangan yang aman.

**Rekomendasi:**

Pisang yang sudah mengalami pembusukan sebaiknya **tidak dikonsumsi**,
terutama apabila terdapat bau menyengat, lendir, jamur, atau bagian
buah yang sudah rusak secara signifikan. Sebaiknya pisang dipisahkan
dari buah lainnya agar kondisi yang tidak diinginkan tidak menyebar.
""",

    "matang": """
### 🟡 Kondisi Pisang: Matang

Pisang terdeteksi berada pada kondisi **matang** dan umumnya sudah
**siap untuk dikonsumsi**.

Pada tingkat kematangan ini, pisang umumnya memiliki warna kulit kuning hingga kuning kehijauan, dan dapat mulai menunjukkan beberapa bercak cokelat atau gelap. Tekstur buah biasanya sudah lebih lunak dibandingkan pisang mentah, tetapi masih cukup padat dengan rasa dan aroma yang optimal.

**Ciri-ciri yang umum ditemukan:**
- Warna kulit kuning hingga kuning kehijauan.
- Dapat terdapat beberapa bercak cokelat atau gelap pada kulit.
- Tekstur buah lebih lunak dibandingkan pisang mentah.
- Rasa cenderung manis.
- Aroma pisang mulai terasa lebih kuat.
- Kondisi buah masih baik dan belum menunjukkan pembusukan.

**Rekomendasi:**

Pisang dalam kondisi ini **sangat cocok untuk langsung dikonsumsi**.
Pisang matang juga dapat digunakan untuk berbagai olahan seperti
smoothie, pisang goreng, pancake, maupun makanan penutup lainnya.

Jika belum ingin dikonsumsi, simpan pisang di tempat yang sejuk dan
kering. Hindari menyimpannya terlalu lama pada kondisi yang dapat
mempercepat proses pematangan.
""",

    "mentah": """
### 🟢 Kondisi Pisang: Mentah

Pisang terdeteksi masih dalam kondisi **mentah** dan belum mencapai
tingkat kematangan optimal.

Pisang mentah biasanya memiliki kulit berwarna hijau dan tekstur buah
yang masih cukup keras. Kandungan pati pada buah masih relatif tinggi,
sehingga rasanya cenderung kurang manis dibandingkan pisang yang sudah
matang.

**Ciri-ciri yang umum ditemukan:**
- Kulit berwarna hijau atau hijau kekuningan.
- Tekstur buah masih keras dan padat.
- Rasa cenderung lebih hambar atau sedikit sepat.
- Aroma pisang belum terlalu kuat.
- Belum mencapai tingkat kematangan optimal.

**Rekomendasi:**

Jika ingin mendapatkan rasa yang lebih manis dan tekstur yang lebih
lembut, pisang dapat **dibiarkan matang terlebih dahulu** pada suhu
ruang.

Proses pematangan biasanya ditandai dengan perubahan warna kulit dari
hijau menjadi kuning. Setelah warna mulai berubah menjadi kuning dan
teksturnya sedikit lebih lunak, pisang umumnya sudah lebih siap untuk
dikonsumsi.

Pisang mentah juga dapat digunakan sebagai bahan untuk beberapa jenis
olahan makanan tertentu.
""",

    "terlalu_matang": """
### 🟠 Kondisi Pisang: Terlalu Matang

Pisang terdeteksi sudah **melewati tingkat kematangan optimal**.
Kondisinya masih dapat berada pada tahap yang bisa dimanfaatkan,
tergantung pada kondisi fisik buah.

Pisang yang terlalu matang biasanya memiliki kulit dengan banyak
bercak cokelat atau hitam. Tekstur buah juga menjadi semakin lunak
karena proses pematangan terus berlangsung. Rasa pisang umumnya menjadi
lebih manis dan aroma pisang semakin kuat.

**Ciri-ciri yang umum ditemukan:**
- Banyak bercak cokelat atau hitam pada kulit.
- Kulit terlihat lebih gelap dibandingkan pisang matang.
- Tekstur buah lebih lunak.
- Rasa cenderung lebih manis.
- Aroma pisang lebih kuat.
- Belum tentu busuk apabila bagian dalam masih dalam kondisi baik.

**Rekomendasi:**

Pisang yang terlalu matang **masih dapat dikonsumsi apabila bagian
dalamnya masih baik**, tidak berbau aneh, tidak berlendir, dan tidak
ditumbuhi jamur.

Kondisi ini juga sangat cocok untuk dijadikan bahan olahan seperti
banana bread, smoothie, pancake, bolu pisang, atau makanan lainnya.
Kandungan rasa manis yang lebih tinggi dapat membuat pisang terlalu
matang menjadi bahan yang baik untuk berbagai olahan.

Namun, apabila sudah terdapat **jamur, bau busuk yang menyengat,
cairan yang tidak normal, atau tanda pembusukan lainnya**, sebaiknya
tidak dikonsumsi.
"""
}

def predict_image(image):

    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(
        image
    ).astype("float32")

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    hasil = model.predict(
        image_array,
        verbose=0
    )

    index = int(
        np.argmax(hasil[0])
    )

    confidence = float(
        hasil[0][index]
    ) * 100

    return (
        kelas[index],
        confidence,
        hasil[0]
    )

with st.sidebar:

    st.title("Sistem Deteksi Kematangan Buah Pisang")

    st.caption(
        "MUHAMMAD AGDAN WIRAYUDHA - TEKNIK INFORMATIKA 25"
    )

    st.divider()

    menu = st.radio(
        "MENU",
        [
            "Dashboard",
            "Deteksi",
            "Metode",
            "Evaluasi",
            "Tentang"
        ]
    )

    st.divider()

    st.caption(
        "Deep Learning • MobileNetV2"
    )


if menu == "Dashboard":

    st.title("Sistem Deteksi Kematangan Buah Pisang")

    st.subheader(
        "Dibuat Oleh Muhammad Agdan Wirayudha - Teknik Informatika 25"
    )

    st.write(
        "Pengelolahan Citra Digital berbasis "
        "Deep Learning untuk mengenali tingkat "
        "kematangan buah pisang."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Dataset",
            "13.478"
        )

    with col2:
        st.metric(
            "Kelas",
            "4"
        )

    with col3:
        st.metric(
            "Accuracy",
            "97.51%"
        )

    with col4:
        st.metric(
            "Model",
            "MobileNetV2"
        )

    st.divider()

    st.header(
        "Kategori Kematangan"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🟢 Mentah")

        st.write(
            deskripsi["mentah"]
        )

        st.subheader("🟡 Matang")

        st.write(
            deskripsi["matang"]
        )

    with col2:

        st.subheader("🟠 Terlalu Matang")

        st.write(
            deskripsi["terlalu_matang"]
        )

        st.subheader("🟤 Busuk")

        st.write(
            deskripsi["busuk"]
        )

elif menu == "Deteksi":

    st.title(
        "🔍 Deteksi Kematangan"
    )

    st.write(
        "Upload gambar atau gunakan kamera."
    )

    sumber = st.radio(
        "Sumber gambar:",
        [
            "📁 Upload Gambar",
            "📷 Kamera"
        ],
        horizontal=True
    )

    gambar = None

    if sumber == "📁 Upload Gambar":

        file = st.file_uploader(
            "Pilih gambar pisang",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp"
            ]
        )

        if file:
            gambar = Image.open(file)

    else:

        kamera = st.camera_input(
            "Ambil foto pisang"
        )
        if kamera:
            gambar = Image.open(kamera)
    if gambar:

        st.divider()

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "📷 Gambar"
            )

            st.image(
                gambar,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "🤖 Hasil Prediksi"
            )

            with st.spinner(
                "AI sedang menganalisis..."
            ):

                hasil, confidence, probabilitas = (
                    predict_image(gambar)
                )

            st.success(
                f"{emoji[hasil]} "
                f"{nama_kelas[hasil]}"
            )

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

            st.divider()

            st.write(
                "### Probabilitas"
            )

            for i, nama in enumerate(kelas):

                nilai = float(
                    probabilitas[i]
                )

                st.write(
                    f"{emoji[nama]} "
                    f"{nama_kelas[nama]}"
                )

                st.progress(
                    min(max(nilai, 0), 1),
                    text=f"{nilai * 100:.2f}%"
                )

            st.info(
                deskripsi[hasil]
            )

elif menu == "Metode":

    st.title(
        "🧠 Metode"
    )

    st.write(
        "Teknologi yang digunakan dalam sistem."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "🤖 Deep Learning"
        )

        st.write(
            "Digunakan untuk mempelajari pola "
            "visual dari gambar buah pisang."
        )

    with col2:

        st.subheader(
            "🧠 CNN"
        )

        st.write(
            "CNN (Convolutional Neural Network) digunakan "
            "untuk ekstraksi fitur citra."
        )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "⚡ Transfer Learning"
        )

        st.write(
            "Memanfaatkan model yang telah dilatih "
            "sebelumnya untuk mempercepat training."
        )

    with col2:

        st.subheader(
            "📱 MobileNetV2"
        )

        st.write(
            "Arsitektur CNN yang ringan dan efisien "
            "untuk klasifikasi gambar."
        )

    st.divider()

    st.subheader(
        "🔄 Alur Sistem"
    )

    st.write(
        "📷 Gambar → "
        "🧠 MobileNetV2 → 📊 Klasifikasi → 🍌 Hasil"
    )

elif menu == "Evaluasi":

    st.title(
        "📊 Evaluasi Model"
    )

    st.write(
        "Performa model berdasarkan dataset test."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Accuracy",
            "97.51%"
        )

    with col2:

        st.metric(
            "Loss",
            "0.0848"
        )

    with col3:

        st.metric(
            "Test Data",
            "562"
        )

    st.divider()

    st.subheader(
        "Kelas yang Dideteksi"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info(
            "🟢 Mentah\n\n"
            "Belum matang."
        )

    with col2:
        st.success(
            "🟡 Matang\n\n"
            "Siap dikonsumsi."
        )

    with col3:
        st.warning(
            "🟠 Terlalu Matang\n\n"
            "Melewati kematangan optimal."
        )

    with col4:
        st.error(
            "🟤 Busuk\n\n"
            "Mengalami kerusakan."
        )
elif menu == "Tentang":

    st.title(
        "ℹ️ Tentang PisangCek"
    )

    st.subheader(
        "🍌 PisangCek"
    )

    st.write(
        "PisangCek merupakan aplikasi klasifikasi "
        "citra yang digunakan untuk mendeteksi "
        "tingkat kematangan buah pisang secara otomatis."
    )

    st.divider()

    st.subheader(
        "Teknologi"
    )

    st.write(
        "• Python\n"
        "• TensorFlow\n"
        "• Keras\n"
        "• CNN\n"
        "• MobileNetV2\n"
        "• Streamlit"
    )

    st.divider()

    st.metric(
        "Test Accuracy",
        "97.51%"
    )