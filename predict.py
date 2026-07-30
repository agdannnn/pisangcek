import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

print("=" * 50)
print("DETEKSI KEMATANGAN BUAH PISANG")
print("=" * 50)

# Load model
model = tf.keras.models.load_model("model/model_pisang.keras")

# Nama kelas
kelas = [
    "busuk",
    "matang",
    "mentah",
    "terlalu matang"
]

# Input gambar
nama_gambar = input("\nMasukkan nama gambar : ")

path = "test/" + nama_gambar

gambar = cv2.imread(path)

if gambar is None:
    print("\nGambar tidak ditemukan!")
    exit()

# Tampilkan gambar
gambar_rgb = cv2.cvtColor(gambar, cv2.COLOR_BGR2RGB)

plt.imshow(gambar_rgb)
plt.title("Gambar Uji")
plt.axis("off")
plt.show()

# Resize
gambar = cv2.resize(gambar_rgb, (224, 224))

# Normalisasi
gambar = gambar.astype("float32")

# Tambah dimensi
gambar = np.expand_dims(gambar, axis=0)

# Prediksi
hasil = model.predict(gambar, verbose=0)

print("\nProbabilitas setiap kelas:")
for i, nama in enumerate(kelas):
    print(f"{nama:18} : {hasil[0][i]*100:.2f}%")

index = np.argmax(hasil)
confidence = hasil[0][index] * 100

print("\n" + "=" * 50)
print("HASIL PREDIKSI")
print("=" * 50)

print("Prediksi :", kelas[index])
print(f"Confidence : {confidence:.2f}%")

print("=" * 50)