import tensorflow as tf

print("=" * 50)
print("EVALUASI MODEL DETEKSI KEMATANGAN BUAH PISANG")
print("=" * 50)

# Load model
model = tf.keras.models.load_model("model/model_pisang.keras")

# Load dataset test
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "dataset/test",
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

print("\nNama Kelas :")
print(test_dataset.class_names)

# Prefetch agar lebih cepat
AUTOTUNE = tf.data.AUTOTUNE
test_dataset = test_dataset.prefetch(AUTOTUNE)

# Evaluasi
loss, accuracy = model.evaluate(test_dataset)

print("\n" + "=" * 50)
print("HASIL EVALUASI")
print("=" * 50)
print(f"Loss     : {loss:.4f}")
print(f"Accuracy : {accuracy*100:.2f}%")
print("=" * 50)