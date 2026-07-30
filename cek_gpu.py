import tensorflow as tf

print("=" * 40)
print("CEK GPU")
print("=" * 40)

print("TensorFlow Version :", tf.__version__)
print("Jumlah GPU :", len(tf.config.list_physical_devices('GPU')))
print("Daftar GPU :", tf.config.list_physical_devices('GPU'))