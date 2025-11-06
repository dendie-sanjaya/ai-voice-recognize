import sys
import os
import tensorflow as tf

print("--- Status Interpreter ---")
print("1. Lokasi Interpreter Python:")
print(sys.executable)

print("\n2. Versi TensorFlow yang ditemukan:")
print(tf.__version__)

print("\n3. Jalur Tempat TensorFlow Diimpor:")
print(os.path.dirname(tf.__file__))