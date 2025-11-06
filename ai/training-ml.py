import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import numpy as np
import librosa
import os
from tensorflow.lite.python.util import convert_bytes_to_c_source # Fungsi TFLite ke C Array

# --- KONFIGURASI PARAMETER ---
# Parameter ini HARUS SAMA saat pelatihan dan inferensi di ESP32
SR = 8000           # Sample Rate (ESP32 biasanya mendukung 8kHz atau 16kHz)
DURATION = 2        # Durasi potongan audio dalam detik
N_MFCC = 13         # Jumlah MFCC (dimensi fitur)
N_TIME_STEPS = 64   # Jumlah langkah waktu (DURATION * SR / hop_length) - Sesuaikan!

INPUT_SHAPE = (N_TIME_STEPS, N_MFCC, 1) # Shape Input Model CNN
CLASS_NAMES = ["Non-Tiger", "Tiger"] # Label Kelas

# --- BAGIAN 1: FUNGSI PRA-PEMROSESAN DATA AUDIO (Wajib di Proyek Nyata) ---

def extract_mfcc(file_path):
    """Memuat audio dan mengekstrak MFCC"""
    try:
        # Muat audio dengan sample rate yang ditentukan
        y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
    except Exception as e:
        print(f"Error memuat file {file_path}: {e}")
        return None

    # Ekstrak MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    
    # Normalisasi/Penyesuaian Ukuran: Ini KUNCI untuk memastikan semua input sama
    if mfccs.shape[1] < N_TIME_STEPS:
        # Lakukan Padding jika durasi terlalu pendek
        mfccs = np.pad(mfccs, ((0, 0), (0, N_TIME_STEPS - mfccs.shape[1])), 
                       mode='constant')
    else:
        # Potong jika durasi terlalu panjang
        mfccs = mfccs[:, :N_TIME_STEPS]

    # Transpose dan tambahkan dimensi channel untuk CNN: (Time, MFCC, 1)
    mfccs = mfccs.T[..., np.newaxis]
    return mfccs


# --- BAGIAN 2: DATA TRAINING (Memuat Data Suara Tiger) ---
print("Memuat data training dari folder sample-voice-tiger dan sample-voice-no-tiger ...")

TIGER_DIR = os.path.join(os.path.dirname(__file__), 'sample-voice-tiger')
NO_TIGER_DIR = os.path.join(os.path.dirname(__file__), 'sample-voice-non-tiger')

X_train = []
y_train = []
count_tiger = 0
count_no_tiger = 0

for fname in os.listdir(TIGER_DIR):
    if fname.lower().endswith('.wav'):
        fpath = os.path.join(TIGER_DIR, fname)
        mfcc = extract_mfcc(fpath)
        if mfcc is not None:
            X_train.append(mfcc)
            y_train.append(1)  # 1 = Tiger
            count_tiger += 1

if os.path.exists(NO_TIGER_DIR):
    for fname in os.listdir(NO_TIGER_DIR):
        if fname.lower().endswith('.wav'):
            fpath = os.path.join(NO_TIGER_DIR, fname)
            mfcc = extract_mfcc(fpath)
            if mfcc is not None:
                X_train.append(mfcc)
                y_train.append(0)  # 0 = Non-Tiger
                count_no_tiger += 1
else:
    print(f"Peringatan: Folder {NO_TIGER_DIR} tidak ditemukan. Data Non-Tiger tidak dimasukkan.")

print(f"Jumlah data Tiger: {count_tiger}")
print(f"Jumlah data Non-Tiger: {count_no_tiger}")

if count_tiger == 0 or count_no_tiger == 0:
    raise RuntimeError("Data kedua kelas harus ada! Tambahkan data Tiger dan Non-Tiger.")

X_train = np.array(X_train, dtype='float32')
y_train = np.array(y_train, dtype='int32')

if len(X_train) == 0:
    raise RuntimeError("Tidak ada data audio yang berhasil dimuat dari folder sample-voice-tiger dan sample-voice-no-tiger!")

# NORMALISASI: ke rentang 0-1
X_train = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
print(f"Bentuk data training: {X_train.shape}, Label: {y_train.shape}")
print("-" * 50)

# --- BAGIAN 3: MEMBANGUN DAN MELATIH MODEL CNN ---

print("Membangun Model CNN...")
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(len(CLASS_NAMES), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Langkah Pelatihan
print("Memulai Pelatihan Model...")
model.fit(X_train, y_train, epochs=50, batch_size=8, validation_split=0.2)
print("Pelatihan Selesai.")
print("-" * 50)


# --- BAGIAN 4: KONVERSI KE TENSORFLOW LITE UNTUK ESP32 ---


print("Memulai Konversi ke TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 4a. Optimasi Kuantisasi (Wajib untuk TinyML agar model kecil & cepat)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

# Representative dataset untuk kuantisasi INT8
def representative_data_gen():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1]]
converter.representative_dataset = representative_data_gen

tflite_model = converter.convert()

# 4b. Simpan TFLite Model ke File
tflite_filename = 'tiger_audio_model.tflite'
with open(tflite_filename, 'wb') as f:
    f.write(tflite_model)
print(f"Model TFLite disimpan di: {tflite_filename}")

# 4c. Konversi TFLite Model ke Array C (untuk deployment di ESP32)
c_array_filename = 'tiger_audio_model.h'
source_text, header_text = convert_bytes_to_c_source(
    tflite_model, 
    array_name="g_model",
    include_guard="MODEL_DATA_H"
)

# Tulis file header C
with open(c_array_filename, 'w') as f:
    f.write(header_text)
    f.write(source_text)

print(f"Konversi ke Array C selesai. File siap deploy: {c_array_filename}")
print("-" * 50)
print("SELANJUTNYA: Pindahkan file 'model_data.h' ke proyek C++ (Arduino/ESP-IDF) Anda.")