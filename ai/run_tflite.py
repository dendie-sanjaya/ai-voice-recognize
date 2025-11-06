import numpy as np
import tensorflow as tf
import librosa
import sys
import os

# Untuk rekam audio
try:
    import sounddevice as sd
    from scipy.io.wavfile import write as wav_write
except ImportError:
    sd = None
    wav_write = None

# --- KONFIGURASI ---
SR = 8000
DURATION = 5
N_MFCC = 13
N_TIME_STEPS = 64
INPUT_SHAPE = (N_TIME_STEPS, N_MFCC, 1)
CLASS_NAMES = ["Non-Tiger", "Tiger"]

# --- LOAD MODEL TFLITE ---
TFLITE_MODEL_PATH = "tiger_audio_model.tflite"
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SR, duration=DURATION)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    if mfccs.shape[1] < N_TIME_STEPS:
        mfccs = np.pad(mfccs, ((0, 0), (0, N_TIME_STEPS - mfccs.shape[1])), mode='constant')
    else:
        mfccs = mfccs[:, :N_TIME_STEPS]
    mfccs = mfccs.T[..., np.newaxis]
    return mfccs

def predict_audio(file_path):
    mfcc = extract_mfcc(file_path)
    x = np.expand_dims(mfcc, axis=0).astype(np.float32)
    # Normalisasi sama seperti training (0-1)
    x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(output, axis=1)[0]
    return CLASS_NAMES[pred], float(np.max(output))

def record_audio(filename, duration=2, sr=8000):
    if sd is None or wav_write is None:
        print("sounddevice dan scipy belum terinstall. Install dengan: pip install sounddevice scipy")
        sys.exit(1)
    print(f"Merekam audio {duration} detik dari microphone...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    wav_write(filename, sr, audio)
    print(f"Audio terekam di {filename}")

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == '--mic':
        temp_wav = "mic_test.wav"
        record_audio(temp_wav, duration=DURATION, sr=SR)
        audio_file = temp_wav
    elif len(sys.argv) == 2:
        audio_file = sys.argv[1]
    else:
        print("Usage:")
        print("  python run_tflite.py <audio_file.wav>")
        print("  python run_tflite.py --mic   # untuk rekam dari microphone")
        sys.exit(1)
    label, conf = predict_audio(audio_file)
    print(f"Prediksi: {label} (confidence: {conf:.2f})")
    if 'temp_wav' in locals() and os.path.exists(temp_wav):
        os.remove(temp_wav)
