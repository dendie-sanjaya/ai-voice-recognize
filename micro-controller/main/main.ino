// Header Utama TFLite untuk ESP32 (Mengandung definisi penting)
#include <TensorFlowLite_ESP32.h> 

// Header Eksplisit TFLite Micro (Digunakan untuk memastikan semua definisi dimuat, 
// terutama jika ada yang terlewat di header utama)
#include <tensorflow/lite/micro/micro_error_reporter.h> 
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/c/common.h> // Untuk TfLiteStatus, kTfLiteOk, dll.

// Include model yang dikonversi dari Python
#include "tiger_audio_model.h"

// Konstanta
const int kTensorArenaSize = 60 * 1024; 
uint8_t tensor_arena[kTensorArenaSize];

// Deklarasi Global TFLite
tflite::MicroInterpreter* interpreter = nullptr; 
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// --- FUNGSI UTAMA UNTUK SETUP ---
void setup() {
    Serial.begin(115200);

    // 1. Dapatkan model dari array C
    const tflite::Model* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model skema tidak cocok!");
        return;
    }

    // 2. Error Reporter dan Resolver
    static tflite::MicroErrorReporter micro_error_reporter; 
    static tflite::AllOpsResolver resolver;

    // 3. Buat interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter); 
    interpreter = &static_interpreter;

    // 4. Alokasikan tensor
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    
    // PERBAIKAN BUG REPORT: Menggunakan Serial.println()
    if (allocate_status != kTfLiteOk) {
        Serial.println("ERROR: Alokasi tensor gagal! Tingkatkan kTensorArenaSize.");
        return;
    }

    // 5. Dapatkan pointer ke tensor input dan output
    input = interpreter->input(0);
    output = interpreter->output(0);
    Serial.println("Model TFLite berhasil dimuat di ESP32!");

    // Cek tipe data
    if (input->type != kTfLiteFloat32) { 
        Serial.println("PERINGATAN: Tipe input bukan Float32.");
    }
}

// --- FUNGSI UTAMA UNTUK INFERENSI/PREDIKSI ---
void loop() {
    // --- 1. Ambil data suara (Simulasi data) ---
    // GANTI BAGIAN INI DENGAN KODE I2S REAL
    const int input_size = input->bytes / sizeof(float); 
    
    // Mengisi tensor input dengan data acak
    for (int i = 0; i < input_size; i++) {
        input->data.f[i] = random(100) / 100.0f; 
    }

    // --- 2. Jalankan Inferensi ---
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        Serial.println("Inferensi (Invoke) gagal!");
        return;
    }

    // --- 3. Dapatkan Hasil Prediksi ---
    float prob_non_harimau = output->data.f[0]; 
    float prob_harimau = output->data.f[1];     

    if (prob_harimau > 0.8) { 
        Serial.println(">>>>> DETEKSI TIGER! <<<<<");
    } else {
        Serial.print("Not Trigger. Probabilitas Harimau: ");
        Serial.println(prob_harimau, 4); 
    }

    delay(2000); 
}