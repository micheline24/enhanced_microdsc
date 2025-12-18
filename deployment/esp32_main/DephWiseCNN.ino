// FINAL CODE
#include <Arduino.h>
#include "driver/i2s.h"
#include "model.h"
#include "tflite_model.h"
#include "mel_weight_matrix.h"
#include "QuantizedMelPowerSpectrogram.h"
// #include "tensorflow/lite/micro/all_ops_resolver.h"

// Define I2S pin numbers for ESP32-S2
// THESE PINS ARE EXAMPLES. YOU MUST CONFIRM THEM FOR YOUR SPECIFIC BOARD
#define I2S_BCK_PIN 35
#define I2S_WS_PIN 36
#define I2S_SD_PIN 37

// Board Configuration
#if defined(CONFIG_IDF_TARGET_ESP32S2)
  #define LED_PIN 13 // Built-in LED pin for ESP32-S2
#else
  #define LED_PIN 13
#endif

// Audio Processing Constants
#define AUDIO_LENGTH 512
#define SAMPLE_RATE 16000
#define NUM_MEL_BINS 40
#define FRAME_LENGTH 480
#define FRAME_STEP 320
#define FFT_SIZE 256
#define TOP_DB 80

// Model and preprocessing setup
Model mlModel(tflite_model, 32 * 1024);
QuantizedMelPowerSpectrogram melPowerSpectrogram(
  49,             // width
  NUM_MEL_BINS,   // # of mel bins
  FRAME_LENGTH,   // frame length
  FRAME_STEP,     // frame step
  FFT_SIZE,       // FFT size
  TOP_DB,         // top dB
  mel_weight_matrix
);

 
short audioBuffer[AUDIO_LENGTH];
float smoothedBirdPrediction = 0;
const float DETECTION_THRESHOLD = 0.90;
const float SMOOTHING_FACTOR = 0.8;

// Function prototypes
void setupModel();
void setupI2S();
void readAudioSamples();
void processSamples();

void setupModel() {
  if (!mlModel.begin()) {
    Serial.println("Failed to initialize ML model!");
    while (1) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(100);
    }
  }

  if (!melPowerSpectrogram.begin()) {
    Serial.println("Failed to initialize Mel Power Spectrogram!");
    while (1) {
      digitalWrite(LED_PIN, !digitalRead(LED_PIN));
      delay(200);
    }
  }

  melPowerSpectrogram.setOutputScale(mlModel.inputScale());
  melPowerSpectrogram.setOutputZeroPoint(mlModel.inputZeroPoint());
}

void setupI2S() {
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = false
  };

  const i2s_pin_config_t pin_config = {
    I2S_BCK_PIN,
    I2S_WS_PIN,
    I2S_SD_PIN,
    I2S_PIN_NO_CHANGE
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
  i2s_set_clk(I2S_NUM_0, SAMPLE_RATE, I2S_BITS_PER_SAMPLE_32BIT, I2S_CHANNEL_MONO);
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 5000);

  Serial.println("Bird Pest Detector Starting...");

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  setupModel();
  setupI2S(); 
  
  Serial.println("Bird Pest Detector Ready!");
  digitalWrite(LED_PIN, LOW);
}

void readAudioSamples() {
  // Use a 32-bit buffer to match the I2S driver's output
  int32_t i2sBuffer[AUDIO_LENGTH]; 
  size_t bytes_read;

  i2s_read(I2S_NUM_0, (char*)i2sBuffer, sizeof(i2sBuffer), &bytes_read, portMAX_DELAY);

  // Convert 32-bit samples to 16-bit
  for (int i = 0; i < AUDIO_LENGTH; i++) {
    audioBuffer[i] = (short)(i2sBuffer[i] >> 16); 
  }
}

void processSamples() {
  melPowerSpectrogram.write(audioBuffer, AUDIO_LENGTH);
  melPowerSpectrogram.read(mlModel.input(), mlModel.inputBytes());

  float predictions[mlModel.numOutputs()];
  mlModel.predict(predictions);

  smoothedBirdPrediction = smoothedBirdPrediction * SMOOTHING_FACTOR + 
                         predictions[0] * (1.0 - SMOOTHING_FACTOR);
}

void loop() {
  readAudioSamples();
  processSamples();

  if (smoothedBirdPrediction > DETECTION_THRESHOLD) {
    Serial.println("Bird pest detected!");
    Serial.printf("Confidence: %.2f%%\n", smoothedBirdPrediction * 100);
    digitalWrite(LED_PIN, HIGH);
  } else {
    digitalWrite(LED_PIN, LOW);
  }

  delay(100);
}
