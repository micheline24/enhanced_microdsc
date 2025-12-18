#ifndef QUANTIZED_MEL_POWER_SPECTROGRAM_H_
#define QUANTIZED_MEL_POWER_SPECTROGRAM_H_

#include "esp32-hal.h"
#include "esp_dsp.h"
#include <math.h>

class QuantizedMelPowerSpectrogram {
  public:
    QuantizedMelPowerSpectrogram(int width, int numMelBins, int frameLength, int frameStep, int fftSize, int topDb, const void* weightMatrix) :
      _width(width),
      _numMelBins(numMelBins),
      _frameLength(frameLength),
      _frameStep(frameStep),
      _fftSize(fftSize),
      _topDb(topDb),
      _weightMatrix((const float*)weightMatrix),
      _fftMagSize(fftSize / 2 + 1),
      _inputScale(1.0),
      _outputScale(1.0),
      _outputZeroPoint(0),
      _data(nullptr),
      _audioBuffer(nullptr),
      _audioBufferIndex(0),
      _window(nullptr)
    {
    }

    ~QuantizedMelPowerSpectrogram()
    {
      end();
    }

    int begin()
    {
      _data = new float[_width * _numMelBins];
      if (_data == nullptr) {
        return 0;
      }
      memset(_data, 0x00, _width * _numMelBins * sizeof(float));

      _audioBuffer = new float[_frameLength];
      if (_audioBuffer == nullptr) {
        return 0;
      }
      memset(_audioBuffer, 0x00, sizeof(float) * _frameStep);
      _audioBufferIndex = 0;

      _window = new float[_frameLength];
      if (_window == nullptr) {
        return 0;
      }
      
      // Calculate Hanning Window using ESP32's DSP
      for (int i = 0; i < _frameLength; i++) {
        _window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / _frameLength));
      }

      // Initialize ESP32 FFT
      if (dsps_fft2r_init_fc32(NULL, _fftSize) != ESP_OK) {
        return 0;
      }

      return 1;
    }

    void end()
    {
      if (_window != nullptr) {
        delete [] _window;
        _window = nullptr;
      }

      if (_audioBuffer != nullptr) {
        delete [] _audioBuffer;
        _audioBuffer = nullptr;
      }

      if (_data != nullptr) {
        delete [] _data;
        _data = nullptr;
      }
    }

    void setInputScale(float inputScale) {
      _inputScale = inputScale;
    }

    void setOutputScale(float scale) {
      _outputScale = scale;
    }

    void setOutputZeroPoint(int zeroPoint) {
      _outputZeroPoint = zeroPoint;
    }

    void write(const int16_t samples[], int count) {
      float fSamples[count + _frameLength];

      // Copy the last samples from last write to the start
      memcpy(fSamples, _audioBuffer, sizeof(float) * _audioBufferIndex);

      // Copy the new samples with scaling
      const int16_t* in16 = samples;
      float* outF = fSamples + _audioBufferIndex;

      for (int i = 0; i < count; i++) {
        *outF++ = (*in16++ / float(1 << 15)) * _inputScale;
      }

      // Calculate shift columns
      int shiftColumns = (count + _audioBufferIndex - (_frameLength - _frameStep)) / _frameStep;
      memmove(_data, _data + (shiftColumns * _numMelBins), 
              sizeof(float) * (_width - shiftColumns) * _numMelBins);

      const float* fIn = fSamples;
      outF = _data + (_width - shiftColumns) * _numMelBins;

      // Process new spectrogram columns
      float windowedInput[_fftSize];
      float fft[_fftSize * 2];
      float fftMag[_fftMagSize];

      for (int i = 0; i < shiftColumns; i++) {
        // Apply window
        for (int j = 0; j < _fftSize; j++) {
          windowedInput[j] = _window[j] * fIn[j];
        }

        // Perform FFT using ESP32's DSP
        dsps_fft2r_fc32(windowedInput, _fftSize);
        
        // Calculate magnitude
        for (int j = 0; j < _fftMagSize; j++) {
          float real = windowedInput[2*j];
          float imag = windowedInput[2*j + 1];
          fftMag[j] = sqrtf(real * real + imag * imag);
        }

        const float* fMelWeightMatrix = _weightMatrix;

        // Calculate Mel bins
        for (int j = 0; j < _numMelBins; j++) {
          const float* fFftMag = fftMag;
          float mel = 0;

          // Dot product with Mel weights
          for (int k = 0; k < _fftMagSize; k++) {
            mel += (*fMelWeightMatrix * *fFftMag);
            fMelWeightMatrix++;
            fFftMag++;
          }

          // Calculate mel power
          float melPower = mel * mel;
          if (melPower < 1e-6f) {
            melPower = 1e-6f;
          }

          *outF++ = 10.0f * log10f(melPower);
        }

        fIn += _frameStep;
      }

      // Save remaining samples
      _audioBufferIndex = (count + _audioBufferIndex) - (shiftColumns * _frameStep);
      memcpy(_audioBuffer, fSamples + (shiftColumns * _frameStep), 
             sizeof(float) * _audioBufferIndex);
    }

    void read(int8_t* buffer, size_t count) const {
      float maxDb = _data[0];
      int maxDbIndex = 0;

      // Find maximum value
      for (int i = 1; i < _width * _numMelBins; i++) {
        if (_data[i] > maxDb) {
          maxDb = _data[i];
          maxDbIndex = i;
        }
      }

      float minOut = maxDb - _topDb;

      // Quantize output
      const float* fIn = _data;
      for (size_t i = 0; i < count; i++) {
        float out = *fIn++;
        if (out < minOut) {
          out = minOut;
        }

        int32_t quantized = (int32_t)((out / _outputScale) + _outputZeroPoint);
        buffer[i] = (int8_t)(quantized < -128 ? -128 : (quantized > 127 ? 127 : quantized));
      }
    }

    void clear() {
      memset(_data, 0x00, sizeof(float) * _width * _numMelBins);
      _audioBufferIndex = 0;
    }

  private:
    int _width;
    int _numMelBins;
    int _frameLength;
    int _frameStep;
    int _fftSize;
    int _topDb;
    const float* _weightMatrix;
    int _fftMagSize;

    float _inputScale;
    float _outputScale;
    int32_t _outputZeroPoint;

    float* _data;
    float* _audioBuffer;
    int _audioBufferIndex;
    float* _window;
};

#endif // QUANTIZED_MEL_POWER_SPECTROGRAM_H_