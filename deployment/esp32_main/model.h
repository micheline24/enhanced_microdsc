#ifndef MODEL_H_
#define MODEL_H_

#include <Arduino.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

class Model {
  public:
    Model(const unsigned char* tfLiteModel, int tensorArenaSize) :
      _tfLiteModel(tfLiteModel),
      _tensorArenaSize(tensorArenaSize),
      _tensorArena(nullptr),
      _interpreter(nullptr)
    {
    }

    ~Model() {
      end();
    }

    int begin() {
      _tensorArena = new uint8_t[_tensorArenaSize];
      if (_tensorArena == nullptr) {
        return 0;
      }

      // Create a MicroMutableOpResolver and add the required ops
      static tflite::MicroMutableOpResolver<3> resolver; 
      resolver.AddRelu();
      resolver.AddAveragePool2D();
      resolver.AddFullyConnected();
      
      static tflite::ErrorReporter* error_reporter = nullptr;
      
      // Create interpreter with all required parameters
      _interpreter = new tflite::MicroInterpreter(
        tflite::GetModel(_tfLiteModel),
        resolver,
        _tensorArena,
        _tensorArenaSize
      );

      if (_interpreter->AllocateTensors() != kTfLiteOk) {
        return 0;
      }

      return 1;
    }
    // Rest of the code remains the same...
    void end() {
      if (_interpreter != nullptr) {
        delete _interpreter;
        _interpreter = nullptr;
      }
      if (_tensorArena != nullptr) {
        delete[] _tensorArena;
        _tensorArena = nullptr;
      }
    }

    float inputScale() const {
      return _interpreter->input(0)->params.scale;
    }

    int32_t inputZeroPoint() const {
      return _interpreter->input(0)->params.zero_point;
    }

    int8_t* input() const {
      TfLiteTensor* inputTensor = _interpreter->input(0);
      return inputTensor->data.int8;
    }

    size_t inputBytes() const {
      TfLiteTensor* inputTensor = _interpreter->input(0);
      return inputTensor->bytes;
    }

    int numOutputs() const {
      TfLiteTensor* outputTensor = _interpreter->output(0);
      return outputTensor->dims->data[1];
    }

    void predict(float predictions[]) {
      TfLiteTensor* outputTensor = _interpreter->output(0);
      
      if (_interpreter->Invoke() != kTfLiteOk) {
        for (int i = 0; i < outputTensor->dims->data[1]; i++) {
          predictions[i] = NAN;
        }
        return;
      }

      for (int i = 0; i < outputTensor->dims->data[1]; i++) {
        float y_quantized = outputTensor->data.int8[i];
        float y = (y_quantized - outputTensor->params.zero_point) * outputTensor->params.scale;
        predictions[i] = y;
      }
    }

  private:
    const unsigned char* _tfLiteModel;
    int _tensorArenaSize;
    uint8_t* _tensorArena;
    tflite::MicroInterpreter* _interpreter;
};

#endif // MODEL_H_