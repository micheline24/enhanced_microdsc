About
This repository contains the complete implementation of “Enhanced MicroDSC (Microcontroller Depthwise Separable Convolution),” an optimized depthwise separable convolution architecture for acoustic pest bird detection on low-cost microcontrollers.

Research Paper:"Lightweight deep learning model for real-time acoustic bird pest detection on edge microcontrollers"

Our model enables real-time bird pest detection on resource-constrained ESP32 microcontrollers, providing an affordable precision agriculture solution for African smallholder farmers while addressing manual labor issues in bird monitoring.

Key Features
- **Ultra-lightweight architecture** optimized for ESP32 microcontrollers
- **Real-time acoustic detection** of 11 bird species including quelea and village weaver
- **Low power consumption** suitable for solar-powered field deployment
- **Offline operation** - no internet connectivity required
- **Cost-effective** solution for resource-constrained agricultural environments

Repository Structure
Experiments/
Contains ablation studies conducted to select the optimal model architecture and hyperparameters.

- Comparison of different depthwise separable convolution configurations
- Analysis of model complexity vs. accuracy trade-offs
- Performance evaluation across varying constraint scenarios

training/
Contains the complete model training pipeline and evaluation framework.

- Data preprocessing and feature extraction
- Enhanced MicroDSC model training
- Performance evaluation and validation
- Confusion matrix analysis and error characterization
- Model quantization for microcontroller deployment

deployment/
Contains ESP32 microcontroller implementation for real-time field deployment.

- Arduino IDE project files
- Optimized inference code for ESP32
- Audio acquisition and preprocessing
- Real-time classification pipeline
- Real-time notification, once bird pests are detected
**Deploy this** on ESP32 hardware for real-world bird pest classification.

Requirements

For Python Notebooks (experiments/ and training/)
tensorflow
keras
numpy
pandas
librosa
scikit-learn
matplotlib
seaborn

### For ESP32 Deployment (deployment/)
- **Hardware:** ESP32 development board, I2S microphone (INMP441)
- **Software:** Arduino IDE 2.0+
- **Libraries:** ESP32 board support, required audio libraries

Quick Start
1. Explore the Ablation Study
```bash
cd experiments/
Jupyter Notebook ablation_study.ipynb
```

 2. Train the Model
```bash
cd training/
jupyter notebook model_training.ipynb
```

3. Deploy on ESP32
```bash
cd deployment/
# Open esp32_main/esp32_main.ino in Arduino IDE
# Follow instructions in deployment/README.md
```

 Model Performance

- **Target Species:** 10 bird species 
- **Inference Time:** <100ms on ESP32
- **Model Size:** Optimized for ESP32 constraints

Author

Micheline Kazeneza  
PhD Student, African Center of Excellence in Internet of Things, 
Regional Scholarship and Innovation Fund (RSIF) fellow
University of Rwanda  

Contact

For questions or collaboration opportunities, please contact: Micheline.kazeneza@ub.edu.bi

Acknowledgments

This research was supported by the Regional Scholarship and Innovation Fund (RSIF) program and conducted at the University of Rwanda under the management of ICIPE.
This research was conducted with financial support from the ICIPE–World Bank Financing Agreement No. D347-3A and the World Bank–Korea Trust Fund Agreement No. TF0A8639 under the PASET Regional Scholarship and Innovation Fund.

**Keywords:** Bird pest detection, Edge AI, Microcontroller deployment, Acoustic classification, Precision agriculture, ESP32, Depthwise separable convolution, TinyML
