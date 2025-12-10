<div align="center">
  
# Embedded AI Final Project
## Autoencoder for Audio Anomaly Detection

<br>

This project utilizes an autoencoder model trained on the [DCASE "Unsupervised Anomalous Sound Detection for Machine Condition Monitoring under Domain Shifted Conditions"](https://www.kaggle.com/datasets/pythonafroz/electrical-motor-anomaly-detection-from-sound-data) data set.

For testing purposes, only the *fan* data was used instead of the gearbox, valve or pump data. This was to decrease overall model size and make it easier to create an autoencoder, as this project is a proof of concept for running neural networks on edge devices.

The model was designed with the [Raspberry Pi 3 B+](https://www.raspberrypi.com/products/raspberry-pi-3-model-b-plus/) in mind, and was tested directly on device. This was to demonstrate the effectiveness of my model compression/quantization as well as the efficiency of the model's inference step when processing audio. The Raspberry Pi 3 B+ has the following specifications:

| **CPU**             | **RAM**          | **Storage**  | **LAN**             | **USB**    | **OS**          |
|---------------------|------------------|--------------|---------------------|------------|-----------------|
| 64-Bit SoC @ 1.4GHz | 1GB LPDDR2 SDRAM | 64GB MicroSD | 2.4GHz wireless LAN | 4x USB 2.0 | Raspberry Pi OS |

<br> 

For the tech stack itself, a variety of platforms, deployment environments & interpreters were used to get the model up and running:

| **Language** | **Primary ML Package** | **Quantization Package/Interpreter** | **IDE(s)**                                                      | **Misc. Packages**                            |
|--------------|------------------------|--------------------------------------|-----------------------------------------------------------------|-----------------------------------------------|
| Python       | TensorFlow (Keras)     | TFLite (TensorFlow Lite)             | Visual Studio Code (inference) Google Colab Notebook (training) | Numpy, librosa, sounddevice, tkinter, psutil  |

<br>

## Use Cases and Safety

<br>

The general use case for an autoencoder such as this would be that of [predictive maintenance](https://www.ibm.com/think/topics/predictive-maintenance), which involves using AI models and sensors to determine the health of equipment. In this particular case, monitoring
the overall health of a desk fan is not entirely practical but demonstrates how such a model can be run on a small, cheap device such as a Raspberry Pi. Regardless, should a user want to run this model to monitor a fan, it absolutely can be done (with some adjustments, of course - more on that later).

There are plenty of safety concerns regarding an anomaly detection model, especially in the area of audio privacy. To have a persistent monitor for a device, the audio stream must always be running, meaning that potential conversations or monologue may be recorded. While
the audio itself is never actually stored anywhere, the simple fact of having persistent audio input means there could be plenty of security issues. For example, if a malicious actor would choose to gain access to the microphone/input device, they could easily capture or listen
in on the audio stream that the model is reading in. With this in mind, if an anomaly detection model is to ever be used in practice, all potential occupants of the space that the input device resides in **must** be notified of the audio stream.

<br>

## Autoencoders and MFCC

<br>

An autoencoder model is a unique model type, since it does not look for correct classification (accuracy) or MSE (in the traditional sense). Instead, autoencoders will only be trained on audio that is considered "normal" - then, using an input stream or validation set, the
encoder will attempt to reconstruct the signal and compares it to its training data, generating a **reconstruction MSE**. This MSE metric helps in determining the proper **threshold** for the model, which is essentially a limit for how high the MSE the model will allow before it detects
an anomaly. 

This works in three steps - the autoencoder compresses the data into "bottleneck" representation, then learns the most effective and essential parts of the data. Then, the model will attempt to reconstruct the data based on what was learned, and will then output the 
reconstruction loss function (MSE in this case). 

</div>
