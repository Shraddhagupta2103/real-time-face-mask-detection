# Real-Time Face Mask Detection using Deep Learning

This project implements a real-time face mask detection system using deep learning and computer vision. It uses a convolutional neural network (CNN) based on MobileNetV2 to identify whether people in a webcam feed are wearing face masks or not.

## Dataset Used

Source: [Kaggle Face Mask Dataset by Omkar Gurav](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)  
Structure:  
- `with_mask/` (3,725 images)  
- `without_mask/` (3,828 images)  
Total: 7,553 labeled images

## Model Architecture

Base Model: MobileNetV2 (pretrained on ImageNet)  
Technique: Transfer learning  
Input Shape: 224 x 224 x 3  
Classifier Head:
- Global Average Pooling
- Dense layers with ReLU and Dropout
- Final layer with sigmoid activation  
Loss Function: Binary Crossentropy  
Optimizer: Adam  
Validation Accuracy: 98.01%

## Features

- Real-time detection from webcam using OpenCV
- Live bounding boxes and mask/no-mask prediction
- Displays confidence score
- Lightweight and fast — suitable for deployment

## Tech Stack

Python  
TensorFlow / Keras  
OpenCV  
NumPy  
Matplotlib  
PyCharm (IDE)

## Getting Started

1. Clone the repository:
git clone https://github.com/Shraddhagupta2103/real-time-face-mask-detection.git
cd real-time-mask-detection

2. Set up virtual environment:
python -m venv .venv

Activate:

- On Windows:
  ```
  .venv\Scripts\activate
  ```
- On macOS/Linux:
  ```
  source .venv/bin/activate
  ```

3. Install dependencies:
pip install -r requirements.txt

4. Download and prepare dataset from Kaggle:
Place folders like this:
dataset/data/with_mask/
dataset/data/without_mask/

5. Train the model:
python train_model.py

6. Run real-time detection:
python realtime_detection.py

## Results

- High model accuracy and low validation loss
- Real-time detection with fast frame processing
- Correct identification of multiple masked/unmasked faces

## What I Learned

- Transfer learning with MobileNetV2
- Data augmentation techniques
- Real-time image processing using OpenCV
- Saving and loading trained models in Keras

