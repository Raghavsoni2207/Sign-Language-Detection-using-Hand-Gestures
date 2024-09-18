
# Sign Language Detection using Hand Gestures

This project focuses on detecting hand gestures and mapping them to corresponding letters of the English alphabet (A-Z). Using a webcam, the system captures real-time hand gestures, processes the image using **MediaPipe**, and predicts the letter using a **Random Forest Classifier**.

## Table of Contents
- [Overview](#overview)
- [Setup](#setup)
- [How It Works](#how-it-works)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Real-Time Gesture Prediction](#real-time-gesture-prediction)
- [Results](#results)
- [Improvements](#improvements)

---

## Overview

The goal of this project is to recognize hand gestures using machine learning and computer vision techniques. By utilizing the **MediaPipe** library, we extract hand landmarks and train a model to classify these gestures into the letters A-Z. This project demonstrates a practical application of real-time hand gesture detection, which could be further extended to sign language interpretation or human-computer interaction.

## Setup

### Dependencies
Before running the project, ensure that the following Python libraries are installed:

```bash
pip install mediapipe opencv-python matplotlib scikit-learn
```

### Running the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/username/sign-language-detection.git
   ```
2. Navigate into the project directory:
   ```bash
   cd sign-language-detection
   ```
3. Collect training data by running:
   ```bash
   python collect_data.py
   ```
4. Train the model:
   ```bash
   python train_model.py
   ```
5. Run the real-time prediction system:
   ```bash
   python predict_real_time.py
   ```

## How It Works

The project consists of three main stages:
1. **Data Collection**: We capture images of hand gestures and store them in a labeled format for training.
2. **Model Training**: Using the labeled hand landmark data, a **RandomForestClassifier** is trained to classify each gesture.
3. **Real-Time Gesture Prediction**: The trained model is used to predict the gesture shown to the webcam in real-time.

### Technologies Used
- **OpenCV**: For webcam access and image processing.
- **MediaPipe**: To detect hand landmarks.
- **Scikit-learn**: For machine learning model training (Random Forest).
- **Matplotlib**: For plotting the results (optional).

## Data Collection

The project collects data by capturing hand gestures through a webcam. Each gesture corresponds to a letter of the alphabet (A-Z), and the images are saved under respective class directories.

- **Image Size**: Each image is captured in real-time using OpenCV.
- **Landmark Extraction**: Hand landmarks (21 points) are extracted for each gesture using MediaPipe.

## Model Training

The model is trained using the collected hand landmark data. The features are normalized and passed into a **RandomForestClassifier** to distinguish between the 26 letters.

### Key Points
- The model uses 84 features (x, y coordinates of 21 landmarks).
- The data is split into training and testing sets to evaluate model performance.
- **Accuracy**: After training, the model achieves around **90% accuracy** on the test data.

## Real-Time Gesture Prediction

Once the model is trained, it can be used to predict hand gestures in real-time via webcam. The prediction results are displayed on the screen, along with a bounding box around the detected hand.

## Results

- **Model Accuracy**: The model performs well with an accuracy of **90%**.
- **Real-Time Prediction**: The system detects and classifies hand gestures with minimal delay.

## Improvements

Possible future enhancements:
- **Add more gestures**: Extend to two-hand gestures or more sign language gestures.
- **Optimize Performance**: Improve real-time performance with techniques like multithreading.
- **Improve Model Accuracy**: Fine-tune the model or try other machine learning algorithms like CNN.