Sign Language Detector using CNN

Overview

This project implements a Sign Language Detector using Convolutional Neural Networks (CNNs). It trains a deep learning model on a dataset of sign language images to recognize different hand gestures.

Features

Image-based sign language detection

Trained using CNN architecture for high accuracy

Uses a dataset of labeled sign images

Deployable for real-time sign recognition

Dataset

The dataset consists of labeled images of hand gestures representing different letters, numbers, or words in sign language. Ensure the dataset is structured properly before training.

Installation

Clone the repository:

Install dependencies:

Model Training

Prepare your dataset by organizing images into respective class folders.

Run the training script:

The trained model will be saved as sign_language_model.h5.

Usage

To test the trained model:

For real-time detection using a webcam:

Dependencies

tensorflow

keras

opencv-python

numpy

matplotlib

Model Architecture

Convolutional Layers for feature extraction

Max-Pooling Layers for dimensionality reduction

Fully Connected Layers for classification

Softmax Activation for multi-class classification

later created a app using java and used the model in the app.
