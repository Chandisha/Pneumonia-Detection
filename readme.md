## Pneumonia Detection from Chest X-Ray Images
This project uses a Convolutional Neural Network (CNN) to detect pneumonia in chest X-ray images. The model is trained using a dataset of X-ray images classified as either normal or pneumonia. After training, the model can predict whether a new chest X-ray belongs to a normal or pneumonia category.

## Table of Contents
1. Overview
2. Installation
3. Prerequisities
4. How to Use
5. Training the Model
6. Testing the Model
7. Results and Performance

## Overview
Chest X-rays are commonly used to detect pneumonia. Pneumonia can be challenging to diagnose, and a machine learning model can help assist medical professionals by automatically analyzing X-ray images. This project uses TensorFlow and Keras to build a deep learning model that classifies chest X-ray images as either normal or indicating pneumonia.

The model is trained using a dataset of chest X-ray images, and it classifies images into one of two categories:

1. NORMAL
2. PNEUMONIA
After training the model, it can be used to predict the class (NORMAL or PNEUMONIA) of new chest X-ray images.

## Installation
To get started with this project, you need to set up your development environment.

## Prerequisites
All the modules are mentioned in the requirements.txt file, to install the packages use the below command in your console:

                        pip install -r requirements.txt


## How to Use

# Training the Model
To train the model on the chest X-ray images, simply run the train.py script. It will use the images from the train folder to train the model. Use the below command in your console:

                        python scripts/train.py

The model will be saved in the model/ folder as saved_model.h5. This file contains the trained neural network that can be used for predictions.


# Testing the Model
Once the model is trained, you can test it using new images in the test folder. The test.py script loads the trained model and processes images in the test folder (which contains subfolders NORMAL and PNEUMONIA). It then prints out the prediction for each image. Use the below command in your console:

                        python scripts/test.py

The model will process each image in the test folder and print predictions in the terminal.

## Results and Performance
Accuracy: The model achieves reasonable accuracy for binary classification tasks.
Evaluation: Validation accuracy and loss are displayed during training.
