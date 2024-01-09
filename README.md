# DCGAN for Furniture Dataset
## Overview:

This script implements a Deep Convolutional Generative Adversarial Network (DCGAN) using PyTorch. The DCGAN is trained on a furniture dataset to generate new images that resemble the input dataset.

## Sections:
## 1. Initialization:

Sets the random seed for reproducibility.
Defines the root directory for the furniture dataset, number of workers, batch size, image size, and other hyperparameters.
Mounts Google Drive (if running in Google Colab) to access the dataset.

## 2. Dataset Loading:

Uses torchvision to create a custom dataset using the ImageFolder class, applying various transformations to the images (resizing, cropping, normalization).

## 3. Generator and Discriminator Model Definition:

Defines the Generator and Discriminator classes.
* Generator:
Uses transpose convolutions to upsample the input noise into realistic images.
Applies batch normalization and ReLU activation functions.
* Discriminator:
Performs convolutional operations to classify images as real or fake.
Applies LeakyReLU activation and batch normalization.

## 4. Model Initialization and Weight Initialization:

Initializes the models and applies custom weight initialization for convolutional and batch normalization layers.

## 5. Training Loop:

Implements the training loop for the DCGAN.
Alternates between updating the Discriminator and Generator.
Uses binary cross-entropy loss for adversarial training.
Saves generated images at intervals during training.
Displays and saves losses for both the Generator and Discriminator.
![](https://imgur.com/1D31v6M.png)

## 6. Visualization:

Visualizes the training images and the generated images at the end of training.
![](https://imgur.com/h8YROqG.png)
![](https://imgur.com/u9VFdxH.png)

## 7. Additional Notes:

The script supports GPU acceleration if available.
Weights for the Generator and Discriminator are saved after each epoch.
The script includes code for loading and displaying saved generated images.

# How to Use:

The script is designed to be run in a Jupyter notebook or a Python environment.
Modify the dataset path and parameters according to your dataset.
Execute the training loop to train the DCGAN model.
Check generated images and losses during and after training.
Additional Considerations:
Ensure that the dataset structure matches the expected folder hierarchy for the ImageFolder class.
Experiment with hyperparameters such as learning rate, batch size, and architecture for better results.
