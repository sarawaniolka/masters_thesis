# Adversarial Attacks on Convolutional Neural Networks

## Introduction
This repository contains a collection of Julia files that demonstrate and implement various adversarial attack methods on images using a deep learning model. Adversarial attacks aim to perturb input data in such a way that it causes the model to make incorrect predictions. This README provides an overview of the files included in this reposirtory and instructions on how to use the main module to perform these attacks.

## Files Description

### 1. `model.jl`
This file defines a Julia module (`model_mod`) that contains functions for image preprocessing, loading a pre-trained deep learning model, and performing predictions. It also includes functions for loading ImageNet labels and calculating model parameters.

### 2. `FGSM_mod.jl`
The `FGSM_mod` module implements the Fast Gradient Sign Method (FGSM) attack, a white-box attack that perturbs an image to mislead a deep learning model. It defines functions for FGSM attacks, custom loss functions, and visualizations.

### 3. `CW_mod.jl`
The `CW_mod` module implements the Carlini-Wagner (CW) attack, a powerful adversarial attack that minimizes the L2 norm of the perturbation while ensuring misclassification. It includes functions for CW attacks, loss computation, and visualizations.

### 4. `onepix_mod.jl`
The `onepix_mod` module implements a one-pixel attack that modifies a minimal number of pixels to achieve misclassification. It defines functions for normalizing image data, visualizing attacks, and performing one-pixel attacks.

### 5. `main.jl`
This is the main module that demonstrates the use of the provided attack methods on an input image. It loads an image, preprocesses it, and performs FGSM, CW, and one-pixel attacks, comparing the predictions before and after the attacks.
