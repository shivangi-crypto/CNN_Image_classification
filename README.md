# CNN_Image_classification# **CNN Image Classification on CIFAR-10**

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Architectures Implemented](#architectures-implemented)
4. [Project Structure](#project-structure)
5. [Installation and Setup](#installation-and-setup)
6. [Usage](#usage)
7. [Experiment Details](#experiment-details)
8. [Results](#results)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License](#license)

---

## Project Overview

This project investigates various Convolutional Neural Network (CNN) architectures—Basic CNN, ResNet-18, AlexNet, and MobileNet—for image classification on the CIFAR-10 dataset. The models were built from scratch, incorporating custom data augmentation, hyperparameter tuning, and architecture-specific optimizations. Our objective is to identify a model that achieves a balance between accuracy and computational efficiency, suitable for resource-constrained environments.

### Key Objectives:
- Compare the performance of different CNN architectures on CIFAR-10.
- Optimize models through data augmentation and hyperparameter tuning.
- Analyze the computational efficiency of each model for practical applications.

---

## Dataset

The **CIFAR-10** dataset, a benchmark dataset for image classification, contains:
- **Training images**: 50,000
- **Testing images**: 10,000
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size**: 32x32 RGB

Data preprocessing steps included normalization and data augmentation (random flipping, rotation, brightness adjustments) to enhance model generalization.

---

## Architectures Implemented

1. **Basic CNN**: A standard convolutional neural network with sequential layers, acting as a baseline model.
2. **ResNet-18**: Includes residual connections to address gradient issues in deeper networks, promoting stability and convergence.
3. **AlexNet**: An early CNN model with dropout regularization to reduce overfitting, modified for CIFAR-10.
4. **MobileNet**: Utilizes depthwise separable convolutions, making it computationally efficient for mobile and real-time applications.

Each model was trained from scratch, enabling a fair comparison of their classification performance on CIFAR-10.

---

## Project Structure

