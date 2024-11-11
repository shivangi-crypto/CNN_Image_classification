# CNN_Image_classification# **CNN Image Classification on CIFAR-10**

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

---

## Installation and Setup

### Prerequisites
- Python 3.7 or higher
- [PyTorch](https://pytorch.org/get-started/locally/) (for GPU/CPU processing)
- [Git](https://git-scm.com/)

### Installation Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/shivangi-crypto/CNN_Image_classification.git
   cd CNN_Image_classification
2. **Install Dependencies** (Install the required Python packages by running):
    pip install -r requirements.txt
4. **Download CIFAR-10 Dataset** (if not automatically downloaded in code):
   ```python
   from torchvision.datasets import CIFAR10
   CIFAR10(root='./data', download=True)
### Usage
1. **Training a Model**
2. **Evaluating a Model**
3. **Hyperparameter Tuning**
### Experiment Details
The experiments in this project were designed to analyze:

**Accuracy**: Training and validation accuracy on CIFAR-10.
**Convergence Rate**: Stability and speed of convergence.
**Computational Efficiency**: Comparison of resource usage across models.
### Hyperparameters
The following hyperparameters were tested:

**Optimizer**: Adam (default), SGD (optional)
**Learning Rates**: 0.001, 0.0001, 0.01
**Regularization**: Dropout in AlexNet and Basic CNN, residual connections in ResNet-18
All experiments included data augmentation (random flips, rotations, brightness) to boost generalization, and early stopping to prevent overfitting.
### Key Observations
## Results

The table below summarizes the test accuracy and computational efficiency of each model on the CIFAR-10 dataset.

| Model       | Test Accuracy | Computational Efficiency |
|-------------|---------------|--------------------------|
| Basic CNN   | 72.4%         | Moderate                 |
| ResNet-18   | 82.3%         | High                     |
| AlexNet     | 78.6%         | Moderate                 |
| MobileNet   | 80.1%         | Very High                |

### Key Observations
- **ResNet-18** achieved the highest accuracy, benefiting from residual connections.
- **MobileNet** demonstrated strong performance with reduced computational requirements, making it ideal for mobile or real-time applications.
- **AlexNet** displayed competitive accuracy, though computationally more intensive than MobileNet.
### Future Work
Future directions for this project include:

- **Testing additional lightweight models**: EfficientNet and SqueezeNet can be evaluated for efficiency.
- **Exploring advanced regularization**: Techniques like batch normalization could enhance performance.
- **Applying transfer learning**: Fine-tuning pretrained models on CIFAR-10 could improve accuracy.

