# CNN_Image_classification
CNN Image Classification on CIFAR-10

Table of Contents
Project Overview
Dataset
Architectures Implemented
Project Structure
Installation and Setup
Usage
Experiment Details
Results
Future Work
Contributing
License
Project Overview
This project investigates various Convolutional Neural Network (CNN) architectures—Basic CNN, ResNet-18, AlexNet, and MobileNet—for image classification on the CIFAR-10 dataset. The models were built from scratch, incorporating custom data augmentation, hyperparameter tuning, and architecture-specific optimizations. Our objective is to identify a model that achieves a balance between accuracy and computational efficiency, suitable for resource-constrained environments.

Key Objectives:
Compare the performance of different CNN architectures on CIFAR-10.
Optimize models through data augmentation and hyperparameter tuning.
Analyze the computational efficiency of each model for practical applications.
Dataset
The CIFAR-10 dataset, a benchmark dataset for image classification, contains:

Training images: 50,000
Testing images: 10,000
Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
Image Size: 32x32 RGB
Data preprocessing steps included normalization and data augmentation (random flipping, rotation, brightness adjustments) to enhance model generalization.

Architectures Implemented
Basic CNN: A standard convolutional neural network with sequential layers, acting as a baseline model.
ResNet-18: Includes residual connections to address gradient issues in deeper networks, promoting stability and convergence.
AlexNet: An early CNN model with dropout regularization to reduce overfitting, modified for CIFAR-10.
MobileNet: Utilizes depthwise separable convolutions, making it computationally efficient for mobile and real-time applications.
Each model was trained from scratch, enabling a fair comparison of their classification performance on CIFAR-10.

Project Structure
bash
Copy code
├── data/                       # Directory for CIFAR-10 data
├── models/                     # Model architectures (Basic CNN, ResNet-18, AlexNet, MobileNet)
│   ├── basic_cnn.py
│   ├── resnet18.py
│   ├── alexnet.py
│   └── mobilenet.py
├── augmentation/               # Custom data augmentation functions
│   └── augment.py
├── training/                   # Training scripts and configurations
│   └── train_model.py
├── evaluation/                 # Evaluation and result visualization
│   └── evaluate.py
├── utils/                      # Helper functions
│   └── utils.py
├── README.md                   # Project README file
└── requirements.txt            # Python dependencies
Installation and Setup
Prerequisites
Python 3.7 or higher
PyTorch (for GPU/CPU processing)
Git
Installation Steps
Clone the Repository:

bash
Copy code
git clone https://github.com/shivangi-crypto/CNN_Image_classification.git
cd CNN_Image_classification
Install Dependencies:

bash
Copy code
pip install -r requirements.txt
Download CIFAR-10 Dataset (optional, if not automated in the code):

python
Copy code
from torchvision.datasets import CIFAR10
CIFAR10(root='./data', download=True)
Usage
Training a Model: Run the following script to train a model:

bash
Copy code
python training/train_model.py --model resnet18 --epochs 50 --batch_size 32 --learning_rate 0.001
Replace --model with one of basic_cnn, resnet18, alexnet, or mobilenet to select the desired architecture.

Evaluation: To evaluate a trained model on the test set, run:

bash
Copy code
python evaluation/evaluate.py --model resnet18 --checkpoint ./checkpoints/resnet18_best.pth
Hyperparameter Tuning: To experiment with different hyperparameters, modify train_model.py or use command-line arguments, such as:

bash
Copy code
python training/train_model.py --model alexnet --learning_rate 0.0005 --optimizer sgd
Experiment Details
The experiment was designed to evaluate each model’s:

Accuracy on training and validation sets
Convergence rate and model stability
Computational efficiency for practical usage
Hyperparameters:

Optimizer: Adam (default), SGD (optional)
Learning Rates: Tested at 0.001, 0.0001, 0.01
Regularization: Dropout for AlexNet and Basic CNN, residual connections for ResNet-18
The experiments were conducted with early stopping based on validation loss to prevent overfitting. Data augmentation (random flips, rotations, brightness) was applied to improve generalization.

Results
The table below summarizes the test accuracy and efficiency across models.

Model	Test Accuracy	Computational Efficiency
Basic CNN	72.4%	Moderate
ResNet-18	82.3%	High
AlexNet	78.6%	Moderate
MobileNet	80.1%	Very High
Key Observations:
ResNet-18 achieved the highest accuracy, benefiting from residual connections.
MobileNet showed strong performance with a significant reduction in computational load, ideal for mobile or real-time applications.
AlexNet displayed effective accuracy with dropout, though computationally more intensive than MobileNet.
Future Work
Future directions include:

Exploring additional lightweight architectures: Models like EfficientNet or SqueezeNet could further improve efficiency.
Advanced regularization techniques: Applying batch normalization or other forms of regularization may enhance performance.
Transfer Learning: Fine-tuning on a larger or different dataset could yield improved results for CIFAR-10 classification.
Contributing
Contributions are welcome! Please follow these steps:

Fork the repository.
Create a new branch with a descriptive name.
Make your changes, and submit a pull request with detailed notes on your changes.
