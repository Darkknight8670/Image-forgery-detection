# Image-forgery-detection


Image Forgery Detection with Enhanced EfficientNetB7
Overview
This repository provides a state-of-the-art pipeline for detecting image forgeries using a custom deep learning architecture based on EfficientNetB7, enhanced with advanced attention mechanisms. The project includes dataset preparation, model architecture, training, and evaluation scripts for robust binary classification of Authentic vs Tampered images.

Features
EfficientNetB7 Backbone: Utilizes a pre-trained EfficientNetB7 model for powerful feature extraction.

Custom Attention Blocks: Integrates Hybrid SE-CBAM and Enhanced Multi-Head Attention (MHA) blocks for improved spatial and channel-wise feature learning.

Mixed Precision Training: Leverages TensorFlow's mixed precision and XLA for faster training on modern GPUs.

Flexible Data Pipeline: Includes scripts to split datasets and apply advanced data augmentation.

Comprehensive Metrics: Tracks accuracy, AUC, precision, and recall during training.

Early Stopping & LR Scheduling: Automatically halts training on plateau and adjusts learning rates for optimal convergence.

Directory Structure

├── dataset/
│   └── dataset/
│       ├── Authentic/
│       └── Tampered/
├── split_dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── model/
├── logs/
├── scripts/
│   └── (all code files)
└── README.md
Getting Started
1. Prerequisites
Python 3.7+

TensorFlow 2.10–2.14 (see [TensorFlow Addons compatibility])

TensorFlow Addons

Keras

Google Colab or a local machine with a CUDA-enabled GPU recommended

Install dependencies:

bash
pip install tensorflow==2.14 keras tensorflow-addons scipy
2. Dataset Preparation
Organize your dataset as follows:

text
dataset/dataset/
    Authentic/
        img1.jpg
        img2.jpg
        ...
    Tampered/
        img1.jpg
        img2.jpg
        ...
Run the data splitting script to create train, validation, and test sets:

python
# In scripts/data_split.py
python data_split.py
This creates a split_dataset/ directory with subfolders for each split and class.

Model Architecture
Base: EfficientNetB7 (Imagenet weights, no top layer)

Attention: Hybrid SE-CBAM block + Enhanced Multi-Head Attention block after specific feature layers

Pooling: Combines global max and average pooling

Dense Layers: Two dense layers with Swish activation, followed by dropout

Output: Single sigmoid neuron for binary classification

Custom layers are implemented in scripts/attention_blocks.py and integrated in scripts/model.py.

Training
Train the model using the provided script:

python
# In scripts/train.py
python train.py
Key training settings:

Batch size: 16

Image size: 224x224

Loss: Binary cross-entropy

Optimizer: Adam (lr=1e-4, clipnorm=1.0)

Metrics: accuracy, AUC, precision, recall

Early stopping and learning rate reduction on plateau

TensorBoard logging enabled

Evaluation
Validation and test metrics are reported after each epoch.

Example performance (from logs):

Accuracy: >98%

AUC: >0.99

Precision/Recall: >0.97 on validation set after convergence.

Usage
To use the trained model for inference:

python
from tensorflow.keras.models import load_model
model = load_model('model/best_model.h5', custom_objects={
    'HybridSECBAM': HybridSECBAM,
    'EnhancedMHABlock': EnhancedMHABlock
})
# Predict on new images...
Notes
TensorFlow Addons: This project uses GroupNormalization from TensorFlow Addons. Ensure your TensorFlow and Addons versions are compatible (see [compatibility matrix]).

Mixed Precision: Mixed precision is enabled after model creation for optimal GPU performance.

Large Dataset: The pipeline is optimized for large datasets (e.g., 140,000+ images), but can be adapted for smaller datasets.

References
EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

CBAM: Convolutional Block Attention Module

TensorFlow Addons Compatibility

License
This project is licensed under the MIT License.

Acknowledgements
TensorFlow, Keras, and the open-source ML community.

: See warnings and compatibility notes in TensorFlow Addons documentation and logs in the provided code.

For questions or contributions, please open an issue or submit a pull request.

Example Model Summary (Excerpt):

Model: "model"

Layer (type) Output Shape Param # Connected to
input_1 (InputLayer) [(None, 224, 224, 3)] 0
...
Total params: 6,792,263
Trainable params: 6,725,440
Non-trainable params: 66,823
