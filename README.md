# 🔍 Image Forgery Detection with Enhanced EfficientNetB7

**Image Forgery Detection** is a high-performance deep learning pipeline designed to identify manipulated (tampered) images with impressive precision and reliability. Leveraging the power of **EfficientNetB7** and **custom attention blocks**, the system excels at binary classification of authentic vs tampered images at scale.

---

## ✨ Features

- ✅ **EfficientNetB7 Backbone** for powerful feature extraction  
- 🧠 **Hybrid SE-CBAM Attention** for channel + spatial attention  
- 🔁 **Enhanced Multi-Head Attention** with window-based spatial focus  
- ⚙️ **Mixed Precision + XLA** for optimized GPU training  
- 🧪 **Advanced Data Augmentation** with photometric and geometric transforms  
- 📈 **Comprehensive Metrics** (Accuracy, AUC, Precision, Recall)  
- ⏱️ **EarlyStopping** and **LearningRateScheduler** for better generalization  
- 📊 **TensorBoard Logging** for real-time training insights  
- 📦 **Custom Data Split Script** for flexible training/validation/testing

---

## 🏗️ Tech Stack

| Category           | Tech Used                     |
|--------------------|-------------------------------|
| Model Backbone     | EfficientNetB7 (Imagenet)     |
| Attention Layers   | Hybrid SE-CBAM, MHA Block     |
| Framework          | TensorFlow, Keras             |
| Data Handling      | tf.data, ImageDataGenerator   |
| Metrics/Callbacks  | AUC, Precision, Recall, EarlyStopping |
| Visualization      | TensorBoard                   |
| Optimization       | Mixed Precision, XLA          |
| Deployment Ready   | -                             |

---

## 🗃️ Directory Structure

    ├── dataset/
       │ └── dataset/
             │ ├── Authentic/
             │ └── Tampered/


    ├── split_dataset/
    │ ├── train/
           ├── Authentic/
           └── Tampered/

    │ ├── val/
           ├── Authentic/
           └── Tampered/

    │ └── test/
       ├── Authentic/
       └── Tampered/





---

## 🧪 Getting Started

### 📦 Prerequisites

- Python 3.7+
- TensorFlow 2.10–2.14
- TensorFlow Addons
- Keras
- Compatible NVIDIA GPU or Google Colab

### 🔧 Installation

     ```bash
       pip install tensorflow==2.10 keras tensorflow-addons scipy





🧠 Model Architecture


🔹 EfficientNetB7 (no top layer, pretrained on ImageNet)

🔸 HybridSECBAM after initial convolutional blocks

🔸 EnhancedMHABlock inserted mid-to-late layers

🔹 Global Avg + Max Pooling

🔹 Dense layers with Swish activation + Dropout

🔸 Sigmoid output for binary classification



Training Settings:

| Parameter 	|  Value|
|---------------|---------|
| Image Size	| 224 x 224|
| Batch Size	| 16      |
| Optimizer	| Adam (lr=1e-4)|
| Loss	    | Binary CrossEntropy|
| Callbacks	|  EarlyStopping, LR Scheduler|
| Mixed Precision	| ✅      |



📝 Notes

🔄  Make sure TensorFlow Addons is version-compatible (check TF Addons Compatibility Matrix)

🖼️  Works with datasets of 140K+ images or smaller custom datasets

🧪  Designed for both research and production-level deployment






📚 References

EfficientNet: Rethinking Model Scaling

CBAM: Convolutional Block Attention Module

TensorFlow Addons
