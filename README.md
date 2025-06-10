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
| Deployment Ready   | ✅                             |

---

## 🗃️ Directory Structure

