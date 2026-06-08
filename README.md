# Neural OCR Engine: Custom CNN Handwritten Text Recognition 📝

An advanced, end-to-end Computer Vision and Deep Learning application that segmentizes and recognizes handwritten alphanumeric sentences across multiple lines. Built as part of my Machine Learning Internship at CodeAlpha, this project bypasses generic pre-built APIs to implement a custom-trained Convolutional Neural Network (CNN) combined with an adaptive OpenCV layout analysis pipeline.

## 🚀 Live Studio Dashboard Preview
The application deploys a high-end, dark-themed Streamlit vision workspace featuring:
1. **Adaptive Pre-processing:** Automatically detects image background brightness to handle both digital dark-mode screenshots and physical light-background notebook paper scans cleanly.
2. **Character Isolation Matrices:** Uses OpenCV contour analysis to draw precise green bounding boxes around individual handwritten characters.
3. **Advanced Layout Restoration:** Implements line-grouping algorithms and vertical overlap filters to correctly preserve multi-line structures and spacing.
4. **Heuristic Post-Processing:** Translates raw neural confidence arrays into sanitized text strings, correcting case and font-scale ambiguities on the fly.

---

## 🧠 System Architecture & Engineering

### 1. The Deep Learning Model (`train_cnn.py`)
* **Framework:** TensorFlow / Keras
* **Dataset Mirror:** Streamed securely via Google TensorFlow Datasets (`emnist/byclass` split featuring 62 balanced alphanumeric classes: 0-9, A-Z, a-z).
* **Architecture:** Multi-layered Conv2D network with max-pooling, spatial flattening, dense feature mapping, dropout regularization, and a Softmax classification tail.

### 2. The Computer Vision Pipeline (`app.py`)
* **Binarization:** Applies Otsu's automatic thresholding to isolate structural letter contours.
* **Component Merging:** Uses custom heuristics to merge vertically detached components (such as matching the dot of an `i` or `j` to its corresponding stem).
* **Row-Clustering:** Automatically groups characters into independent lines based on Y-axis cluster midpoints before running sequential inference from left to right.

---

## 📂 Project Directory Structure

```text
CodeAlpha_Handwritten_Character_Recognition/
│
├── models/                    # Saved binary model architecture (.h5)
│   └── emnist_cnn_model.h5
│
├── train_cnn.py               # Custom CNN training pipeline using TensorFlow Datasets
├── app.py                     # Streamlit Vision Lab UI & OpenCV Pre-processing
├── requirements.txt           # Core system dependencies
└── README.md                  # Project documentation
