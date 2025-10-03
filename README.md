# 🌸 Iris Flower Classifier (CLI)

This project implements a **machine learning pipeline** to classify iris flowers using a **Random Forest Classifier**.  
It covers the complete ML workflow: **data loading**, **training**, **evaluation**, and **prediction**, all wrapped in a simple **command-line interface (CLI)**.

---

## 📂 Project Overview

The pipeline is split into modular components:
- **Data Loader** → loads and preprocesses the Iris dataset.
- **Trainer** → trains a Random Forest Classifier and saves the model.
- **Evaluator** → evaluates the model with accuracy, classification report, and confusion matrix visualization.
- **Predictor** → loads the trained model and makes predictions on new samples.
- **Main CLI** → ties everything together for easy execution.

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/iris-classifier.git
   cd iris-classifier

## Example Output
=== Iris Flower Classifier CLI ===
Training Model...
Evaluating model...
Accuracy: 0.97
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       0.95      0.95      0.95        20
           2       0.95      0.95      0.95        20

Testing single prediction...
Predicted: 1, Actual: 1

## Features
✅ Train a RandomForestClassifier on the Iris dataset
✅ Split dataset into training & test sets
✅ Model evaluation with:
    - Accuracy score
    - Classification report
    - Confusion matrix heatmap (Seaborn + Matplotlib)
✅ Save trained models as .pkl files
✅ Predict single flower samples
