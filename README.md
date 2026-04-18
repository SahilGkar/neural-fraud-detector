# 🛡️ Neural Fraud Detector

> A robust, end-to-end deep learning web application built with TensorFlow and Flask that detects fraudulent credit card transactions in real-time.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow) ![Flask](https://img.shields.io/badge/Flask-3.0-lightgrey?logo=flask)

---

## 🌟 Overview
Credit card fraud is an ever-evolving threat. This project leverages a custom-built **Multi-Layer Perceptron (MLP) Neural Network** trained on a massive dataset of simulated transactions to predict whether a new incoming transaction is **Legitimate** or **Fraudulent**.

It isn't just a backend model—it features a beautifully interactive frontend GUI where you can manually feed transaction scenarios (such as amount, locations, dates, and state) to see real-time fraud probability mapping and architecture diagnostics.

### Key Features
- **Real-Time Prediction Interface**: A responsive Flask-driven web dashboard handling backend API calls instantly.
- **Advanced Preprocessing Pipeline**: Calculates geological Haversine distances, parses chronological time data, scales numerical clusters, and rigorously encodes categorical features on the fly.
- **Automated Hyperparameter Tuning**: Dynamically searches for the optimal hidden layer configurations, dropout rates, and learning parameters to maximize the ROC metric.
- **Out-of-Distribution Handling**: Custom backend logic specifically programmed to safely assess wildly abnormal transaction coordinates or missing criteria.

---

## 🗄️ The Dataset
This model was formulated using a comprehensive simulated credit card transactions dataset. 

- **`fraudTrain.csv`**: Contains roughly **1.3 million rows** of simulated transaction data to train the intricate internal weights of the neural network.
- **`fraudTest.csv`**: Contains roughly **500,000+ rows** of unseen validation data to test the final model's metrics (Accuracy, Precision, Recall).
- **Extracted Features**: Merchant Category, Transaction Amount, User Gender, State, Client Latitude/Longitude, Merchant Latitude/Longitude, City Population, and specific transaction timings (Age at transaction, Hour, Day, Month).
- **Link**: https://www.kaggle.com/datasets/kartik2112/fraud-detection?select=fraudTest.csv

> **⚠️ NOTE:** Because the CSV dataset files exceed 300MB, they are **not** included in this GitHub repository to abide by Git limits. To train the model from scratch on your end, you must download the raw dataset from your provider and place the CSVs in this root directory.

---

## 🏗️ Technical Architecture

### AI/ML Stack
- **Architecture**: Custom Keras/TensorFlow Sequential Model
- **Layers**: `15 Features -> Dense(256, ReLU) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(1, Sigmoid)`
- **Metrics Targeted**: Highly optimized emphasizing **Precision-Recall AUC**, as fraud datasets contain extreme class imbalances (often less than a `0.5%` target baseline).
- **Core Libraries**: `scikit-learn`, `pandas`, `numpy`, `tensorflow`, `joblib`.

### Application Stack
- **Backend Environment**: Python powered by Flask.
- **Frontend Presentation**: Client-side HTML5 & CSS serving real-time dynamic JSON responses.

---

## 🚀 Getting Started

Follow these steps to run the web application cleanly on your own machine.

### 1. Prerequisites
Ensure you have Python installed (preferably version 3.8 to 3.11 for maximum TensorFlow system compatibility).

### 2. Clone the Repository
Clone the code to your local machine:
```bash
git clone https://github.com/SahilGkar/neural-fraud-detector.git
cd neural-fraud-detector
```

### 3. Create the Virtual Environment
Create an isolated Python environment so your global packages aren't affected:

**macOS / Linux**:
```bash
python3 -m venv venv
source venv/bin/activate
```
**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Running the Web Platform
Since the repository securely includes the pre-trained `fraud_model.keras` state and the `preprocessor.pkl` transformers, you can directly launch the dashboard without retraining:
```bash
python app.py
```
*(Visit `http://127.0.0.1:5000` in your web browser to start hunting fraud!)*

---

## 🧠 Training & Tuning From Scratch

### Primary Training Protocol
If you have the dataset CSV files and wish to retrain the neural network:
```bash
python fraud_detection.py
```
This script handles the heavy lifting: processing data grids, training the pipeline with built-in early stopping, evaluating confusion matrix performance against unseen data, and archiving the exported `.keras` and `.pkl` artifacts back to the disk.

### Hyperparameter Tuning
To manually commence random-search trials forcing the script to evaluate alternative learning pathways (searching for better optimal accuracy):
```bash
python tune_model.py
```

---

## 📝 License
This project's logic and pipeline implementation are open-source and free to be utilized for academic purposes or security modeling research.
