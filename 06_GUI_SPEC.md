# GUI & Visualization Spec
## For Handover to Developer

---

## Overview

This document specifies the requirements for building a **Graphical User Interface (GUI)** and **visualization tools** for the fraud detection system. The core ML/NN implementation is complete — you just need to build the user-facing layer.

---

## What Already Exists

### Core Implementation (Done)
- `fraud_detection.py` — Preprocessing, training, evaluation
- `tune_model.py` — Hyperparameter tuning
- `fraud_model.keras` — Trained model
- `preprocessor.pkl` — Fitted encoders/scaler

### Key Parameters
- **Input:** 15 features (amt, category, gender, state, lat, long, city_pop, merch_lat, merch_long, hour, day_of_week, month, day_of_month, distance_km, age)
- **Architecture:** MLP with layers [256, 128, 64]
- **Output:** Probability (0-1) of fraud

---

## Required Features

### 1. Interactive Prediction Interface

A simple GUI where users can:
- Enter transaction details manually
- Get instant fraud probability prediction
- See "Fraud" or "Legitimate" classification

**Input fields needed:**
| Field | Type | Description |
|-------|------|-------------|
| amount | float | Transaction amount ($) |
| category | dropdown | e.g., grocery_pos, gas_station |
| gender | dropdown | M, F |
| state | dropdown | US state abbreviation |
| lat/long | float | Customer location |
| city_pop | int | City population |
| merch_lat/long | float | Merchant location |
| hour | int (0-23) | Transaction hour |
| day_of_week | int (0-6) | Day of week |
| month | int (1-12) | Month |
| distance_km | float | Calculated distance |
| age | int | Customer age |

---

### 2. Model Architecture Visualization

A visual representation of the neural network:

**What to show:**
- Input layer (14-15 neurons)
- Hidden layers (256, 128, 64 neurons)
- Output layer (1 neuron)
- Connections between layers
- Activation functions per layer

**Visual style:**
- Neurons as circles
- Layers as columns
- Connections as lines
- Color coding by layer type

**Interactivity:**
- Zoom in/out
- Hover for neuron details
- Click layer for info

---

### 3. Training History Visualization

Charts showing:
- **Loss curve** — Training vs validation loss over epochs
- **AUC curve** — Training vs validation AUC over epochs
- **Metrics over time** — Precision, recall, F1

---

### 4. Evaluation Results Dashboard

Show test set performance:
- Confusion matrix (TN, FP, FN, TP)
- ROC curve with AUC score
- Precision-Recall curve with AUC
- Precision, Recall, F1 scores
- Classification report

---

### 5. Hyperparameter Tuning Results

Display tune_model.py results:
- Table of all trials
- Best parameters highlighted
- Compare metrics across trials
- Visual comparison of architectures

---

## Suggested Tech Stack

Choose one:

| Option | Pros | Cons |
|--------|------|------|
| **Tkinter** | Built into Python, simple | Looks dated |
| **PyQt/PySide** | Modern look, powerful | More complex |
| **Streamlit** | Quick to build, web-based | Requires browser |
| **Flask + HTML** | Full web control | More setup |
| **CustomTkinter** | Modern tkinter look | Extra dependency |
| **Tkinter + matplotlib** | Charts embedded | Limited interactivity |

---

## File Loading

You need to load:

```python
import joblib
from tensorflow.keras.models import load_model

# Load preprocessor
preprocessor = joblib.load('preprocessor.pkl')
encoders = preprocessor['encoders']
scaler = preprocessor['scaler']
feature_cols = preprocessor['feature_cols']

# Load model
model = load_model('fraud_model.keras')
```

---

## Prediction Function

To make predictions on new data:

```python
import numpy as np

def predict_fraud(model, encoders, scaler, feature_cols, transaction_data):
    """
    transaction_data: dict with keys matching feature_cols
    """
    # Encode categoricals
    for col in ['category', 'gender', 'state']:
        if col in feature_cols:
            transaction_data[col] = encoders[col].transform([transaction_data[col]])[0]
    
    # Create feature array in correct order
    X = np.array([[transaction_data[col] for col in feature_cols]])
    
    # Scale
    X = scaler.transform(X)
    
    # Predict
    prob = model.predict(X, verbose=0)[0][0]
    
    return prob, "Fraud" if prob > 0.5 else "Legitimate"
```

---

## Deliverables

1. **Prediction GUI** — Enter transaction → get fraud prediction
2. **Network Visualizer** — See the neural network architecture
3. **Training Dashboard** — View loss/AUC curves
4. **Evaluation Dashboard** — View confusion matrix, ROC, PR curves
5. **Tuning Dashboard** — View hyperparameter search results

---

## Example User Flow

1. User launches app → sees dashboard with model info
2. Clicks "Predict" → enters transaction details → sees fraud probability
3. Clicks "Visualize" → sees network diagram
4. Clicks "Results" → sees confusion matrix, ROC curve
5. Clicks "Tuning" → sees hyperparameter search results

---

## Notes

- The model uses class weights for imbalance handling (fraud is rare)
- Threshold is 0.5 by default (can be adjusted)
- All data preprocessing is in `fraud_detection.py` — reuse those functions
- The model was trained on ~1.3M transactions with ~0.58% fraud rate

---

## Questions?

If you need clarification on:
- Feature meanings → See `01_THEORY_TUTORIAL.md`
- Model architecture → See `04_CODE_GUIDE_fraud_detection.md`
- Tuning process → See `05_CODE_GUIDE_tune_model.md`