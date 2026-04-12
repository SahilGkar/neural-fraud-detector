# Credit Card Fraud Detection: Deep Learning Approach
## Complete Theory & Implementation Guide with TensorFlow/Keras

---

## Table of Contents

1. [Introduction: Why Deep Learning?](#1-introduction-why-deep-learning)
2. [Neural Network Fundamentals](#2-neural-network-fundamentals)
3. [The MLP Architecture for Fraud Detection](#3-the-mlp-architecture-for-fraud-detection)
4. [Data Preprocessing for Neural Networks](#4-data-preprocessing-for-neural-networks)
5. [Building the Model](#5-building-the-model)
6. [Training the Network](#6-training-the-network)
7. [Handling Class Imbalance](#7-handling-class-imbalance)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Making Predictions](#9-making-predictions)
10. [Hyperparameters Explained](#10-hyperparameters-explained)
11. [Complete Implementation](#11-complete-implementation)
12. [Hyperparameter Tuning & Iteration](#12-hyperparameter-tuning--iteration)
13. [Neural Network Visualization GUI](#13-neural-network-visualization-gui)
14. [Complete Project Workflow](#14-complete-project-workflow)

---

## 1. Introduction: Why Deep Learning?

### 1.1 The Problem with Traditional ML

```
Traditional ML (XGBoost, Random Forest):
┌─────────────────────────────────────────────────┐
│ 1. Feature Engineering → Handcrafted features    │
│ 2. Model learns → Linear rules from features    │
│ 3. Limited interaction between features          │
└─────────────────────────────────────────────────┘

Deep Learning:
┌─────────────────────────────────────────────────┐
│ 1. Raw data → Minimal preprocessing             │
│ 2. Network learns → Hierarchical representations│
│ 3. Automatic feature discovery & interactions    │
└─────────────────────────────────────────────────┘
```

### 1.2 What Neural Networks Learn

```
Input: Transaction Data
    ↓
[Raw Features]
    │
    ├── amt: 150.00
    ├── category: grocery_pos
    ├── distance_km: 250.3
    ├── hour: 3
    ├── age: 32
    └── ... (14 features)
    ↓
[Hidden Layer 1: 256 neurons]
    │ Each neuron learns basic patterns:
    │   - "High amount + late hour"
    │   - "Large distance + small city"
    │   - "Young + high frequency category"
    ↓
[Hidden Layer 2: 128 neurons]
    │ Combines basic patterns:
    │   - "Suspicious transaction detected"
    │   - "Normal spending pattern"
    ↓
[Hidden Layer 3: 64 neurons]
    │ Higher-level abstractions:
    │   - "High fraud risk"
    │   - "Low fraud risk"
    ↓
[Output Layer: 1 neuron]
    │
    └── Probability: 0.87 (87% fraud likelihood)
```

### 1.3 Advantages for Fraud Detection

| Aspect | Tree-Based Models | Neural Networks |
|--------|-------------------|-----------------|
| Feature engineering | Manual required | Learns automatically |
| Feature interactions | Limited | Captures complex |
| Scalability | Good | Excellent |
| Real-time inference | Fast | Very fast |
| Probability calibration | Good | Requires care |
| **Best for** | Quick baseline | Production systems |

---

## 2. Neural Network Fundamentals

### 2.1 The Perceptron: Single Neuron

```
                    ┌─────────────┐
                    │             │
x₁ ───────────────► │             │
                    │   w₁        │  ┌────────────┐
x₂ ───────────────► │   w₂        │──│            │
                    │   w₃   Σ ──►│  │ Activation │──► Output
x₃ ───────────────► │      b      │  │  Function   │
                    │             │  └────────────┘
... ───────────────► │             │
                    │             │
xₙ ───────────────► │             │
                    └─────────────┘

Formula: output = activation(Σ(xᵢ × wᵢ) + b)
```

**Code Implementation**:
```python
import numpy as np

def perceptron(inputs, weights, bias):
    """
    Single perceptron calculation
    
    inputs:  [x1, x2, x3, ..., xn] - input values
    weights: [w1, w2, w3, ..., wn] - learned weights
    bias:    b - learned bias
    """
    # Step 1: Weighted sum
    # Multiply each input by its weight and sum
    weighted_sum = np.dot(inputs, weights) + bias
    
    # Step 2: Activation (explained next)
    output = activation(weighted_sum)
    
    return output

# Example
inputs  = np.array([150.0, 1.0, 250.3, 3.0, 32.0])  # amt, category, distance, hour, age
weights  = np.array([0.5, 0.3, 0.8, 0.2, 0.1])       # learned from training
bias = -2.5

output = perceptron(inputs, weights, bias)
print(f"Output before activation: {output}")
# Output before activation: 148.58
```

### 2.2 Activation Functions

**Question**: Why do we need activation functions?

**Answer**: Without them, a network is just a linear equation — too simple!

```
WITHOUT activation:
    y = w₁x₁ + w₂x₂ + b
    (Just a straight line - can't learn complex patterns)

WITH activation:
    y = activation(w₁x₁ + w₂x₂ + b)
    (Can learn curves, bends, complex shapes)
```

**Common Activation Functions**:

```python
import numpy as np

# 1. Sigmoid - For binary classification output
def sigmoid(x):
    """Squashes output to (0, 1) - good for probabilities"""
    return 1 / (1 + np.exp(-x))

# Example
print(f"sigmoid(0) = {sigmoid(0)}")      # 0.5
print(f"sigmoid(2) = {sigmoid(2)}")      # 0.88
print(f"sigmoid(-3) = {sigmoid(-3)}")   # 0.05

# 2. ReLU (Rectified Linear Unit) - Hidden layers
def relu(x):
    """Returns x if positive, else 0. Fast and effective."""
    return np.maximum(0, x)

# Example
print(f"relu(5) = {relu(5)}")    # 5
print(f"relu(-3) = {relu(-3)}")  # 0

# 3. Leaky ReLU - Fixes "dying ReLU" problem
def leaky_relu(x, alpha=0.01):
    """Returns x if positive, else alpha*x (small negative slope)"""
    return np.where(x > 0, x, alpha * x)

# 4. Tanh - Squashes to (-1, 1)
def tanh(x):
    """Symmetric around 0, good for hidden layers"""
    return np.tanh(x)

print(f"tanh(0) = {tanh(0)}")     # 0
print(f"tanh(2) = {tanh(2)}")     # 0.96
```

**Visual Comparison**:
```
Sigmoid:          ReLU:            Tanh:
  1│    ╭──╮         1│   ╱          1│   ╭─
   │   ╱    ╲        │  ╱           0│──╯
  0│──╯      ╲──    0│╱            -1│  
   │             ╲     │╲              
  0│              ╲  │  ╲           
   └───────────────   └───          ──┴────
      0                  0            0
```

### 2.3 Neural Network Layers

```python
# How layers work in TensorFlow/Keras

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential

model = Sequential([
    # Input layer - defines input shape
    Input(shape=(14,)),  # 14 features
    
    # Hidden layer 1 - 256 neurons
    Dense(256, activation='relu'),  # Fully connected
    
    # Hidden layer 2 - 128 neurons
    Dense(128, activation='relu'),
    
    # Hidden layer 3 - 64 neurons
    Dense(64, activation='relu'),
    
    # Output layer - 1 neuron for binary classification
    Dense(1, activation='sigmoid')  # Output: probability
])
```

**What "Dense" Means**:
```
Dense Layer = Fully Connected Layer

Input (14)          Hidden (256)
─────────          ────────────
neuron 1 ─────────► neuron 1  ──────┐
neuron 2 ─────────► neuron 2  ──────┼──► Every input connects
neuron 3 ─────────► neuron 3  ──────┤    to every neuron!
  ...    ─────────►   ...    ──────┤
neuron14 ─────────► neuron256 ─────┘
                          │
                          └─── Each connection has a weight
```

### 2.4 Forward Propagation: How Data Flows

```python
import numpy as np

def forward_propagation(X, weights, biases, activations):
    """
    X: Input data (batch_size, num_features)
    weights: List of weight matrices for each layer
    biases: List of bias vectors for each layer
    activations: List of activation functions
    """
    A = X  # Start with input
    
    # For each layer
    for W, b, activation in zip(weights, biases, activations):
        # Linear transformation: Z = W·A + b
        Z = np.dot(A, W) + b
        
        # Non-linear activation: A = activation(Z)
        A = activation(Z)
    
    return A  # Final output (probabilities)

# Example with 1 sample
X = np.array([[150.0, 1.0, 250.3, 3.0, 32.0]])  # 1 sample, 5 features

# Simplified weights (random)
W1 = np.random.randn(5, 3)   # 5 inputs → 3 hidden
b1 = np.zeros(3)
W2 = np.random.randn(3, 1)   # 3 hidden → 1 output
b2 = np.zeros(1)

# Forward pass
Z1 = np.dot(X, W1) + b1      # Layer 1
A1 = relu(Z1)                # Activation
Z2 = np.dot(A1, W2) + b2     # Layer 2
output = sigmoid(Z2)          # Final output (probability)

print(f"Fraud probability: {output[0,0]:.4f}")
```

---

## 3. The MLP Architecture for Fraud Detection

### 3.1 Recommended Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRAUD DETECTION MLP                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT LAYER                                                    │
│  ┌─────────────────────────────────┐                           │
│  │ 14 neurons (one per feature)   │                           │
│  │ • amt, category, gender, state │                           │
│  │ • lat, long, city_pop           │                           │
│  │ • merch_lat, merch_long         │                           │
│  │ • hour, day_of_week, month       │                           │
│  │ • distance_km, age               │                           │
│  └─────────────────────────────────┘                           │
│                          │                                     │
│                          ▼                                     │
│  HIDDEN LAYER 1                                              │
│  ┌─────────────────────────────────┐                           │
│  │ 256 neurons                     │                           │
│  │ Activation: ReLU               │                           │
│  │ Batch Normalization             │                           │
│  │ Dropout: 0.3 (prevents overfitting) │                      │
│  └─────────────────────────────────┘                           │
│                          │                                     │
│                          ▼                                     │
│  HIDDEN LAYER 2                                              │
│  ┌─────────────────────────────────┐                           │
│  │ 128 neurons                     │                           │
│  │ Activation: ReLU               │                           │
│  │ Batch Normalization             │                           │
│  │ Dropout: 0.3                   │                           │
│  └─────────────────────────────────┘                           │
│                          │                                     │
│                          ▼                                     │
│  HIDDEN LAYER 3                                              │
│  ┌─────────────────────────────────┐                           │
│  │ 64 neurons                      │                           │
│  │ Activation: ReLU               │                           │
│  │ Batch Normalization             │                           │
│  │ Dropout: 0.3                   │                           │
│  └─────────────────────────────────┘                           │
│                          │                                     │
│                          ▼                                     │
│  OUTPUT LAYER                                               │
│  ┌─────────────────────────────────┐                           │
│  │ 1 neuron                        │                           │
│  │ Activation: Sigmoid            │                           │
│  │ Output: Probability (0 to 1)   │                           │
│  └─────────────────────────────────┘                           │
│                          │                                     │
│                          ▼                                     │
│                    PROBABILITY                                 │
│                         0.87                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Architecture Decisions Explained

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, BatchNormalization, Dropout
)

def build_fraud_detection_model(input_dim=14):
    """
    Build MLP for fraud detection
    
    Architecture:
    - Input → Hidden1(256) → Hidden2(128) → Hidden3(64) → Output(1)
    - Each hidden layer has: Dense + BatchNorm + Dropout + ReLU
    """
    model = Sequential([
        # Input layer
        Input(shape=(input_dim,)),
        
        # Hidden Layer 1: 256 neurons
        Dense(256),
        BatchNormalization(),  # Stabilizes training
        Dropout(0.3),          # 30% of neurons "turned off" during training
        Activation('relu'),
        
        # Hidden Layer 2: 128 neurons
        Dense(128),
        BatchNormalization(),
        Dropout(0.3),
        Activation('relu'),
        
        # Hidden Layer 3: 64 neurons
        Dense(64),
        BatchNormalization(),
        Dropout(0.3),
        Activation('relu'),
        
        # Output Layer: Binary classification
        Dense(1, activation='sigmoid')  # Output: fraud probability
    ])
    
    return model

# Build the model
model = build_fraud_detection_model(input_dim=14)
model.summary()
```

---

## 4. Data Preprocessing for Neural Networks

### 4.1 Why Preprocess Data?

```
Neural networks are SENSITIVE to:
┌─────────────────────────────────────────────────┐
│ 1. SCALE    │ Different ranges break learning   │
│              │ amt: 0-5000 vs hour: 0-23       │
├──────────────┼──────────────────────────────────┤
│ 2. DISTRIB.  │ Skewed data hurts performance    │
│              │ Most amounts: $10-100, some: $5000│
├──────────────┼──────────────────────────────────┤
│ 3. CATEGORIES│ Networks need numbers            │
│              │ "grocery_pos" → 0               │
└─────────────────────────────────────────────────┘
```

### 4.2 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

def scale_features(X_train, X_test):
    """
    StandardScaler: Transform to mean=0, std=1
    
    Formula: z = (x - μ) / σ
    
    This centers the data around 0 with unit variance.
    """
    scaler = StandardScaler()
    
    # Fit ONLY on training data (prevent data leakage)
    scaler.fit(X_train)
    
    # Transform both train and test
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

# Example
X_train = np.array([
    [150.0, 250.3],    # amt, distance
    [25.0, 5.0],
    [5000.0, 1000.0]
])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

print("Before scaling:")
print(f"  Mean: {X_train.mean(axis=0)}")
print(f"  Std:  {X_train.std(axis=0)}")

print("After scaling:")
print(f"  Mean: {X_scaled.mean(axis=0)}")  # ~[0, 0]
print(f"  Std:  {X_scaled.std(axis=0)}")  # ~[1, 1]
```

### 4.3 Complete Preprocessing Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def preprocess_for_nn(df, encoders=None, scaler=None, fit=True):
    """
    Preprocess fraud data for neural network
    """
    df = df.copy()
    
    # Drop columns with too many unique values
    drop_cols = ['merchant', 'job', 'first', 'last', 'street', 'city',
                 'trans_num', 'cc_num', 'zip', 'unix_time', '']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Feature engineering
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    df['distance_km'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days / 365).astype(int)
    
    df = df.drop(columns=['trans_date_trans_time', 'dob'])
    
    # Encode categoricals
    categorical_cols = ['category', 'gender', 'state']
    
    if fit:
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col].astype(str))
    
    # Separate features and target
    X = df.drop(columns=['is_fraud']).values.astype(np.float32)
    y = df['is_fraud'].values
    
    # Scale numerical features
    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)
    
    return X, y, encoders, scaler
```

---

## 5. Building the Model

### 5.1 Complete Model Definition

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, BatchNormalization, Dropout, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

def build_model(input_dim=14, learning_rate=0.001):
    """
    Build MLP for fraud detection
    """
    model = Sequential([
        # Input layer
        Input(shape=(input_dim,)),
        
        # Hidden Layer 1: 256 neurons
        Dense(256, kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Activation('relu'),
        
        # Hidden Layer 2: 128 neurons
        Dense(128, kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Activation('relu'),
        
        # Hidden Layer 3: 64 neurons
        Dense(64, kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Activation('relu'),
        
        # Output Layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=BinaryCrossentropy(from_logits=False),
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model

# Build and display
model = build_model(input_dim=14, learning_rate=0.001)
model.summary()
```

### 5.2 Understanding Each Component

```python
# Kernel Initializer: How initial weights are set
# 'he_normal' - Good for ReLU activations
# 'glorot_uniform' - Good for tanh/sigmoid

# BatchNormalization: Normalizes activations per batch
# - Stabilizes training
# - Allows higher learning rates
# - Acts as regularization

# Dropout: Randomly "turns off" neurons during training
# - Rate 0.3 = 30% of neurons dropped each batch
# - Prevents overfitting
# - Forces network to learn redundant representations
```

### 5.3 Model Summary Output

```
Model: "fraud_detection_mlp"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 256)               3,840     
 batch_normalization (Batch  (None, 256)               1,024     
 normalization)                                              
 dropout (Dropout)           (None, 256)               0         
 activation (Activation)     (None, 256)               0         
 dense_1 (Dense)             (None, 128)               32,896    
 batch_normalization_1       (None, 128)               512       
 dropout_1 (Dropout)         (None, 128)               0         
 activation_1                (None, 128)               0         
 dense_2 (Dense)             (None, 64)                8,256     
 batch_normalization_2       (None, 64)                256       
 dropout_2 (Dropout)         (None, 64)                0         
 activation_2                (None, 64)                0         
 dense_3 (Dense)             (None, 1)                 65        
 activation_3 (Activation)   (None, 1)                 0         
=================================================================
Total params: 47,745 (186.50 KB)
Trainable params: 46,849
Non-trainable params: 896
_________________________________________________________________
```

---

## 6. Training the Network

### 6.1 The Training Loop

```python
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

def train_model(model, X_train, y_train, X_val, y_val, class_weights):
    """
    Train the fraud detection model
    """
    # Callbacks: Actions during training
    callbacks = [
        # Stop if validation loss doesn't improve for 5 epochs
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        # Reduce learning rate if plateau
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,        # Reduce by half
            patience=3,
            min_lr=0.00001
        )
    ]
    
    # Training
    history = model.fit(
        X_train, y_train,
        
        # Batch training
        batch_size=2048,        # 2048 samples per update
        epochs=100,             # Max epochs
        validation_data=(X_val, y_val),
        class_weight=class_weights,  # Handle imbalance
        callbacks=callbacks,
        
        # Progress output
        verbose=1
    )
    
    return history

# Train
history = train_model(model, X_train, y_train, X_val, y_val, class_weights)
```

### 6.2 Understanding Training Parameters

```python
# batch_size: How many samples before weight update
"""
Batch Size Effects:

batch_size = 32 (Small)
├── More updates per epoch
├── Noisy gradients (can escape local minima)
├── Lower memory usage
└── May not fully utilize GPU

batch_size = 2048 (Large)
├── Fewer updates per epoch
├── Smoother gradients
├── Better GPU utilization
└── May need more epochs
"""

# epochs: How many times to see entire dataset
"""
Watch for overfitting:
- val_loss decreasing → Good, keep training
- val_loss increasing → Overfitting, stop!
"""

# EarlyStopping stops when val_loss stops improving
```

### 6.3 The Backpropagation Algorithm

```python
"""
Training Process: Forward + Backward Pass

FORWARD PASS:
┌─────────────────────────────────────────────────────┐
│ Input → Layer1 → Layer2 → Layer3 → Output           │
│         Calculate: output, loss                     │
└─────────────────────────────────────────────────────┘
                      ↓
LOSS CALCULATION:
┌─────────────────────────────────────────────────────┐
│ Compare prediction to actual                        │
│                                                     │
│ Binary Cross-Entropy Loss:                          │
│ L = -[y·log(p) + (1-y)·log(1-p)]                  │
│                                                     │
│ Example:                                            │
│ Actual: 1 (fraud)                                  │
│ Predicted: 0.87                                    │
│ Loss = -[1·log(0.87) + 0·log(0.13)] = 0.139       │
└─────────────────────────────────────────────────────┘
                      ↓
BACKWARD PASS (Backpropagation):
┌─────────────────────────────────────────────────────┐
│ Calculate how much each weight contributed to error│
│                                                     │
│ Layer 3 weights → Large error contribution?         │
│ Layer 2 weights → Medium error?                     │
│ Layer 1 weights → Small error?                     │
└─────────────────────────────────────────────────────┘
                      ↓
WEIGHT UPDATE:
┌─────────────────────────────────────────────────────┐
│ Use gradient to adjust weights                     │
│                                                     │
│ new_weight = old_weight - learning_rate × gradient│
│                                                     │
│ Learning rate = 0.001 (small step)                 │
│ Too large → overshoot optimal                      │
│ Too small → very slow learning                     │
└─────────────────────────────────────────────────────┘
```

---

## 7. Handling Class Imbalance

### 7.1 The Challenge

```
Your Dataset:
┌─────────────────────────────────┐
│ 1,289,169 Legitimate (99.42%)  │ ████████████████████████████
│       7,506 Fraudulent (0.58%) │ ▏
└─────────────────────────────────┘

Problem: Network learns to predict "legit" always → 99.4% accuracy!
Solution: Class weighting
```

### 7.2 Method: Class Weights

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def calculate_class_weights(y):
    """
    Calculate weights for each class to balance the dataset
    """
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    
    # Convert to dictionary
    class_weights = {0: weights[0], 1: weights[1]}
    
    return class_weights

# Calculate
class_weights = calculate_class_weights(y_train)
print(f"Class weights: {class_weights}")
# Output: {0: 0.5029, 1: 86.1454}

# This means:
# - Legitimate (0): Weight = 0.5 (less important)
# - Fraud (1): Weight = 86.1 (much more important!)

# Use in training
history = model.fit(
    X_train, y_train,
    batch_size=2048,
    epochs=100,
    validation_data=(X_val, y_val),
    class_weight=class_weights,  # Add this!
    callbacks=callbacks,
    verbose=1
)
```

---

## 8. Evaluation Metrics

### 8.1 Confusion Matrix Explained

```
                        PREDICTED
                     LEGIT    FRAUD
        LEGIT    ┌─────────┬─────────┐
ACTUAL           │   TN    │   FP    │
                 │  True   │  False  │
                 │   Neg    │  Pos    │
                 ├─────────┼─────────┤
                 │   FN    │   TP    │
                 │  False  │  True   │
                 │   Neg    │  Pos    │
                 └─────────┴─────────┘

TN (True Negative):  Predicted LEGIT, was LEGIT ✓
FP (False Positive): Predicted FRAUD, was LEGIT ✗ (false alarm)
FN (False Negative): Predicted LEGIT, was FRAUD ✗ (missed fraud)
TP (True Positive):  Predicted FRAUD, was FRAUD ✓ (caught fraud)
```

### 8.2 Key Metrics

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Comprehensive model evaluation
    """
    # Get predictions
    y_proba = model.predict(X_test).flatten()
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Metrics
    print(f"\nPrecision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    
    return {
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_proba)
    }
```

### 8.3 Threshold Optimization

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_proba, target_recall=0.8):
    """
    Find threshold that achieves target recall
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold closest to target recall
    distances = np.abs(recalls - target_recall)
    best_idx = np.argmin(distances)
    
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    achieved_recall = recalls[best_idx]
    achieved_precision = precisions[best_idx]
    
    return optimal_threshold, achieved_recall, achieved_precision

# Find optimal threshold for 80% recall
optimal_thresh, achieved_recall, achieved_precision = find_optimal_threshold(
    y_val, y_proba, target_recall=0.8
)

print(f"\nOptimal Threshold for 80% Recall:")
print(f"  Threshold: {optimal_thresh:.3f}")
print(f"  Achieved Recall: {achieved_recall:.4f}")
print(f"  Achieved Precision: {achieved_precision:.4f}")
```

---

## 9. Making Predictions

### 9.1 Complete Prediction Pipeline

```python
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def predict_single_transaction(model, transaction_data, encoders, scaler, threshold=0.5):
    """
    Complete prediction pipeline for a single transaction
    """
    df = pd.DataFrame([transaction_data])
    
    # Feature engineering
    trans_time = pd.to_datetime(transaction_data['trans_date_trans_time'])
    df['hour'] = trans_time.hour
    df['day_of_week'] = trans_time.dayofweek
    df['month'] = trans_time.month
    
    df['distance_km'] = haversine_distance(
        transaction_data['lat'], transaction_data['long'],
        transaction_data['merch_lat'], transaction_data['merch_long']
    )
    
    dob = pd.to_datetime(transaction_data['dob'])
    df['age'] = ((trans_time - dob).days / 365)
    
    # Encode categoricals
    df['category'] = encoders['category'].transform([transaction_data['category']])[0]
    df['gender'] = encoders['gender'].transform([transaction_data['gender']])[0]
    df['state'] = encoders['state'].transform([transaction_data['state']])[0]
    
    # Select features
    features = ['amt', 'category', 'gender', 'state', 'lat', 'long',
                'city_pop', 'merch_lat', 'merch_long', 'hour',
                'day_of_week', 'month', 'distance_km', 'age']
    
    X = df[features].values.astype(np.float32)
    X = scaler.transform(X)
    
    # Predict
    proba = model.predict(X, verbose=0)[0, 0]
    is_fraud = proba >= threshold
    
    risk = 'HIGH' if proba >= 0.7 else 'MEDIUM' if proba >= 0.4 else 'LOW'
    
    return {
        'probability': round(proba * 100, 2),
        'is_fraud': bool(is_fraud),
        'risk_level': risk,
        'threshold': threshold
    }

# Example usage
if __name__ == "__main__":
    # Load saved model and preprocessors
    model = tf.keras.models.load_model('models/fraud_model.keras')
    encoders = joblib.load('models/encoders.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    with open('models/optimal_threshold.txt', 'r') as f:
        threshold = float(f.read())
    
    # Example transaction
    transaction = {
        'trans_date_trans_time': '2024-01-15 14:30:00',
        'category': 'grocery_pos',
        'amt': 150.00,
        'gender': 'M',
        'state': 'CA',
        'lat': 34.0522,
        'long': -118.2437,
        'city_pop': 5000000,
        'merch_lat': 34.0522,
        'merch_long': -118.2437,
        'dob': '1985-06-15'
    }
    
    result = predict_single_transaction(model, transaction, encoders, scaler, threshold)
    
    print("\n" + "="*50)
    print("FRAUD PREDICTION RESULT")
    print("="*50)
    print(f"Fraud Probability: {result['probability']}%")
    print(f"Prediction: {'⚠️ FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
    print(f"Risk Level: {result['risk_level']}")
    print("="*50)
```

---

## 10. Hyperparameters Explained

### 10.1 Architecture Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Layer 1 Size | 256 | 64-512 | Model capacity |
| Layer 2 Size | 128 | 32-256 | Model capacity |
| Layer 3 Size | 64 | 16-128 | Model capacity |

### 10.2 Training Hyperparameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Learning Rate | 0.001 | 0.0001-0.01 | Speed vs stability |
| Batch Size | 2048 | 32-4096 | Memory vs smoothness |
| Dropout | 0.3 | 0.0-0.5 | Overfitting control |
| Early Stop | patience=5 | 3-10 | Prevents overfitting |

---

## 11. Complete Implementation

### 11.1 Project Structure

```
fraud-detection/
├── fraudTrain.csv           # Training data
├── fraudTest.csv            # Test data
├── requirements.txt         # Dependencies
├── train_model.py           # Training script
├── predict.py               # Prediction script
├── models/                  # Saved models
│   ├── fraud_model.keras    # Trained model
│   ├── encoders.pkl         # Label encoders
│   ├── scaler.pkl           # StandardScaler
│   └── optimal_threshold.txt
├── evaluation/              # Results
├── templates/
│   └── index.html           # Web UI
└── app.py                   # Flask web app
```

### 11.2 Requirements (requirements.txt)

```
tensorflow>=2.15.0
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
flask
tkinter          # Usually included with Python, for GUI visualization
```

### 11.3 Training Script (train_model.py)

```python
"""
Credit Card Fraud Detection - Neural Network Training
Deep Learning with TensorFlow/Keras
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, BatchNormalization, Dropout, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def preprocess_data(df, encoders=None, scaler=None, fit=True):
    """Preprocess fraud dataset for neural network"""
    df = df.copy()
    
    # Drop high-cardinality columns
    drop_cols = ['merchant', 'job', 'first', 'last', 'street', 'city',
                 'trans_num', 'cc_num', 'zip', 'unix_time', '']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    # Feature engineering
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    df['distance_km'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days / 365).astype(int)
    
    df = df.drop(columns=['trans_date_trans_time', 'dob'])
    
    # Encode categoricals
    categorical_cols = ['category', 'gender', 'state']
    
    if fit:
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col].astype(str))
    
    # Separate features and target
    X = df.drop(columns=['is_fraud']).values.astype(np.float32)
    y = df['is_fraud'].values
    
    # Scale numerical features
    if fit:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)
    
    return X, y, encoders, scaler

# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(input_dim):
    """Build MLP for fraud detection"""
    model = Sequential([
        Input(shape=(input_dim,)),
        
        Dense(256, kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Activation('relu'),
        
        Dense(128, kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Activation('relu'),
        
        Dense(64, kernel_initializer='he_normal'),
        BatchNormalization(),
        Dropout(0.3),
        Activation('relu'),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model

# ============================================================================
# THRESHOLD OPTIMIZATION
# ============================================================================

def find_optimal_threshold(y_true, y_proba, target_recall=0.8):
    """Find threshold for target recall"""
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    distances = np.abs(recalls - target_recall)
    best_idx = np.argmin(distances)
    
    optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    return optimal_threshold

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - NEURAL NETWORK")
    print("="*60)
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    # Load data
    print("\n[1/7] Loading data...")
    train_df = pd.read_csv('fraudTrain.csv')
    test_df = pd.read_csv('fraudTest.csv')
    print(f"  Training samples: {len(train_df):,}")
    print(f"  Test samples: {len(test_df):,}")
    
    # Preprocess data
    print("\n[2/7] Preprocessing data...")
    X_train_full, y_train_full, encoders, scaler = preprocess_data(train_df, fit=True)
    X_test, y_test, _, _ = preprocess_data(test_df, encoders, scaler, fit=False)
    
    print(f"  Features: {X_train_full.shape[1]}")
    print(f"  Fraud rate (train): {y_train_full.mean()*100:.2f}%")
    
    # Split for validation
    print("\n[3/7] Creating train/validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.2,
        stratify=y_train_full,
        random_state=42
    )
    print(f"  Training: {len(X_train):,}")
    print(f"  Validation: {len(X_val):,}")
    
    # Calculate class weights
    print("\n[4/7] Calculating class weights...")
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    print(f"  Class 0 (legit) weight: {class_weights[0]:.4f}")
    print(f"  Class 1 (fraud) weight: {class_weights[1]:.4f}")
    
    # Build model
    print("\n[5/7] Building model...")
    model = build_model(input_dim=X_train.shape[1])
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    ]
    
    # Train
    print("\n[6/7] Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=2048,
        epochs=100,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n[7/7] Evaluating on test set...")
    
    # Find optimal threshold
    y_val_proba = model.predict(X_val, verbose=0).flatten()
    optimal_threshold = find_optimal_threshold(y_val, y_val_proba, 0.8)
    print(f"  Optimal threshold (80% recall): {optimal_threshold:.4f}")
    
    # Predict on test
    y_test_proba = model.predict(X_test, verbose=0).flatten()
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    
    # Metrics
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              LEGIT      FRAUD")
    print(f"Actual LEGIT  {cm[0,0]:>8,}   {cm[0,1]:>8,}")
    print(f"       FRAUD  {cm[1,0]:>8,}   {cm[1,1]:>8,}")
    
    print(f"\nMetrics at threshold={optimal_threshold:.4f}:")
    print(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_test_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_test_proba):.4f}")
    
    # Save model
    print("\nSaving model and preprocessors...")
    model.save('models/fraud_model.keras')
    joblib.dump(encoders, 'models/encoders.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    with open('models/optimal_threshold.txt', 'w') as f:
        f.write(str(optimal_threshold))
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
```

### 11.4 Prediction Script (predict.py)

```python
"""
Credit Card Fraud Detection - Neural Network Prediction
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load model and preprocessors
model = tf.keras.models.load_model('models/fraud_model.keras')
encoders = joblib.load('models/encoders.pkl')
scaler = joblib.load('models/scaler.pkl')

with open('models/optimal_threshold.txt', 'r') as f:
    THRESHOLD = float(f.read())

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def predict_single_transaction(transaction_data):
    """Predict fraud for a transaction"""
    df = pd.DataFrame([transaction_data])
    
    trans_time = pd.to_datetime(transaction_data['trans_date_trans_time'])
    df['hour'] = trans_time.hour
    df['day_of_week'] = trans_time.dayofweek
    df['month'] = trans_time.month
    df['distance_km'] = haversine_distance(
        transaction_data['lat'], transaction_data['long'],
        transaction_data['merch_lat'], transaction_data['merch_long']
    )
    dob = pd.to_datetime(transaction_data['dob'])
    df['age'] = ((trans_time - dob).days / 365)
    
    df['category'] = encoders['category'].transform([transaction_data['category']])[0]
    df['gender'] = encoders['gender'].transform([transaction_data['gender']])[0]
    df['state'] = encoders['state'].transform([transaction_data['state']])[0]
    
    features = ['amt', 'category', 'gender', 'state', 'lat', 'long',
                'city_pop', 'merch_lat', 'merch_long', 'hour',
                'day_of_week', 'month', 'distance_km', 'age']
    
    X = df[features].values.astype(np.float32)
    X = scaler.transform(X)
    
    proba = model.predict(X, verbose=0)[0, 0]
    is_fraud = proba >= THRESHOLD
    
    risk = 'HIGH' if proba >= 0.7 else 'MEDIUM' if proba >= 0.4 else 'LOW'
    
    return {
        'probability': round(proba * 100, 2),
        'is_fraud': bool(is_fraud),
        'risk_level': risk,
        'threshold': THRESHOLD
    }

if __name__ == "__main__":
    transaction = {
        'trans_date_trans_time': '2024-01-15 14:30:00',
        'category': 'grocery_pos',
        'amt': 150.00,
        'gender': 'M',
        'state': 'CA',
        'lat': 34.0522,
        'long': -118.2437,
        'city_pop': 5000000,
        'merch_lat': 34.0522,
        'merch_long': -118.2437,
        'dob': '1985-06-15'
    }
    
    result = predict_single_transaction(transaction)
    
    print("\n" + "="*50)
    print("FRAUD PREDICTION RESULT")
    print("="*50)
    print(f"Fraud Probability: {result['probability']}%")
    print(f"Prediction: {'⚠️ FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
    print(f"Risk Level: {result['risk_level']}")
    print("="*50)
```

---

## Part 12: Hyperparameter Tuning & Iteration

### 12.1 Why Tune Hyperparameters?

```
Hyperparameters = Settings we choose before training

┌─────────────────────────────────────────────────────────────┐
│  HYPERPARAMETERS (We Choose)     │  PARAMETERS (Learned)  │
├──────────────────────────────────┼─────────────────────────┤
│  • Number of layers              │  • Weights              │
│  • Number of neurons per layer   │  • Biases               │
│  • Learning rate                 │                         │
│  • Dropout rate                  │                         │
│  • Batch size                    │                         │
│  • Activation functions          │                         │
└──────────────────────────────────┴─────────────────────────┘

Why it matters:
- Right settings → Fast training, good accuracy
- Wrong settings → Slow, poor accuracy, or no learning
```

### 12.2 What to Tune

```python
# Key hyperparameters to experiment with:
hyperparameters = {
    # Architecture
    'layer_sizes': [[256, 128, 64], [512, 256, 128], [128, 64, 32]],
    'num_layers': [2, 3, 4],
    
    # Training
    'learning_rate': [0.0001, 0.001, 0.01],
    'batch_size': [512, 1024, 2048],
    
    # Regularization
    'dropout_rate': [0.2, 0.3, 0.5],
    'l2_reg': [0.0, 0.001, 0.01],
}
```

### 12.3 Tuning Strategy

```python
"""
TUNING APPROACHES:

1. MANUAL TUNING (You try settings)
   - Good for understanding
   - Time consuming
   - Best when you have few parameters

2. GRID SEARCH (Try ALL combinations)
   - Exhaustive
   - Expensive (lots of combinations)
   - Good for < 5 parameters

3. RANDOM SEARCH (Try RANDOM combinations)
   - Faster than grid
   - Often finds good solutions
   - Good for many parameters

4. BAYESIAN OPTIMIZATION (Smart search)
   - Learns from previous attempts
   - Most efficient
   - Complex to implement

RECOMMENDATION: Start with Random Search
"""

# Grid Search Example (for 3 params × 3 values each = 27 combinations)
from sklearn.model_selection import ParameterGrid

param_grid = {
    'learning_rate': [0.001, 0.01],
    'batch_size': [1024, 2048],
    'dropout_rate': [0.2, 0.3],
}

grid = list(ParameterGrid(param_grid))
print(f"Total combinations: {len(grid)}")
# Output: Total combinations: 8
```

### 12.4 Complete Tuning Script (tune_model.py)

```python
"""
Credit Card Fraud Detection - Hyperparameter Tuning
Systematically tests different configurations
"""

import pandas as pd
import numpy as np
import os
import joblib
import time
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, BatchNormalization, Dropout, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ============================================================================
# PREPROCESSING (Same as before)
# ============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def preprocess_data(df, encoders=None, scaler=None, fit=True):
    df = df.copy()
    
    drop_cols = ['merchant', 'job', 'first', 'last', 'street', 'city',
                 'trans_num', 'cc_num', 'zip', 'unix_time', '']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    df['distance_km'] = haversine_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = ((df['trans_date_trans_time'] - df['dob']).dt.days / 365).astype(int)
    df = df.drop(columns=['trans_date_trans_time', 'dob'])
    
    categorical_cols = ['category', 'gender', 'state']
    
    if fit:
        from sklearn.preprocessing import LabelEncoder
        encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        for col in categorical_cols:
            df[col] = encoders[col].transform(df[col].astype(str))
    
    X = df.drop(columns=['is_fraud']).values.astype(np.float32)
    y = df['is_fraud'].values
    
    if fit:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = scaler.fit_transform(X).astype(np.float32)
    else:
        X = scaler.transform(X).astype(np.float32)
    
    return X, y, encoders, scaler

# ============================================================================
# MODEL BUILDER (PARAMETRIC)
# ============================================================================

def build_model(input_dim, config):
    """
    Build model with given configuration
    
    config = {
        'layers': [256, 128, 64],  # Neurons per layer
        'dropout': 0.3,              # Dropout rate
        'learning_rate': 0.001       # Learning rate
    }
    """
    model = Sequential([Input(shape=(input_dim,))])
    
    # Add hidden layers
    for neurons in config['layers']:
        model.add(Dense(neurons, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(config['dropout']))
        model.add(Activation('relu'))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    return model

# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_and_evaluate(X_train, y_train, X_val, y_val, config, class_weights):
    """Train model with config and return metrics"""
    
    model = build_model(X_train.shape[1], config)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=0.00001, verbose=0)
    ]
    
    # Train (reduced epochs for tuning)
    history = model.fit(
        X_train, y_train,
        batch_size=config['batch_size'],
        epochs=30,  # Reduced for faster tuning
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate
    y_proba = model.predict(X_val, verbose=0).flatten()
    y_pred = (y_proba >= 0.5).astype(int)
    
    metrics = {
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'auc': roc_auc_score(y_val, y_proba),
        'epochs_trained': len(history.history['loss'])
    }
    
    return model, metrics

def find_optimal_threshold(y_true, y_proba, target_recall=0.8):
    """Find threshold for target recall"""
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    distances = np.abs(recalls - target_recall)
    best_idx = np.argmin(distances)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

# ============================================================================
# HYPERPARAMETER SEARCH
# ============================================================================

def random_search_tuning(X_train, y_train, class_weights, n_trials=10):
    """
    Random search over hyperparameter space
    """
    # Define search space
    search_space = {
        'layers': [
            [512, 256, 128],
            [256, 128, 64],
            [128, 64, 32],
            [256, 256, 128, 64],
            [128, 128, 64, 64],
        ],
        'dropout': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.0001, 0.0005, 0.001, 0.005],
        'batch_size': [512, 1024, 2048, 4096],
    }
    
    results = []
    best_score = 0
    best_config = None
    best_model = None
    
    print("="*70)
    print("HYPERPARAMETER TUNING - RANDOM SEARCH")
    print("="*70)
    print(f"Running {n_trials} trials...")
    print("="*70)
    
    for trial in range(n_trials):
        # Randomly sample configuration
        config = {
            'layers': search_space['layers'][np.random.randint(len(search_space['layers']))],
            'dropout': np.random.choice(search_space['dropout']),
            'learning_rate': np.random.choice(search_space['learning_rate']),
            'batch_size': np.random.choice(search_space['batch_size']),
        }
        
        print(f"\nTrial {trial + 1}/{n_trials}")
        print(f"  Config: {config}")
        
        # Split for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=trial
        )
        
        start_time = time.time()
        
        # Train and evaluate
        model, metrics = train_and_evaluate(X_tr, y_tr, X_val, y_val, config, class_weights)
        
        elapsed = time.time() - start_time
        
        # Find optimal threshold for F1
        y_proba = model.predict(X_val, verbose=0).flatten()
        optimal_thresh = find_optimal_threshold(y_val, y_proba, 0.8)
        y_pred_opt = (y_proba >= optimal_thresh).astype(int)
        f1_optimal = f1_score(y_val, y_pred_opt, zero_division=0)
        
        print(f"  Results: F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}, "
              f"F1@80%Recall={f1_optimal:.4f}, Time={elapsed:.1f}s")
        
        results.append({
            'trial': trial + 1,
            'config': config,
            'metrics': metrics,
            'f1_optimal': f1_optimal,
            'threshold': optimal_thresh,
            'time': elapsed
        })
        
        # Track best
        if f1_optimal > best_score:
            best_score = f1_optimal
            best_config = config
            best_model = model
            print(f"  ★ NEW BEST!")
    
    return results, best_config, best_model

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("CREDIT CARD FRAUD DETECTION - HYPERPARAMETER TUNING")
    print("="*70)
    
    # Load and preprocess data
    print("\nLoading data...")
    train_df = pd.read_csv('fraudTrain.csv')
    X, y, encoders, scaler = preprocess_data(train_df, fit=True)
    
    print(f"Data shape: {X.shape}")
    print(f"Fraud rate: {y.mean()*100:.2f}%")
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    
    # Run tuning
    results, best_config, best_model = random_search_tuning(
        X, y, class_weights, n_trials=15
    )
    
    # Save results
    print("\n" + "="*70)
    print("TUNING COMPLETE")
    print("="*70)
    
    print("\nTop 5 Configurations:")
    results_sorted = sorted(results, key=lambda x: x['f1_optimal'], reverse=True)
    
    for i, r in enumerate(results_sorted[:5]):
        print(f"\n{i+1}. Trial {r['trial']}")
        print(f"   Config: {r['config']}")
        print(f"   F1@80%Recall: {r['f1_optimal']:.4f}")
        print(f"   Threshold: {r['threshold']:.4f}")
    
    print("\n" + "="*70)
    print("BEST CONFIGURATION:")
    print("="*70)
    print(f"  Layers: {best_config['layers']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  F1 Score: {best_score:.4f}")
    
    # Save best model and config
    print("\nSaving best model...")
    best_model.save('models/best_fraud_model.keras')
    joblib.dump(best_config, 'models/best_config.pkl')
    joblib.dump({'encoders': encoders, 'scaler': scaler}, 'models/preprocessors.pkl')
    
    # Save tuning results
    import json
    with open('evaluation/tuning_results.json', 'w') as f:
        # Convert non-serializable items
        json_results = []
        for r in results:
            json_results.append({
                'trial': r['trial'],
                'config': {k: list(v) if isinstance(v, tuple) else v 
                          for k, v in r['config'].items()},
                'metrics': r['metrics'],
                'f1_optimal': r['f1_optimal'],
                'threshold': r['threshold'],
                'time': r['time']
            })
        json.dump(json_results, f, indent=2)
    
    print("\nFiles saved:")
    print("  - models/best_fraud_model.keras")
    print("  - models/best_config.pkl")
    print("  - evaluation/tuning_results.json")

if __name__ == "__main__":
    main()
```

### 12.5 Understanding Tuning Output

```
Example output from tuning:

======================================================================
HYPERPARAMETER TUNING - RANDOM SEARCH
======================================================================
Running 15 trials...

Trial 1/15
  Config: {'layers': [256, 128, 64], 'dropout': 0.3, 'learning_rate': 0.001, 'batch_size': 2048}
  Results: F1=0.6523, AUC=0.9456, F1@80%Recall=0.7123, Time=45.2s
  ★ NEW BEST!

Trial 2/15
  Config: {'layers': [512, 256, 128], 'dropout': 0.2, 'learning_rate': 0.0005, 'batch_size': 1024}
  Results: F1=0.6812, AUC=0.9512, F1@80%Recall=0.7345, Time=52.1s
  ★ NEW BEST!

...

======================================================================
BEST CONFIGURATION:
======================================================================
  Layers: [512, 256, 128]
  Dropout: 0.2
  Learning Rate: 0.0005
  Batch Size: 1024
  F1 Score: 0.7654
```

### 12.6 Tuning Best Practices

```
TUNING TIPS:

1. START BROAD, THEN NARROW
   - First: Try very different architectures
   - Then: Fine-tune around best found

2. PRIORITIZE IMPORTANT PARAMS
   ┌────────────────────────────────────────┐
   │  HIGH IMPACT          │  LOW IMPACT    │
   ├───────────────────────┼────────────────┤
   │  • Learning rate      │  • Kernel init │
   │  • Layer sizes       │  • Bias init   │
   │  • Dropout rate      │                │
   │  • Batch size        │                │
   └───────────────────────┴────────────────┘

3. USE VALIDATION SET
   - Always tune on validation, not test
   - Test set is for final evaluation only

4. TIME MANAGEMENT
   - Start with fewer epochs for tuning
   - Final training can use more epochs

5. TRACK EVERYTHING
   - Log all results
   - Compare systematically
```

---

## Part 13: Neural Network Visualization GUI

### 13.1 Overview

```
VISUALIZATION COMPONENTS:

┌─────────────────────────────────────────────────────────────┐
│                    WHAT WE'LL BUILD                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. ARCHITECTURE VIEW                                      │
│     - Show all layers                                       │
│     - Show number of neurons per layer                     │
│     - Show connections between layers                       │
│                                                             │
│  2. LAYER DETAILS                                          │
│     - Click on layer to see details                        │
│     - Weights, biases, activations                         │
│                                                             │
│  3. TRAINING VISUALIZATION                                 │
│     - Live loss/accuracy curves                            │
│     - Animation of training progress                       │
│                                                             │
│  4. FEATURE IMPORTANCE (approximation)                     │
│     - Which inputs matter most                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 13.2 Visualization Script (visualize_model.py)

```python
"""
Credit Card Fraud Detection - Neural Network Visualization GUI
Interactive visualization of the neural network architecture and training
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import joblib

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ============================================================================
# NEURAL NETWORK CANVAS DRAWER
# ============================================================================

class NeuralNetworkCanvas:
    """
    Draws the neural network architecture on a canvas
    """
    
    def __init__(self, parent, layer_sizes, input_features):
        self.canvas = tk.Canvas(parent, width=1000, height=600, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        self.layer_sizes = layer_sizes
        self.input_features = input_features
        self.layer_positions = []
        
        self.draw_network()
        
        # Add legend
        self.draw_legend()
    
    def draw_network(self):
        """Draw the neural network"""
        self.canvas.delete("all")
        
        # Calculate positions
        num_layers = len(self.layer_sizes)
        start_x = 80
        end_x = 920
        spacing_x = (end_x - start_x) / (num_layers - 1)
        
        # Colors
        input_color = '#3498db'    # Blue
        hidden_color = '#2ecc71'   # Green
        output_color = '#e74c3c'   # Red
        connection_color = '#bdc3c7'  # Gray
        
        self.layer_positions = []
        
        for i, size in enumerate(self.layer_sizes):
            x = start_x + i * spacing_x
            
            # Determine layer type and color
            if i == 0:
                color = input_color
                label = f"Input\n({size} features)"
            elif i == num_layers - 1:
                color = output_color
                label = f"Output\n(1)"
            else:
                color = hidden_color
                label = f"Hidden\n({size})"
            
            # Limit visible neurons for display
            visible_neurons = min(size, 20)
            
            # Calculate vertical positions
            start_y = 100
            end_y = 500
            spacing_y = (end_y - start_y) / (visible_neurons + 1)
            
            positions = []
            for j in range(visible_neurons):
                y = start_y + (j + 1) * spacing_y
                positions.append((x, y))
                
                # Draw neuron
                radius = 15 if size <= 20 else 10
                self.canvas.create_oval(
                    x - radius, y - radius,
                    x + radius, y + radius,
                    fill=color, outline='#2c3e50', width=2
                )
                
                # Add labels for first and last neurons
                if visible_neurons > 5:
                    if j == 0:
                        self.canvas.create_text(x, y - 25, text=f"n₀", 
                                              fill='#2c3e50', font=('Arial', 8))
                    elif j == visible_neurons - 1:
                        self.canvas.create_text(x, y + 25, text=f"n{visible_neurons-1}", 
                                              fill='#2c3e50', font=('Arial', 8))
            
            self.layer_positions.append(positions)
            
            # Draw layer label
            self.canvas.create_text(x, 50, text=label, 
                                  fill='#2c3e50', font=('Arial', 12, 'bold'))
        
        # Draw connections (simplified - only first few)
        self.draw_connections()
    
    def draw_connections(self):
        """Draw connections between layers (simplified)"""
        connection_skip = 5  # Only draw every nth connection
        
        for layer_idx in range(len(self.layer_positions) - 1):
            current_layer = self.layer_positions[layer_idx]
            next_layer = self.layer_positions[layer_idx + 1]
            
            # Only draw connections for first few neurons
            for i in range(min(len(current_layer), 5)):
                for j in range(min(len(next_layer), 3)):
                    x1, y1 = current_layer[i]
                    x2, y2 = next_layer[j]
                    
                    self.canvas.create_line(
                        x1, y1, x2, y2,
                        fill='#bdc3c7', width=0.5
                    )
    
    def draw_legend(self):
        """Draw a legend"""
        legend_y = 560
        
        # Input
        self.canvas.create_oval(100, legend_y, 115, legend_y + 15, 
                              fill='#3498db', outline='#2c3e50')
        self.canvas.create_text(125, legend_y + 7, text="Input Layer", 
                              anchor=tk.W, font=('Arial', 9))
        
        # Hidden
        self.canvas.create_oval(260, legend_y, 275, legend_y + 15, 
                              fill='#2ecc71', outline='#2c3e50')
        self.canvas.create_text(285, legend_y + 7, text="Hidden Layers", 
                              anchor=tk.W, font=('Arial', 9))
        
        # Output
        self.canvas.create_oval(420, legend_y, 435, legend_y + 15, 
                              fill='#e74c3c', outline='#2c3e50')
        self.canvas.create_text(445, legend_y + 7, text="Output Layer", 
                              anchor=tk.W, font=('Arial', 9))


# ============================================================================
# TRAINING VISUALIZER
# ============================================================================

class TrainingVisualizer:
    """
    Shows live training progress with matplotlib
    """
    
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        
        # Create figure
        self.fig = Figure(figsize=(10, 4), dpi=100)
        
        # Subplot for loss
        self.ax_loss = self.fig.add_subplot(121)
        self.ax_loss.set_title('Loss Over Epochs')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        
        # Subplot for AUC
        self.ax_auc = self.fig.add_subplot(122)
        self.ax_auc.set_title('AUC Over Epochs')
        self.ax_auc.set_xlabel('Epoch')
        self.ax_auc.set_ylabel('AUC')
        self.ax_auc.grid(True, alpha=0.3)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.epochs = []
        self.train_loss = []
        self.val_loss = []
        self.train_auc = []
        self.val_auc = []
    
    def add_epoch(self, epoch, train_loss, val_loss, train_auc, val_auc):
        """Add a new epoch's data"""
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_auc.append(train_auc)
        self.val_auc.append(val_auc)
        
        self.update_plot()
    
    def update_plot(self):
        """Redraw the plots"""
        # Clear
        self.ax_loss.clear()
        self.ax_auc.clear()
        
        # Loss plot
        self.ax_loss.plot(self.epochs, self.train_loss, 'b-', label='Train', linewidth=2)
        self.ax_loss.plot(self.epochs, self.val_loss, 'r-', label='Validation', linewidth=2)
        self.ax_loss.set_title('Loss Over Epochs')
        self.ax_loss.set_xlabel('Epoch')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.legend()
        self.ax_loss.grid(True, alpha=0.3)
        
        # AUC plot
        self.ax_auc.plot(self.epochs, self.train_auc, 'b-', label='Train', linewidth=2)
        self.ax_auc.plot(self.epochs, self.val_auc, 'r-', label='Validation', linewidth=2)
        self.ax_auc.set_title('AUC Over Epochs')
        self.ax_auc.set_xlabel('Epoch')
        self.ax_auc.set_ylabel('AUC')
        self.ax_auc.legend()
        self.ax_auc.grid(True, alpha=0.3)
        
        self.canvas.draw()


# ============================================================================
# FEATURE IMPORTANCE DISPLAY
# ============================================================================

class FeatureImportanceDisplay:
    """
    Shows approximate feature importance using input perturbation
    """
    
    def __init__(self, parent, features):
        self.frame = ttk.Frame(parent)
        self.features = features
        
        # Create text widget for display
        self.text = tk.Text(self.frame, height=15, width=40, font=('Courier', 10))
        self.text.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.frame, command=self.text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.config(yscrollcommand=scrollbar.set)
    
    def show_importance(self, model, X_sample, y_sample):
        """Calculate and display approximate feature importance"""
        self.text.delete('1.0', tk.END)
        
        self.text.insert(tk.END, "Approximate Feature Importance\n")
        self.text.insert(tk.END, "=" * 40 + "\n\n")
        self.text.insert(tk.END, "(Using input perturbation method)\n\n")
        
        # Get baseline prediction
        baseline_proba = model.predict(X_sample, verbose=0).mean()
        
        # Perturb each feature
        importance = []
        for i, feat in enumerate(self.features):
            X_perturbed = X_sample.copy()
            X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])
            
            perturbed_proba = model.predict(X_perturbed, verbose=0).mean()
            impact = abs(baseline_proba - perturbed_proba)
            importance.append((feat, impact))
        
        # Sort by importance
        importance.sort(key=lambda x: x[1], reverse=True)
        
        # Display
        max_impact = max(i[1] for i in importance) if importance else 1
        
        for feat, impact in importance:
            bar_len = int((impact / max_impact) * 30)
            bar = '█' * bar_len + '░' * (30 - bar_len)
            self.text.insert(tk.END, f"{feat:15s} {bar} {impact:.4f}\n")


# ============================================================================
# MAIN APPLICATION
# ============================================================================

class FraudDetectionVisualizer:
    """
    Main application window
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Visualization - Fraud Detection")
        self.root.geometry("1100x800")
        
        self.model = None
        self.encoders = None
        self.scaler = None
        self.history = None
        
        # Create menu bar
        self.create_menu()
        
        # Create main layout
        self.create_layout()
    
    def create_menu(self):
        """Create the menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Model...", command=self.load_model)
        file_menu.add_command(label="Load Training History...", command=self.load_history)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
    
    def create_layout(self):
        """Create the main layout"""
        # Top section - Model info
        self.info_frame = ttk.LabelFrame(self.root, text="Model Information")
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_label = ttk.Label(self.info_frame, 
                                    text="No model loaded. Use File > Load Model to load a trained model.")
        self.info_label.pack(padx=10, pady=10)
        
        # Middle section - Network visualization
        self.viz_frame = ttk.LabelFrame(self.root, text="Network Architecture")
        self.viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = None
        self.default_layer_sizes = [14, 64, 32, 16, 1]
        self.default_features = [
            'amt', 'category', 'gender', 'state', 'lat', 'long',
            'city_pop', 'merch_lat', 'merch_long', 'hour',
            'day_of_week', 'month', 'distance_km', 'age'
        ]
        
        # Bottom section - Training history and feature importance
        self.bottom_frame = ttk.Frame(self.root)
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Training history
        self.history_frame = ttk.LabelFrame(self.bottom_frame, text="Training History")
        self.history_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.training_viz = TrainingVisualizer(self.history_frame)
        self.training_viz.frame.pack(fill=tk.BOTH, expand=True)
        
        # Feature importance
        self.importance_frame = ttk.LabelFrame(self.bottom_frame, text="Feature Importance")
        self.importance_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self.importance_viz = FeatureImportanceDisplay(self.importance_frame, self.default_features)
        self.importance_viz.frame.pack(fill=tk.BOTH, expand=True)
        
        # Draw default network
        self.show_network(self.default_layer_sizes, self.default_features)
    
    def load_model(self):
        """Load a trained model"""
        if not TF_AVAILABLE:
            messagebox.showerror("Error", "TensorFlow is not installed!")
            return
        
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Keras Models", "*.keras"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                self.model = tf.keras.models.load_model(filename)
                
                # Get layer info
                layer_sizes = [self.model.input_shape[-1]]
                for layer in self.model.layers:
                    if hasattr(layer, 'units'):
                        layer_sizes.append(layer.units)
                
                # Update info
                info_text = f"Model: {filename.split('/')[-1]}\n"
                info_text += f"Layers: {len(self.model.layers)}\n"
                info_text += f"Architecture: {' → '.join(map(str, layer_sizes))}\n"
                info_text += f"Total Parameters: {self.model.count_params():,}"
                
                self.info_label.config(text=info_text)
                
                # Show network
                self.show_network(layer_sizes, self.default_features)
                
                messagebox.showinfo("Success", "Model loaded successfully!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def load_history(self):
        """Load training history"""
        filename = filedialog.askopenfilename(
            title="Select History File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if filename:
            try:
                import json
                with open(filename, 'r') as f:
                    history = json.load(f)
                
                # Add data to visualizer
                for i, (loss, val_loss, auc, val_auc) in enumerate(zip(
                    history.get('loss', []),
                    history.get('val_loss', []),
                    history.get('auc', []),
                    history.get('val_auc', [])
                )):
                    self.training_viz.add_epoch(i+1, loss, val_loss, auc, val_auc)
                
                messagebox.showinfo("Success", "Training history loaded!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load history: {str(e)}")
    
    def show_network(self, layer_sizes, features):
        """Show the neural network"""
        if self.canvas:
            self.canvas.destroy()
        
        self.canvas = NeuralNetworkCanvas(self.viz_frame, layer_sizes, features)


# ============================================================================
# RUN
# ============================================================================

def main():
    root = tk.Tk()
    app = FraudDetectionVisualizer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
```

### 13.3 How to Use the Visualization

```bash
# Run the visualizer
python visualize_model.py
```

**What you'll see:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Neural Network Visualization                       │
├─────────────────────────────────────────────────────────────────────┤
│  File > Load Model                                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Model Information                                                │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Model: fraud_model.keras                                    │  │
│  │ Layers: 4                                                  │  │
│  │ Architecture: 14 → 256 → 128 → 64 → 1                     │  │
│  │ Total Parameters: 47,745                                     │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  Network Architecture                                             │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                                                             │  │
│  │  INPUT    HIDDEN    HIDDEN    HIDDEN    OUTPUT             │  │
│  │   (14)     256       128       64         1              │  │
│  │                                                             │  │
│  │    ○ ───────○───────○───────○───────○                     │  │
│  │    ○ ───────○───────○───────○───────○                     │  │
│  │    ○ ───────○───────○───────○───────○                     │  │
│  │    ○ ───────○───────○───────○───────○                     │  │
│  │    ...                                                      │  │
│  │                                                             │  │
│  │  ○ = Blue (Input)  ○ = Green (Hidden)  ○ = Red (Output) │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌──────────────────────┐  ┌────────────────────────────────┐   │
│  │   Training History    │  │     Feature Importance          │   │
│  │  ┌────────────────┐  │  │  ┌────────────────────────────┐  │   │
│  │  │ Loss: 0.0234  │  │  │  │ distance_km  ████████░░░ │  │   │
│  │  │ AUC: 0.9823   │  │  │  │ amt         ███████░░░░ │  │   │
│  │  │                │  │  │  │ hour        █████░░░░░░░ │  │   │
│  │  │ [Loss Graph]  │  │  │  │ category    ████░░░░░░░░ │  │   │
│  │  │ [AUC Graph]   │  │  │  │ age         ███░░░░░░░░░ │  │   │
│  │  └────────────────┘  │  │  │ ...                        │  │   │
│  └──────────────────────┘  └────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 13.4 Visualizer Features Explained

```
VISUALIZATION COMPONENTS:

1. NETWORK ARCHITECTURE (Top Section)
   ┌────────────────────────────────────────────────────────┐
   │ INPUT    HIDDEN    HIDDEN    HIDDEN    OUTPUT        │
   │  (14)     256       128       64         1        │
   │                                                     │
   │    ●───────●───────●───────●───────●               │
   │    ●───────●───────●───────●───────●               │
   │    ●───────●───────●───────●───────●               │
   └────────────────────────────────────────────────────────┘
   
   - Each ● = Neuron
   - Lines = Connections
   - Color coding by layer type

2. TRAINING HISTORY (Bottom Left)
   ┌────────────────────────────────┐
   │ Loss          │    AUC          │
   │               │                 │
   │   📈          │     📈         │
   │  Train/Val    │    Train/Val   │
   └────────────────────────────────┘
   
   - Shows loss decreasing over epochs
   - Shows AUC increasing over epochs
   - Compares train vs validation

3. FEATURE IMPORTANCE (Bottom Right)
   ┌────────────────────────────────┐
   │ distance_km   ████████░░░ 0.23 │
   │ amt           ███████░░░░ 0.18 │
   │ hour          █████░░░░░░ 0.12 │
   └────────────────────────────────┘
   
   - Shows which input features matter most
   - Uses perturbation method to estimate
```

---

## Part 14: Complete Project Workflow

### 14.1 Step-by-Step Execution

```
COMPLETE WORKFLOW:

┌─────────────────────────────────────────────────────────────────┐
│  STEP 1: HYPERPARAMETER TUNING                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  python tune_model.py                                           │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Tests 15+ random configurations                            │  │
│  │ • Different layer sizes                                   │  │
│  │ • Different dropout rates                                 │  │
│  │ • Different learning rates                                │  │
│  │ • Different batch sizes                                   │  │
│  └──────────────────────────────────────────────────────────┘  │
│       │                                                         │
│       ▼                                                         │
│  Best configuration found!                                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 2: TRAIN FINAL MODEL                                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  python train_model.py                                          │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Uses best config from tuning                             │  │
│  │ Trains with more epochs                                  │  │
│  │ Evaluates on test set                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│       │                                                         │
│       ▼                                                         │
│  Saved: models/fraud_model.keras + preprocessors                │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 3: VISUALIZE MODEL                                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  python visualize_model.py                                       │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ • See network architecture                               │  │
│  │ • Load trained model                                     │  │
│  │ • View feature importance                                │  │
│  │ • View training history                                  │  │
│  └──────────────────────────────────────────────────────────┘  │
│       │                                                         │
│       ▼                                                         │
│  Interactive GUI!                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  STEP 4: MAKE PREDICTIONS                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  python predict.py                                              │
│       OR                                                        │
│  python app.py  (Web UI)                                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 14.2 Updated File Structure

```
fraud-detection/
├── fraudTrain.csv                    # Training data
├── fraudTest.csv                     # Test data
├── requirements.txt                  # Dependencies
│
├── train_model.py                    # Train with best config
├── tune_model.py                    # Hyperparameter tuning
├── predict.py                       # Make predictions
├── visualize_model.py               # Visualize network (GUI)
├── app.py                           # Web UI
│
├── models/
│   ├── fraud_model.keras           # Trained model
│   ├── best_config.pkl             # Best hyperparameters
│   ├── encoders.pkl               # Label encoders
│   ├── scaler.pkl                 # StandardScaler
│   └── optimal_threshold.txt       # Decision threshold
│
├── evaluation/
│   ├── tuning_results.json         # All tuning results
│   ├── training_history.png         # Training plots
│   └── classification_report.txt    # Metrics
│
└── FRAUD_DETECTION_TUTORIAL.md     # This tutorial
```

---

## Running the Project

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Tune hyperparameters (find best configuration)
python tune_model.py

# Step 3: Train the model with best configuration
python train_model.py

# Step 4: Visualize the neural network (GUI)
python visualize_model.py
# Then: File > Load Model > Select models/fraud_model.keras

# Step 5: Test prediction
python predict.py

# Step 6: Run web UI (optional)
python app.py
# Open http://localhost:5000
```

---

## Quick Reference

| Component | Description |
|-----------|-------------|
| Architecture | MLP: Configurable layers (tuned) |
| Activation | ReLU (hidden), Sigmoid (output) |
| Regularization | BatchNorm + Dropout (tuned) |
| Optimizer | Adam (tuned learning rate) |
| Loss | Binary Cross-Entropy |
| Class Handling | Class weights (balanced) |
| Framework | TensorFlow/Keras |
| Tuning | Random Search |
| Visualization | Tkinter GUI + Matplotlib |

| Component | Description |
|-----------|-------------|
| Architecture | MLP: 256 → 128 → 64 → 1 |
| Activation | ReLU (hidden), Sigmoid (output) |
| Regularization | BatchNorm + Dropout(0.3) |
| Optimizer | Adam (lr=0.001) |
| Loss | Binary Cross-Entropy |
| Class Handling | Class weights (balanced) |
| Framework | TensorFlow/Keras |

---

## Deeper Topics to Explore

1. **Convolutional Neural Networks (CNN)** - For spatial patterns
2. **Recurrent Neural Networks (LSTM/GRU)** - For sequential transaction history
3. **Attention Mechanisms** - For focusing on important features
4. **Autoencoders** - For anomaly detection
5. **Transfer Learning** - Pre-trained models for fraud detection

---

## References

- [TensorFlow Keras Guide](https://www.tensorflow.org/guide/keras)
- [Deep Learning for Fraud Detection](https://arxiv.org/abs/1910.01777)
- [Class Imbalance Problem](https://arxiv.org/abs/1710.05381)
