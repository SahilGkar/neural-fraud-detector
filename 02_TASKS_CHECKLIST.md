# Credit Card Fraud Detection - Step-by-Step Tasks

## Overview

This guide breaks down the entire fraud detection project into small, manageable tasks. Complete each task in order, checking them off as you go.

---

## Prerequisites Checklist

Before starting, verify you have:

- [ ] Python 3.8+ installed (`python3 --version`)
- [ ] Git installed (optional)
- [ ] 10GB+ free disk space
- [ ] Text editor (VS Code recommended)
- [ ] Terminal/Command prompt access

---

## Phase 1: Environment Setup

### Task 1.1: Create Project Folder
```bash
mkdir fraud-detection
cd fraud-detection
```
- [ ] Create the folder
- [ ] Navigate into it
- [ ] Verify you're in the correct directory (`pwd`)

### Task 1.2: Create Virtual Environment
```bash
python3 -m venv venv

# Activate it:
# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```
- [ ] Create virtual environment
- [ ] Activate it
- [ ] Verify activation (you should see `(venv)` in terminal)

### Task 1.3: Install Dependencies
Create a file called `requirements.txt`:
```
tensorflow>=2.15.0
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
flask
```

Then run:
```bash
pip install -r requirements.txt
```
- [ ] Create `requirements.txt`
- [ ] Run `pip install`
- [ ] Wait for installation to complete (may take 5-10 minutes)
- [ ] Verify TensorFlow installed: `python -c "import tensorflow as tf; print(tf.__version__)"`

### Task 1.4: Move Dataset Files
```bash
# Move your CSV files into this project folder
mv /path/to/fraudTrain.csv .
mv /path/to/fraudTest.csv .

# Or copy:
cp /path/to/fraudTrain.csv .
cp /path/to/fraudTest.csv .
```
- [ ] Move `fraudTrain.csv` into project folder
- [ ] Move `fraudTest.csv` into project folder
- [ ] Verify both files exist: `ls -la *.csv`

---

## Phase 2: Data Exploration

### Task 2.1: Create Data Explorer Script
Create `explore_data.py`:
```python
import pandas as pd

# Load data
train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')

print("=" * 50)
print("TRAINING DATA")
print("=" * 50)
print(f"Shape: {train_df.shape}")
print(f"\nColumns: {list(train_df.columns)}")
print(f"\nData Types:\n{train_df.dtypes}")
print(f"\nMissing Values:\n{train_df.isnull().sum()}")
print(f"\nFraud Distribution:\n{train_df['is_fraud'].value_counts()}")

print("\n" + "=" * 50)
print("TEST DATA")
print("=" * 50)
print(f"Shape: {test_df.shape}")
print(f"\nFraud Distribution:\n{test_df['is_fraud'].value_counts()}")
```

Run it:
```bash
python explore_data.py
```

**Deliverables to note:**
- [ ] Number of rows in train
- [ ] Number of rows in test
- [ ] Fraud percentage in train
- [ ] Fraud percentage in test
- [ ] List of all columns

### Task 2.2: Explore Individual Columns
Create `explore_columns.py`:
```python
import pandas as pd

df = pd.read_csv('fraudTrain.csv')

# Check each column
columns_to_check = ['category', 'gender', 'state', 'amt', 'lat', 'long']

for col in columns_to_check:
    print(f"\n{'='*40}")
    print(f"Column: {col}")
    print(f"{'='*40}")
    print(f"Type: {df[col].dtype}")
    
    if df[col].dtype == 'object':
        print(f"Unique values: {df[col].nunique()}")
        print(f"Top 5:\n{df[col].value_counts().head()}")
    else:
        print(f"Min: {df[col].min()}")
        print(f"Max: {df[col].max()}")
        print(f"Mean: {df[col].mean():.2f}")
```

Run it:
```bash
python explore_columns.py
```

**Deliverables to note:**
- [ ] List of categorical columns
- [ ] List of numerical columns
- [ ] Range of transaction amounts
- [ ] Number of unique categories

---

## Phase 3: Data Preprocessing

### Task 3.1: Create Directory Structure
```bash
mkdir models
mkdir evaluation
mkdir templates
```
- [ ] Create `models/` folder
- [ ] Create `evaluation/` folder
- [ ] Create `templates/` folder

### Task 3.2: Implement Preprocessing Function
Create `preprocess.py`:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points in km"""
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

# Test it
if __name__ == "__main__":
    print("Loading training data...")
    train_df = pd.read_csv('fraudTrain.csv')
    
    print("Preprocessing...")
    X, y, encoders, scaler = preprocess_data(train_df, fit=True)
    
    print(f"\nProcessed data shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Fraud rate: {y.mean()*100:.2f}%")
    print(f"Features: {X.shape[1]}")
    
    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump({'encoders': encoders, 'scaler': scaler}, 'models/preprocessors.pkl')
    print("\nSaved preprocessors to models/preprocessors.pkl")
```

Run it:
```bash
python preprocess.py
```

**Verification:**
- [ ] Script runs without errors
- [ ] X shape matches expected (should be ~1.3M rows × 14 features)
- [ ] Fraud rate matches original data
- [ ] Preprocessors saved

---

## Phase 4: Neural Network Basics

### Task 4.1: Understand the Architecture
Create `build_model.py`:
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, BatchNormalization, Dropout, Activation
)

def build_model(input_dim=14):
    """Build MLP for fraud detection"""
    model = Sequential([
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
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model

# Test it
if __name__ == "__main__":
    model = build_model(input_dim=14)
    model.summary()
    
    print("\n" + "="*50)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*50)
    print(f"Total layers: {len(model.layers)}")
    print(f"Total parameters: {model.count_params():,}")
```

Run it:
```bash
python build_model.py
```

**Expected output:**
```
Layer 1: 256 neurons (Dense + BatchNorm + Dropout + ReLU)
Layer 2: 128 neurons (Dense + BatchNorm + Dropout + ReLU)
Layer 3: 64 neurons (Dense + BatchNorm + Dropout + ReLU)
Output: 1 neuron (Dense + Sigmoid)
Total parameters: ~47,745
```

**Verification:**
- [ ] Model builds successfully
- [ ] Summary displays correctly
- [ ] Note total parameter count

### Task 4.2: Test Forward Pass
Add to `build_model.py`:
```python
# After the model summary section, add:

if __name__ == "__main__":
    # ... existing code ...
    
    # Test forward pass
    import numpy as np
    print("\n" + "="*50)
    print("TESTING FORWARD PASS")
    print("="*50)
    
    # Create random input
    X_test = np.random.randn(5, 14).astype(np.float32)
    print(f"Input shape: {X_test.shape}")
    
    # Get prediction
    predictions = model.predict(X_test, verbose=0)
    print(f"Output shape: {predictions.shape}")
    print(f"Sample predictions: {predictions.flatten()}")
```

Run it:
```bash
python build_model.py
```

**Verification:**
- [ ] Forward pass works
- [ ] Output shape is (5, 1)
- [ ] Predictions are between 0 and 1 (due to sigmoid)

---

## Phase 5: Training

### Task 5.1: Implement Training Script
Create `train_model.py`:
```python
"""
Credit Card Fraud Detection - Neural Network Training
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import preprocessing from Task 3.2
from preprocess import preprocess_data

def build_model(input_dim):
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
        metrics=['accuracy', 'AUC']
    )
    
    return model

def find_optimal_threshold(y_true, y_proba, target_recall=0.8):
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    distances = np.abs(recalls - target_recall)
    best_idx = np.argmin(distances)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def main():
    print("="*60)
    print("CREDIT CARD FRAUD DETECTION - TRAINING")
    print("="*60)
    
    # Load and preprocess
    print("\n[1/5] Loading data...")
    train_df = pd.read_csv('fraudTrain.csv')
    X, y, encoders, scaler = preprocess_data(train_df, fit=True)
    print(f"  Data shape: {X.shape}")
    
    # Split
    print("\n[2/5] Creating train/validation split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    # Class weights
    print("\n[3/5] Calculating class weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    print(f"  Class 0: {class_weights[0]:.4f} | Class 1: {class_weights[1]:.4f}")
    
    # Build and train
    print("\n[4/5] Training model...")
    model = build_model(X_train.shape[1])
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
    ]
    
    history = model.fit(
        X_train, y_train,
        batch_size=2048,
        epochs=50,
        validation_data=(X_val, y_val),
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n[5/5] Evaluating...")
    y_proba = model.predict(X_val, verbose=0).flatten()
    optimal_threshold = find_optimal_threshold(y_val, y_proba, 0.8)
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"Precision: {precision_score(y_val, y_pred):.4f}")
    print(f"Recall: {recall_score(y_val, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_val, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_proba):.4f}")
    
    # Save
    print("\nSaving model...")
    model.save('models/fraud_model.keras')
    joblib.dump({'encoders': encoders, 'scaler': scaler}, 'models/preprocessors.pkl')
    with open('models/optimal_threshold.txt', 'w') as f:
        f.write(str(optimal_threshold))
    
    # Save training history
    import json
    with open('evaluation/training_history.json', 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
```

Run it:
```bash
python train_model.py
```

**Note:** This will take 15-30 minutes depending on your hardware.

**Verification:**
- [ ] Training starts
- [ ] Loss decreases over epochs
- [ ] Validation loss decreases
- [ ] Metrics calculated
- [ ] Model saved to `models/fraud_model.keras`
- [ ] Preprocessors saved

---

## Phase 6: Hyperparameter Tuning

### Task 6.1: Create Tuning Script
Create `tune_model.py`:
```python
"""
Hyperparameter Tuning - Random Search
"""

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Input, BatchNormalization, Dropout, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from preprocess import preprocess_data

def build_model(input_dim, config):
    model = Sequential([Input(shape=(input_dim,))])
    
    for neurons in config['layers']:
        model.add(Dense(neurons, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Dropout(config['dropout']))
        model.add(Activation('relu'))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['AUC']
    )
    
    return model

def main():
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('fraudTrain.csv')
    X, y, _, _ = preprocess_data(train_df, fit=True)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = {0: class_weights[0], 1: class_weights[1]}
    
    # Define search space
    search_space = {
        'layers': [[256, 128, 64], [512, 256, 128], [128, 64, 32]],
        'dropout': [0.2, 0.3, 0.4],
        'learning_rate': [0.0005, 0.001, 0.005],
        'batch_size': [1024, 2048],
    }
    
    results = []
    best_score = 0
    best_config = None
    
    print(f"\nRunning 10 trials...")
    
    for trial in range(10):
        config = {
            'layers': search_space['layers'][trial % len(search_space['layers'])],
            'dropout': search_space['dropout'][trial % len(search_space['dropout'])],
            'learning_rate': search_space['learning_rate'][trial % len(search_space['learning_rate'])],
            'batch_size': search_space['batch_size'][trial % len(search_space['batch_size'])],
        }
        
        print(f"\nTrial {trial + 1}/10: {config}")
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=trial
        )
        
        model = build_model(X_tr.shape[1], config)
        
        model.fit(
            X_tr, y_tr,
            batch_size=config['batch_size'],
            epochs=20,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
            verbose=0
        )
        
        y_proba = model.predict(X_val, verbose=0).flatten()
        auc = roc_auc_score(y_val, y_proba)
        
        print(f"  AUC: {auc:.4f}")
        
        if auc > best_score:
            best_score = auc
            best_config = config
            print(f"  ★ NEW BEST!")
        
        results.append({'trial': trial + 1, 'config': config, 'auc': auc})
    
    print("\n" + "="*60)
    print(f"BEST CONFIGURATION (AUC: {best_score:.4f})")
    print("="*60)
    print(f"Layers: {best_config['layers']}")
    print(f"Dropout: {best_config['dropout']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Batch Size: {best_config['batch_size']}")
    
    # Save results
    import json
    with open('evaluation/tuning_results.json', 'w') as f:
        json.dump({'best_config': best_config, 'results': results}, f, indent=2)
    
    print("\nResults saved to evaluation/tuning_results.json")

if __name__ == "__main__":
    main()
```

Run it:
```bash
python tune_model.py
```

**Note:** This will take 30-60 minutes.

**Verification:**
- [ ] All 10 trials complete
- [ ] Best configuration identified
- [ ] Results saved

---

## Phase 7: Evaluation

### Task 7.1: Evaluate on Test Set
Create `evaluate_model.py`:
```python
"""
Evaluate trained model on test set
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

def find_threshold(y_true, y_proba, target_recall=0.8):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    distances = np.abs(recalls - target_recall)
    best_idx = np.argmin(distances)
    return thresholds[best_idx] if best_idx < len(thresholds) else 0.5

def main():
    print("="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load model and preprocessors
    print("\nLoading model...")
    model = tf.keras.models.load_model('models/fraud_model.keras', compile=False)
    preprocessors = joblib.load('models/preprocessors.pkl')
    
    from preprocess import preprocess_data
    
    # Load and preprocess test data
    print("Loading test data...")
    test_df = pd.read_csv('fraudTest.csv')
    X_test, y_test, _, _ = preprocess_data(
        test_df, 
        encoders=preprocessors['encoders'],
        scaler=preprocessors['scaler'],
        fit=False
    )
    
    print(f"Test samples: {len(X_test):,}")
    print(f"Test fraud rate: {y_test.mean()*100:.2f}%")
    
    # Predict
    print("\nMaking predictions...")
    y_proba = model.predict(X_test, verbose=1).flatten()
    
    # Find optimal threshold
    optimal_threshold = find_threshold(y_test, y_proba, 0.8)
    y_pred = (y_proba >= optimal_threshold).astype(int)
    
    # Metrics
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"\nPrecision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['LEGIT', 'FRAUD']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              LEGIT      FRAUD")
    print(f"Actual LEGIT  {cm[0,0]:>8,}   {cm[0,1]:>8,}")
    print(f"       FRAUD  {cm[1,0]:>8,}   {cm[1,1]:>8,}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LEGIT', 'FRAUD'],
                yticklabels=['LEGIT', 'FRAUD'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set')
    plt.savefig('evaluation/confusion_matrix.png', dpi=150)
    plt.close()
    
    print("\nConfusion matrix saved to evaluation/confusion_matrix.png")

if __name__ == "__main__":
    import tensorflow as tf
    main()
```

Run it:
```bash
python evaluate_model.py
```

**Verification:**
- [ ] Predictions complete
- [ ] Metrics calculated
- [ ] Confusion matrix generated
- [ ] Results match expectations

---

## Phase 8: Visualization GUI

### Task 8.1: Create Visualization Script
Create `visualize_model.py`:
```python
"""
Neural Network Visualization GUI
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class NetworkCanvas:
    def __init__(self, parent, layer_sizes):
        self.canvas = tk.Canvas(parent, width=900, height=400, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.layer_sizes = layer_sizes
        self.draw_network()
    
    def draw_network(self):
        self.canvas.delete("all")
        
        num_layers = len(self.layer_sizes)
        start_x, end_x = 50, 850
        spacing_x = (end_x - start_x) / (num_layers - 1)
        
        colors = {0: '#3498db', num_layers-1: '#e74c3c'}
        colors = {i: colors.get(i, '#2ecc71') for i in range(num_layers)}
        
        layer_labels = ['Input'] + [f'Hidden {i}' for i in range(1, num_layers-1)] + ['Output']
        
        for i, (size, label) in enumerate(zip(self.layer_sizes, layer_labels)):
            x = start_x + i * spacing_x
            
            # Layer label
            self.canvas.create_text(x, 30, text=label, font=('Arial', 10, 'bold'))
            self.canvas.create_text(x, 50, text=f'({size})', font=('Arial', 9))
            
            # Draw neurons (max 10 visible)
            visible = min(size, 10)
            start_y, end_y = 80, 350
            spacing_y = (end_y - start_y) / (visible + 1)
            
            for j in range(visible):
                y = start_y + (j + 1) * spacing_y
                radius = 15
                self.canvas.create_oval(
                    x - radius, y - radius, x + radius, y + radius,
                    fill=colors[i], outline='#2c3e50', width=2
                )

class TrainingPlot:
    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.fig = Figure(figsize=(8, 4))
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title('Training History')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack()
        
        self.epochs = []
        self.losses = []
        self.val_losses = []
    
    def add_epoch(self, epoch, loss, val_loss):
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.val_losses.append(val_loss)
        self.update()
    
    def update(self):
        self.ax.clear()
        self.ax.plot(self.epochs, self.losses, 'b-', label='Train Loss')
        self.ax.plot(self.epochs, self.val_losses, 'r-', label='Val Loss')
        self.ax.set_title('Training History')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        self.ax.legend()
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

class FraudVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Visualizer")
        self.root.geometry("1000x700")
        
        # Info frame
        info_frame = ttk.LabelFrame(root, text="Model Info")
        info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_label = ttk.Label(info_frame, text="Load a model to see its architecture")
        self.info_label.pack(padx=10, pady=10)
        
        # Canvas frame
        canvas_frame = ttk.LabelFrame(root, text="Network Architecture")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Default architecture: 14 -> 256 -> 128 -> 64 -> 1
        self.network_canvas = NetworkCanvas(canvas_frame, [14, 256, 128, 64, 1])
        
        # Training plot frame
        plot_frame = ttk.LabelFrame(root, text="Training Progress")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.training_plot = TrainingPlot(plot_frame)
        self.training_plot.frame.pack(fill=tk.BOTH, expand=True)
        
        # Load button
        btn_frame = ttk.Frame(root)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(btn_frame, text="Load Model", command=self.load_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Load History", command=self.load_history).pack(side=tk.LEFT, padx=5)
    
    def load_model(self):
        try:
            import tensorflow as tf
            filename = filedialog.askopenfilename(
                title="Select Model",
                filetypes=[("Keras Models", "*.keras"), ("All Files", "*.*")]
            )
            
            if filename:
                model = tf.keras.models.load_model(filename, compile=False)
                
                layer_sizes = [model.input_shape[-1]]
                for layer in model.layers:
                    if hasattr(layer, 'units'):
                        layer_sizes.append(layer.units)
                
                self.network_canvas.layer_sizes = layer_sizes
                self.network_canvas.draw_network()
                
                params = sum(layer.count_params() for layer in model.layers if hasattr(layer, 'count_params'))
                
                self.info_label.config(
                    text=f"Model: {filename.split('/')[-1]}\n"
                         f"Layers: {len(layer_sizes)}\n"
                         f"Architecture: {' → '.join(map(str, layer_sizes))}\n"
                         f"Parameters: {params:,}"
                )
                
                messagebox.showinfo("Success", "Model loaded!")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def load_history(self):
        try:
            filename = filedialog.askopenfilename(
                title="Select History",
                filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
            )
            
            if filename:
                import json
                with open(filename, 'r') as f:
                    history = json.load(f)
                
                for i, (loss, val_loss) in enumerate(zip(
                    history.get('loss', []),
                    history.get('val_loss', [])
                )):
                    self.training_plot.add_epoch(i+1, loss, val_loss)
                
                messagebox.showinfo("Success", "History loaded!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    app = FraudVisualizerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
```

Run it:
```bash
python visualize_model.py
```

**Verification:**
- [ ] GUI opens
- [ ] Default network displayed
- [ ] Can load a trained model
- [ ] Can load training history

---

## Phase 9: Predictions

### Task 9.1: Create Prediction Script
Create `predict.py`:
```python
"""
Make predictions on new transactions
"""

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def preprocess_transaction(data, encoders, scaler):
    df = pd.DataFrame([data])
    
    trans_time = pd.to_datetime(data['trans_date_trans_time'])
    df['hour'] = trans_time.hour
    df['day_of_week'] = trans_time.dayofweek
    df['month'] = trans_time.month
    df['distance_km'] = haversine_distance(
        data['lat'], data['long'], data['merch_lat'], data['merch_long']
    )
    dob = pd.to_datetime(data['dob'])
    df['age'] = ((trans_time - dob).days / 365)
    
    df['category'] = encoders['category'].transform([data['category']])[0]
    df['gender'] = encoders['gender'].transform([data['gender']])[0]
    df['state'] = encoders['state'].transform([data['state']])[0]
    
    features = ['amt', 'category', 'gender', 'state', 'lat', 'long',
                'city_pop', 'merch_lat', 'merch_long', 'hour',
                'day_of_week', 'month', 'distance_km', 'age']
    
    X = df[features].values.astype(np.float32)
    X = scaler.transform(X)
    
    return X

def predict(transaction_data):
    model = tf.keras.models.load_model('models/fraud_model.keras', compile=False)
    preprocessors = joblib.load('models/preprocessors.pkl')
    
    with open('models/optimal_threshold.txt', 'r') as f:
        THRESHOLD = float(f.read())
    
    X = preprocess_transaction(transaction_data, preprocessors['encoders'], preprocessors['scaler'])
    proba = model.predict(X, verbose=0)[0, 0]
    is_fraud = proba >= THRESHOLD
    
    risk = 'HIGH' if proba >= 0.7 else 'MEDIUM' if proba >= 0.4 else 'LOW'
    
    return {
        'probability': round(proba * 100, 2),
        'is_fraud': bool(is_fraud),
        'risk_level': risk,
        'threshold': THRESHOLD
    }

# Test it
if __name__ == "__main__":
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
    
    print("\n" + "="*50)
    print("FRAUD PREDICTION")
    print("="*50)
    
    result = predict(transaction)
    
    print(f"\nTransaction: ${transaction['amt']} at {transaction['merchant']}")
    print(f"Fraud Probability: {result['probability']}%")
    print(f"Prediction: {'⚠️ FRAUD' if result['is_fraud'] else '✅ LEGITIMATE'}")
    print(f"Risk Level: {result['risk_level']}")
    print("="*50)
```

Run it:
```bash
python predict.py
```

**Verification:**
- [ ] Prediction runs without errors
- [ ] Probability returned
- [ ] Correct fraud/legit classification

---

## Phase 10: Web UI (Optional)

### Task 10.1: Create Flask App
Create `app.py`:
```python
"""
Flask Web UI for Fraud Detection
"""

from flask import Flask, render_template, request, jsonify
import joblib
import tensorflow as tf
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load at startup
model = tf.keras.models.load_model('models/fraud_model.keras', compile=False)
preprocessors = joblib.load('models/preprocessors.pkl')

with open('models/optimal_threshold.txt', 'r') as f:
    THRESHOLD = float(f.read())

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def predict(data):
    df = pd.DataFrame([data])
    
    trans_time = pd.to_datetime(data['trans_date_trans_time'])
    df['hour'] = trans_time.hour
    df['day_of_week'] = trans_time.dayofweek
    df['month'] = trans_time.month
    df['distance_km'] = haversine_distance(
        float(data['lat']), float(data['long']),
        float(data['merch_lat']), float(data['merch_long'])
    )
    dob = pd.to_datetime(data['dob'])
    df['age'] = (trans_time - dob).days / 365
    
    df['category'] = preprocessors['encoders']['category'].transform([data['category']])[0]
    df['gender'] = preprocessors['encoders']['gender'].transform([data['gender']])[0]
    df['state'] = preprocessors['encoders']['state'].transform([data['state']])[0]
    
    features = ['amt', 'category', 'gender', 'state', 'lat', 'long',
                'city_pop', 'merch_lat', 'merch_long', 'hour',
                'day_of_week', 'month', 'distance_km', 'age']
    
    X = df[features].values.astype(np.float32)
    X = preprocessors['scaler'].transform(X)
    
    proba = model.predict(X, verbose=0)[0, 0]
    is_fraud = proba >= THRESHOLD
    
    return {
        'probability': round(proba * 100, 1),
        'prediction': 'FRAUD' if is_fraud else 'LEGITIMATE',
        'risk': 'HIGH' if proba >= 0.7 else 'MEDIUM' if proba >= 0.4 else 'LOW',
        'threshold': THRESHOLD
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    result = predict(data)
    return jsonify(result)

if __name__ == '__main__':
    print(f"Starting server with threshold: {THRESHOLD:.4f}")
    app.run(debug=True, port=5000)
```

### Task 10.2: Create HTML Template
Create `templates/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Credit Card Fraud Detection</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .card {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 { color: #333; text-align: center; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input, select {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            box-sizing: border-box;
        }
        button {
            width: 100%;
            padding: 15px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover { background: #0056b3; }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 5px;
            text-align: center;
            display: none;
        }
        .fraud { background: #f8d7da; border: 2px solid #dc3545; }
        .legit { background: #d4edda; border: 2px solid #28a745; }
        .prob { font-size: 36px; font-weight: bold; }
    </style>
</head>
<body>
    <div class="card">
        <h1>🔒 Fraud Detection</h1>
        <form id="fraudForm">
            <div class="form-group">
                <label>Amount ($)</label>
                <input type="number" step="0.01" name="amt" required>
            </div>
            <div class="form-group">
                <label>Category</label>
                <select name="category">
                    <option value="grocery_pos">Grocery</option>
                    <option value="entertainment">Entertainment</option>
                    <option value="gas_transport">Gas/Transport</option>
                    <option value="food_dining">Food/Dining</option>
                    <option value="shopping_pos">Shopping</option>
                </select>
            </div>
            <div class="form-group">
                <label>Cardholder Lat/Long</label>
                <input type="number" step="0.0001" name="lat" placeholder="Lat" required>
                <input type="number" step="0.0001" name="long" placeholder="Long" style="margin-top:5px" required>
            </div>
            <div class="form-group">
                <label>Merchant Lat/Long</label>
                <input type="number" step="0.0001" name="merch_lat" placeholder="Lat" required>
                <input type="number" step="0.0001" name="merch_long" placeholder="Long" style="margin-top:5px" required>
            </div>
            <div class="form-group">
                <label>Gender</label>
                <select name="gender">
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                </select>
            </div>
            <div class="form-group">
                <label>State (2-letter code)</label>
                <input type="text" name="state" maxlength="2" placeholder="CA" required>
            </div>
            <div class="form-group">
                <label>City Population</label>
                <input type="number" name="city_pop" required>
            </div>
            <div class="form-group">
                <label>Date of Birth</label>
                <input type="date" name="dob" required>
            </div>
            <div class="form-group">
                <label>Transaction Date/Time</label>
                <input type="datetime-local" name="trans_date_trans_time" required>
            </div>
            <button type="submit">Analyze Transaction</button>
        </form>
        
        <div id="result" class="result">
            <div class="prediction"></div>
            <div class="prob"></div>
            <div class="risk"></div>
        </div>
    </div>

    <script>
        document.getElementById('fraudForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const form = e.target;
            const data = Object.fromEntries(new FormData(form));
            
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(data)
            });
            
            const result = await res.json();
            
            const resultDiv = document.getElementById('result');
            resultDiv.style.display = 'block';
            resultDiv.className = 'result ' + (result.prediction === 'FRAUD' ? 'fraud' : 'legit');
            
            resultDiv.querySelector('.prediction').textContent = result.prediction;
            resultDiv.querySelector('.prob').textContent = result.probability + '%';
            resultDiv.querySelector('.risk').textContent = 'Risk: ' + result.risk;
        });
    </script>
</body>
</html>
```

Run it:
```bash
python app.py
# Open http://localhost:5000
```

**Verification:**
- [ ] Web app opens
- [ ] Form works
- [ ] Predictions display correctly

---

## Final Checklist

Before declaring the project complete, verify:

### Training
- [ ] Model trains without errors
- [ ] Loss decreases over epochs
- [ ] Validation metrics improve
- [ ] Model saves correctly

### Evaluation
- [ ] Test predictions complete
- [ ] Metrics calculated
- [ ] Confusion matrix generated
- [ ] Results documented

### Predictions
- [ ] Single transaction predictions work
- [ ] Probabilities are reasonable (0-100%)
- [ ] Threshold applied correctly

### Documentation
- [ ] README explains how to use
- [ ] Code has comments
- [ ] All scripts are functional

---

## Quick Reference: File Checklist

```
fraud-detection/
├── requirements.txt              ✅ Created
├── fraudTrain.csv               ✅ Moved
├── fraudTest.csv                ✅ Moved
│
├── models/                      ✅ Created
│   ├── fraud_model.keras       ⬜ After training
│   ├── preprocessors.pkl        ⬜ After training
│   └── optimal_threshold.txt  ⬜ After training
│
├── evaluation/                  ✅ Created
│   ├── confusion_matrix.png     ⬜ After evaluation
│   └── training_history.json    ⬜ After training
│
├── templates/                   ✅ Created
│   └── index.html              ✅ Created
│
├── explore_data.py              ⬜ Task 2.1
├── explore_columns.py           ⬜ Task 2.2
├── preprocess.py                 ⬜ Task 3.2
├── build_model.py               ⬜ Task 4.1
├── train_model.py               ⬜ Task 5.1
├── tune_model.py                ⬜ Task 6.1
├── evaluate_model.py            ⬜ Task 7.1
├── visualize_model.py           ⬜ Task 8.1
├── predict.py                   ⬜ Task 9.1
└── app.py                      ⬜ Task 10.1
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: pandas` | Activate virtual environment: `source venv/bin/activate` |
| Training too slow | Reduce batch size or use GPU |
| Out of memory | Reduce model size or use smaller batch size |
| Model won't load | Check file path and file exists |
| Predictions all same | Check threshold, may need retraining |
| Web app won't start | Check port 5000 is not in use |

---

## Next Steps After Completion

1. **Experiment**: Try different architectures
2. **Optimize**: Fine-tune hyperparameters
3. **Deploy**: Put model into production
4. **Monitor**: Track model performance over time
5. **Improve**: Collect more data, retrain periodically

---

**Congratulations!** You've built a complete fraud detection system with neural networks.
