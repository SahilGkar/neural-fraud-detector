import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense,
    Input,
    BatchNormalization,
    Dropout,
    Activation,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def preprocess_for_nn(df, encoders=None, scaler=None, fit=True):
    df = df.copy()

    df["trans_datetime"] = pd.to_datetime(df["trans_date_trans_time"])
    df["hour"] = df["trans_datetime"].dt.hour
    df["day_of_week"] = df["trans_datetime"].dt.dayofweek
    df["month"] = df["trans_datetime"].dt.month
    df["day_of_month"] = df["trans_datetime"].dt.day

    df["distance_km"] = haversine_distance(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )

    df["dob_datetime"] = pd.to_datetime(df["dob"])
    df["age"] = (df["trans_datetime"] - df["dob_datetime"]).dt.days // 365

    drop_cols = [
        "merchant",
        "job",
        "first",
        "last",
        "street",
        "city",
        "trans_num",
        "cc_num",
        "zip",
        "unix_time",
        "trans_date_trans_time",
        "trans_datetime",
        "dob",
        "dob_datetime",
        "Unnamed: 0",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    categorical_cols = ["category", "gender", "state"]

    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if col in df.columns:
            if fit:
                encoders[col] = LabelEncoder()
                df[col] = encoders[col].fit_transform(df[col].astype(str))
            else:
                le = encoders[col]
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)
                )

    if "is_fraud" in df.columns:
        y = df["is_fraud"].values
        df = df.drop(columns=["is_fraud"])
    else:
        y = None

    feature_cols = [
        c for c in df.columns if df[c].dtype in ["int64", "float64", "int32", "float32"]
    ]
    X = df[feature_cols].values

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, y, encoders, scaler, feature_cols


def build_model(
    input_dim, hidden_layers=[256, 128, 64], dropout_rate=0.3, learning_rate=0.001
):
    model = Sequential()

    model.add(Input(shape=(input_dim,)))

    for i, neurons in enumerate(hidden_layers):
        model.add(Dense(neurons))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Activation("relu"))

    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    return model


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    hidden_layers=[256, 128, 64],
    dropout_rate=0.3,
    learning_rate=0.001,
    batch_size=2048,
    epochs=50,
    class_weight=None,
    model_path="fraud_model.keras",
):
    if class_weight is None:
        n_neg = np.sum(y_train == 0)
        n_pos = np.sum(y_train == 1)
        weight_pos = n_neg / n_pos
        class_weight = {0: 1.0, 1: weight_pos}
        print(f"Using computed class weights: {class_weight}")
    elif class_weight == "balanced":
        from sklearn.utils.class_weight import compute_class_weight

        classes = np.unique(y_train)
        cw = compute_class_weight("balanced", classes=classes, y=y_train)
        class_weight = dict(zip(classes, cw))
        print(f"Using balanced class weights: {class_weight}")

    model = build_model(
        input_dim=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    has_validation = X_val is not None and y_val is not None
    monitor_metric = "val_loss" if has_validation else "loss"

    callbacks = [
        ReduceLROnPlateau(
            monitor=monitor_metric, factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
        EarlyStopping(
            monitor=monitor_metric, patience=5, restore_best_weights=True, verbose=1
        ),
        ModelCheckpoint(
            model_path,
            monitor="val_auc" if has_validation else "auc",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
    ]

    print("\n" + "=" * 60)
    print("TRAINING FRAUD DETECTION MODEL")
    print("=" * 60)
    print(f"Training samples: {len(X_train)}")
    print(f"Fraud cases: {np.sum(y_train == 1)} ({100 * np.mean(y_train):.2f}%)")
    if X_val is not None:
        print(f"Validation samples: {len(X_val)}")
    print(f"Hidden layers: {hidden_layers}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print("=" * 60 + "\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val) if has_validation else None,
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def evaluate_model(model, X_test, y_test, threshold=0.5):
    y_pred_prob = model.predict(X_test, batch_size=4096, verbose=0).flatten()
    y_pred = (y_pred_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    roc_auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc = average_precision_score(y_test, y_pred_prob)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")

    print(f"\nThreshold: {threshold}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    print(f"Precision (Fraud): {precision:.4f}")
    print(f"Recall (Fraud): {recall:.4f}")
    print(f"F1-Score (Fraud): {f1:.4f}")
    print("=" * 60 + "\n")

    return {
        "confusion_matrix": cm,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "y_pred_prob": y_pred_prob,
        "y_pred": y_pred,
    }


def predict(model, X, threshold=0.5):
    X = np.array(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    probabilities = model.predict(X, batch_size=4096, verbose=0).flatten()
    predictions = (probabilities >= threshold).astype(int)

    return predictions, probabilities


def main():
    print("=" * 60)
    print("CREDIT CARD FRAUD DETECTION - NEURAL NETWORK")
    print("=" * 60 + "\n")

    data_dir = "."
    train_path = os.path.join(data_dir, "fraudTrain.csv")
    test_path = os.path.join(data_dir, "fraudTest.csv")
    model_path = "fraud_model.keras"
    preprocessor_path = "preprocessor.pkl"

    print("Loading training data...")
    df_train = pd.read_csv(train_path)
    print(f"Training data: {len(df_train):,} rows")
    print(f"Fraud rate: {100 * df_train['is_fraud'].mean():.2f}%\n")

    print("Loading test data...")
    df_test = pd.read_csv(test_path)
    print(f"Test data: {len(df_test):,} rows")
    print(f"Fraud rate: {100 * df_test['is_fraud'].mean():.2f}%\n")

    print("Preprocessing training data...")
    X_train, y_train, encoders, scaler, feature_cols = preprocess_for_nn(
        df_train, fit=True
    )
    print(f"Training features shape: {X_train.shape}")
    print(f"Feature columns: {feature_cols}\n")

    print("Preprocessing test data...")
    X_test, y_test, _, _, _ = preprocess_for_nn(
        df_test, encoders=encoders, scaler=scaler, fit=False
    )
    print(f"Test features shape: {X_test.shape}\n")

    preprocessor = {
        "encoders": encoders,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }
    joblib.dump(preprocessor, preprocessor_path)
    print(f"Saved preprocessor to {preprocessor_path}\n")

    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    print(f"Training split: {len(X_tr):,} samples")
    print(f"Validation split: {len(X_val):,} samples\n")

    model, history = train_model(
        X_tr,
        y_tr,
        X_val,
        y_val,
        hidden_layers=[256, 128, 64],
        dropout_rate=0.3,
        learning_rate=0.001,
        batch_size=2048,
        epochs=30,
        class_weight="balanced",
        model_path=model_path,
    )

    model.save(model_path)
    print(f"\nModel saved to {model_path}\n")

    results = evaluate_model(model, X_test, y_test, threshold=0.5)

    print("\n[OK] Training complete!")
    print(f"  - Model: {model_path}")
    print(f"  - Preprocessor: {preprocessor_path}")
    print(f"  - ROC-AUC: {results['roc_auc']:.4f}")
    print(f"  - PR-AUC: {results['pr_auc']:.4f}")
    print(f"  - F1-Score: {results['f1']:.4f}")

    return model, results


if __name__ == "__main__":
    model, results = main()
