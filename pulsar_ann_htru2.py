# 1. IMPORT LIBRARIES & SET SEEDS ///
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from IPython.display import Image

# Add imports for loading the zip file
import io
import zipfile
import urllib.request
import os
import random

# NEW: Add Reproducibility Seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("--- Libraries Imported and Seeds Set ---")

# 2. LOAD AND PREPROCESS DATA////

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00372/HTRU2.zip"
column_names = [
    "profile_mean", "profile_std", "profile_kurtosis", "profile_skewness",
    "dm_mean", "dm_std", "dm_kurtosis", "dm_skewness", "target_class"
]

try:
    with urllib.request.urlopen(data_url) as response:
        zip_content = io.BytesIO(response.read())
    with zipfile.ZipFile(zip_content) as zf:
        with zf.open('HTRU_2.csv') as csv_file:
            df = pd.read_csv(csv_file, names=column_names)
    print(f"Dataset loaded. Total shape: {df.shape}")
except Exception as e:
    print(f"Error loading dataset: {e}")

print("\nClass Distribution:")
print(df['target_class'].value_counts())

# /// NEW - Data Exploration Visuals (
# Class Imbalance Bar Chart
plt.figure(figsize=(6, 4))
sns.countplot(x='target_class', data=df, hue='target_class', palette='viridis', legend=False)
plt.title('Class Distribution: Non-Pulsar (0) vs Pulsar (1)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Non-Pulsar', 'Pulsar'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Feature Correlation Matrix')
plt.show()
# ---

# ---
# Data Preparation for Models

X = df.drop('target_class', axis=1)
y = df['target_class']

# Stratified split to maintain class ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

# Standard Scaling (Critical for ANNs)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n--- Data Preprocessing Complete ---")


# 3. MODEL 1: ANN BASELINE  ///////
print("\n--- Training Model 1: ANN Baseline (Simple MLP) ---")

model_baseline = Sequential([
    Dense(8, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(1, activation='sigmoid')
])

model_baseline.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_baseline = model_baseline.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)

y_pred_prob_baseline = model_baseline.predict(X_test)
y_pred_baseline = (y_pred_prob_baseline > 0.5).astype(int)
print("--- Model 1 Trained ---")


# 4. MODEL 2: NAIVE MLP ////
print("\n--- Training Model 2: 'Naive' MLP (Deeper) ---")

model_naive = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_naive.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history_naive = model_naive.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)

y_pred_prob_naive = model_naive.predict(X_test)
y_pred_naive = (y_pred_prob_naive > 0.5).astype(int)
print("--- Model 2 Trained ---")


# 5. MODEL 3: OPTIMIZED MLP (WITH CLASS WEIGHTING)//
print("\n--- Training Model 3: 'Optimized' MLP (with Class Weighting & Dropout) ---")

# Calculate Class Weights to handle imbalance
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: weights[0], 1: weights[1]}
print(f"Calculated Class Weights: {class_weight_dict}")

model_optimized = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model_optimized.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# --- NEW Vis--
print("\nGenerating Model Architecture Diagram...")
plot_model(model_optimized,
           to_file='model_optimized_architecture.png',
           show_shapes=True,
           show_layer_names=True,
           rankdir='TB',
           dpi=96)
from IPython.display import Image
display(Image('model_optimized_architecture.png'))
#  NEW VIS CODE END ---

history_optimized = model_optimized.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    class_weight=class_weight_dict,
    verbose=1
)

y_pred_prob_optimized = model_optimized.predict(X_test)
y_pred_optimized = (y_pred_prob_optimized > 0.5).astype(int)
print("--- Model 3 Trained ---")

# 6. FINAL RESULTS AND COMPARISON///
print("\n" + "="*50)
print("           FINAL MODEL COMPARISON")
print("="*50 + "\n")

target_names = ['Class 0 (Non-Pulsar)', 'Class 1 (Pulsar)']
print("--- Model 1: ANN Baseline ---")
print(classification_report(y_test, y_pred_baseline, target_names=target_names))

print("\n--- Model 2: 'Naive' MLP (Deeper) ---")
print(classification_report(y_test, y_pred_naive, target_names=target_names))

print("\n--- Model 3: 'Optimized' MLP (with Class Weighting) ---")
print(classification_report(y_test, y_pred_optimized, target_names=target_names))

# Plot Confusion Matrices
models = {
    "ANN Baseline (Model 1)": y_pred_baseline,
    "'Naive' MLP (Model 2)": y_pred_naive,
    "'Optimized' MLP (Model 3)": y_pred_optimized
}

plt.figure(figsize=(20, 5))
for i, (name, y_pred) in enumerate(models.items()):
    plt.subplot(1, 3, i+1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

plt.tight_layout()
plt.show()

# Plot Training History
plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(history_baseline.history['loss'], label='Baseline Train Loss')
plt.plot(history_baseline.history['val_loss'], label='Baseline Val Loss', linestyle='--')
plt.plot(history_naive.history['loss'], label='Naive MLP Train Loss')
plt.plot(history_naive.history['val_loss'], label='Naive MLP Val Loss', linestyle='--')
plt.plot(history_optimized.history['loss'], label='Optimized MLP Train Loss')
plt.plot(history_optimized.history['val_loss'], label='Optimized MLP Val Loss', linestyle='--')
plt.title('Model Loss Comparison')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_baseline.history['accuracy'], label='Baseline Train Accuracy')
plt.plot(history_baseline.history['val_accuracy'], label='Baseline Val Accuracy', linestyle='--')
plt.plot(history_naive.history['accuracy'], label='Naive MLP Train Accuracy')
plt.plot(history_naive.history['val_accuracy'], label='Naive MLP Val Accuracy', linestyle='--')
plt.plot(history_optimized.history['accuracy'], label='Optimized MLP Train Accuracy')
plt.plot(history_optimized.history['val_accuracy'], label='Optimized MLP Val Accuracy', linestyle='--')
plt.title('Model Accuracy Comparison')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()