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

history_naive = model_naive.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
    verbose=1
)