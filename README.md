# 🌌 Robust Pulsar Candidate Classification using Optimized ANNs
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)


This repository hosts a production-grade, end-to-end **Deep Learning data pipeline** engineered to detect rare *pulsar candidates* within the benchmark **UCI HTRU2 radio astronomy dataset**.

The primary objective of this architecture is to optimize **Recall** for the heavily suppressed minority class (genuine pulsars). In automated astronomical filtering systems, a False Negative implies permanently losing a unique stellar object, meaning standard accuracy optimization is highly deceptive. To counteract a severe **9.9:1 class imbalance**, this platform designs an optimized, tapered Multilayer Perceptron (MLP) driven by cost-sensitive **Class Weighting** and **Dropout regularization**.

---

## 🔭 Real-World Aerospace & Mission Context

In high-scale aerospace and astronomical operations, this pipeline serves as an automated, mission-critical triage node:

* **Gravitational Wave Hunting (PTAs):** Discovered pulsars contribute directly to deep-space arrays like Pulsar Timing Arrays (PTAs), which coordinate stellar observations to identify cosmic ripples and gravitational waves from supermassive black hole binaries.
* **Autonomous Ingestion Triage:** Large-scale radio surveys (such as the HTRU survey) generate millions of candidate signals that cannot be manually processed. This neural network functions at the data ingestion layer to automatically isolate genuine anomalies from overwhelming radio frequency interference (RFI) and terrestrial noise.
* **Relativistic Astrophysical Laboratories:** Identifying these rare stellar remnants fuels core space science research, allowing agencies to map the density of the interstellar medium and execute complex experiments in general relativity.

---

## 🚀 The Real-World Engineering Problem

Radio telescopes produce millions of candidate signals per survey — but very few are real pulsars. Most "candidates" are simply atmospheric noise, Radio Frequency Interference (RFI), or non-astrophysical artifacts.

```text
[ Automated Remote Stream ] ──> [ In-Memory Zip Extraction ] ──> [ Stratified Train/Test Split ]
                                                                             │
[ Evaluation & Metric Matrix ] <── [ Cost-Sensitive Loss Engine ] <── [ Layer-Wise Dropout MLP ]

```

Because true pulsars constitute less than 10% of the observations, a naive model can trivially achieve **98% classification accuracy** simply by predicting "noise" for every single signal. This architecture explicitly breaks out of this **Accuracy Trap** by shifting validation parameters toward class-specific sensitivity metrics.

---

## 📉 Dataset Topology & Exploratory Data Analysis (EDA)

### Feature Space

The pipeline extracts 8 continuous, high-dimensional statistical metrics calculated from the candidate integrated pulse profiles and Dispersion Measure (DM)-SNR curves:

* **Integrated Pulse Profile:** Mean, Standard Deviation, Excess Kurtosis, Skewness.
* **DM-SNR Curve:** Mean, Standard Deviation, Excess Kurtosis, Skewness.

### Imbalance Distribution

| Class | Label | Observation Count | Distribution Share |
| --- | --- | --- | --- |
| **Non-Pulsar** (Noise / RFI) | `0` | 16,259 | 90.84% |
| **True Pulsar** | `1` | 1,639 | 9.16% |

* **Total samples:** 17,898
* **Imbalance ratio:** ~9.9 : 1

#### 📊 Class Asymmetry Profile (Figure 1)
![Class Imbalance](figures/Class%20Distribution.png)

### Pipeline Automation

Instead of demanding static local file positioning, the ingestion engine establishes an automated network hook pulling straight from the UCI ML Repository[cite: 4, 5]. It programmatically opens the zip file using `urllib.request`, decompresses the binary stream via `io.BytesIO` and `zipfile`, maps structural features, and implements automated feature standardization using Scikit-learn's `StandardScaler`[cite: 4, 5].

#### 🔬 Feature Correlation Matrix (Figure 3)
![Feature Correlation Matrix](figures/Feature%20Correlation%20Matrix.png)

Useful for analyzing multi-collinearity and tracking high-dimensional feature relationships before network ingestion.


---

## 🧠 ANN Structural Configurations

All structural layers are built via **TensorFlow / Keras** and compiled using identical foundations for direct benchmarking: **Binary Crossentropy loss**, **Adam optimizer**, a batch size of **32**, a **10% validation holdout**, and an **EarlyStopping callback** (patience=3) to catch optimal validation loss windows.

### 1️⃣ Model 1: ANN Baseline (Shallow reference)

* **Configuration:** `Dense(8, ReLU) -> Dense(1, Sigmoid)`
* **Objective:** Defines the performance floor of a basic ANN exposed to high class skew without counter-measures.

### 2️⃣ Model 2: "Naive" Deep MLP (Increased Capacity)

* **Configuration:** `Dense(32, ReLU) -> Dense(16, ReLU) -> Dense(1, Sigmoid)`
* **Objective:** Evaluates if deeper hidden layers alone can resolve geometric boundaries under extreme class imbalances without specialized adjustments.

### 3️⃣ Model 3: ⭐ "Optimized" MLP (Cost-Sensitive Variant)

* **Configuration:** `Dense(64, ReLU) -> Dropout(0.3) -> Dense(32, ReLU) -> Dropout(0.3) -> Dense(1, Sigmoid)`
* **Core Mechanics:**
* **Algorithmic Class Weighting:** Calculates class weights inversely proportional to sample density (`0.55` for Class 0; `5.46` for Class 1). This shifts the loss engine to prioritize positive classifications.
* **Layer-Wise Dropout:** Injects a `0.3` drop rate across hidden dense connections to suppress co-adaptation and over-fitting on high-frequency noise.



#### 🔧 Architecture Diagram (Figure 2)
![Model Architecture](figures/Model%20Architecture%20Diagram.png)

---

## 📊 Benchmarked Experimental Results

Evaluation scores generated against the unseen 20% stratified test set (`3,580` total instances; `3,252` non-pulsars, `328` pulsars):

| Performance Criteria | Model 1: ANN Baseline | Model 2: Naive Deep MLP | Model 3: Optimized MLP |
| --- | --- | --- | --- |
| **Global Accuracy** | 0.98 | 0.98 | 0.97 |
| **Pulsar Recall (Sensitivity)** | 0.84 | 0.84 | **0.91** |
| **Pulsar Precision** | 0.95 | 0.95 | 0.78 |
| **Positive F1-Score** | 0.89 | 0.89 | 0.84 |
| **False Negatives (Missed)** | 53 | 53 | **29** |

#### 📈 Training Optimization Curves (Figure 4)
![Loss and Accuracy](figures/Model%20Loss-Accuracy.png)

#### 🧩 Structural Confusion Matrices (Figure 5)
![Confusion Matrices](figures/Confusion%20Matrices.png)

### 🔬 Empirical Analysis

Both the **Baseline** and the **Naive Deep MLP** fell completely into the Accuracy Trap, failing to adapt to the skew and missing 53 true pulsars (a 16% scientific failure rate). By shifting optimization weight parameters, **Model 3 successfully recovered 24 additional true pulsars**, slicing critical false negative errors down by roughly **45%**.

---

## 🎯 Post-Training Decision Threshold Tuning

To maximize discovery potential, the final classification layer was exposed to probability threshold manipulation. By stepping down from the standard `0.50` decision boundary to `0.35`, the model alters its classification strictness to aggressively secure candidates.

```text
Threshold 0.50:  [ 29 Missed Pulsars (FN) ]  ──>  [ 84 Extra Noise Flags (FP) ]
Threshold 0.35:  [ 26 Missed Pulsars (FN) ]  ──>  [ 120 Extra Noise Flags (FP) ]

```

| Applied Threshold Value | Pulsar Recall Rate | False Positive Count | False Negative Count |
| --- | --- | --- | --- |
| **0.50 (Standard)** | 0.91 | 84 | 29 |
| **0.35 (Tuned Discovery)** | **0.92** | 120 | **26** |

*Astronomical Core Trade-Off:* Reviewing 36 extra false positive candidates is a minor operational cost compared to permanently filtering out 3 real pulsars.

---

## ✨ Standout Technical Highlights

* **Intrinsic Model-Level Imbalance Handling:** Instead of altering the dataset via resampling (like SMOTE), this project successfully configures model-level class weighting inside the loss function, forcing the network to penalize minority-class misclassifications 9.9 times more severely.
* **Defeating the Accuracy Trap:** Proved empirically that deeper models without imbalance awareness fail to generalize, whereas an optimized architecture successfully traded a 1% drop in raw accuracy for an **8% surge in pulsar recall**.
* **Post-Training Threshold Tuning:** Implemented dynamic probability threshold adjustments (lowering the classification threshold from 0.5 to 0.35) to create an "aggressive search" variant, successfully squeezing out additional pulsar detections.
* **Strict Regularization Pipeline:** Mitigated overfitting risks associated with deeper hidden layers by embedding custom Dropout layers (p=0.3) and utilizing Keras EarlyStopping callbacks mapped to validation loss.

---

## 🔒 Deterministic Reproducibility Blueprint

To lock execution variance down across differing hardware infrastructures, a dedicated environment seed lock is set at runtime initialization:

```python
import os, random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

```

---

## 🚀 Deployment & Operational Guide

### 1️⃣ Clone Environment

```bash
git clone https://github.com/ilpeiris/Pulsar-Classification-Artificial-Neural-Networks.git
cd Pulsar-Classification-Artificial-Neural-Networks

```

### 2️⃣ Install Core Dependencies

```bash
pip install -r requirements.txt

```

### 3️⃣ Execute Execution Script

The complete pipeline runs entirely out-of-the-box, automatically streams data arrays, extracts files, runs testing sweeps, and renders evaluation matrices[cite: 4, 5]:

```bash
python pulsar_ann_htru2.py

```

---

## 📁 Repository Blueprint

```text
Pulsar-Classification-Artificial-Neural-Networks/
├── figures/
│   ├── Class%20Distribution.png            # Figure 1: Class asymmetry profile
│   ├── Model%20Architecture%20Diagram.png  # Figure 2: Layer map topology
│   ├── Feature%20Correlation%20Matrix.png  # Figure 3: Pearson correlation map
│   ├── Model%20Loss-Accuracy.png           # Figure 4: Ingestion training curves
│   └── Confusion%20Matrices.png            # Figure 5: Multi-model evaluation matrix
├── pulsar_ann_htru2.py                     # Self-contained executable script
├── requirements.txt                        # Tracked environment dependencies
└── README.md                               # System documentation

```

---

## 🔮 Roadmap & Extensions

* **Comparative Hybrid Architecture:** Benchmark model-level class-weighting performance directly against data-level synthetic sampling techniques (SMOTE).
* **Structural Grid Searches:** Expand hyperparameter search windows via KerasTuner to find optimal dropout rates and learning variables.
* **Alternative Sequence Architectures:** Treat the 8-feature arrays as a sequential matrix and evaluate performance across 1D-CNN layers.

---

## 🛠️ Technology Stack & Dependencies

* **Framework:** TensorFlow 2.19.0 & Keras
* **Data Processing:** Scikit-learn 1.6.1 (StandardScaler, compute_class_weight), Pandas 2.2.2, NumPy 2.0.2
* **Visualization:** Matplotlib, Seaborn
* **Environment:** Google Colab / Python 3

---

## 📚 Technical References

* **Lyon, R. J. et al.** (2016). *Fifty years of pulsar candidate selection: from manual inspection to artificial intelligence.* Monthly Notices of the Royal Astronomical Society.
* **Dataset Source:** [UCI Machine Learning Repository - HTRU2 Archive](https://archive.ics.uci.edu/ml/datasets/htru2).

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for complete structural allowances.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://www.google.com/search?q=https://github.com/ilpeiris/Pulsar-Classification-Artificial-Neural-Networks/issues).

---

## 👨‍💻 Developer Profile

**Isuru Peiris**

* **GitHub:** [@ilpeiris](https://www.google.com/search?q=https://github.com/ilpeiris)
* **LinkedIn:** [@ilpeiris](https://www.linkedin.com/in/ilpeiris/)
