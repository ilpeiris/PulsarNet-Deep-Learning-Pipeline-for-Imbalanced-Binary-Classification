# 🌌 Pulsar Candidate Classification using Deep Learning (HTRU2 Dataset)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![Status](https://img.shields.io/badge/Status-Completed-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

This repository contains a full end-to-end **Deep Learning pipeline** for detecting *pulsar candidates* in the **HTRU2 radio astronomy dataset**.  
The project focuses on maximizing **Recall** for the minority pulsar class, where missing a true pulsar harms scientific discovery.

A tapered Multilayer Perceptron (MLP) with **Class Weighting** and **Dropout** is used to counter the extreme **9.9:1 imbalance** between noise and real pulsars.

---

## 📌 Project Overview

Radio telescopes produce millions of candidate signals per survey — but very few are real pulsars. Most "candidates" are just:

- Noise  
- Radio Frequency Interference (RFI)  
- Non-astrophysical artifacts  

A naive classifier can achieve **98% accuracy** simply by predicting *"not pulsar"* for every input.  
This project avoids that **Accuracy Trap** and focuses on correctly identifying real pulsars.

---

## 📉 Dataset Summary (HTRU2)

| Class | Count | Description                     |
|-------|-------|---------------------------------|
| 0     | 16,259| Non-pulsar (Noise / RFI)        |
| 1     | 1,639 | True Pulsar                     |

- **Total samples:** 17,898  
- **Imbalance ratio:** ~9.9 : 1

### 📊 Class Imbalance (Figure 1)

![Class Imbalance](figures/Class%20Distribution.png)

---

## 🧠 Model Architectures

All models were trained using **TensorFlow / Keras** on standardized numerical features.

---

### 1️⃣ Baseline MLP

A simple model used as a performance reference.

- `Dense(8, ReLU)`  
- `Dense(1, Sigmoid)`

---

### 2️⃣ Naive Deep MLP

Tests whether depth alone improves results.

- `Dense(32, ReLU)`  
- `Dense(16, ReLU)`  
- `Dense(1, Sigmoid)`

---

### 3️⃣ ⭐ Optimized MLP (Final Model)

Designed for scientific discovery tasks where **Recall matters most**.

- `Dense(64, ReLU)`  
- `Dropout(0.3)`  
- `Dense(32, ReLU)`  
- `Dropout(0.3)`  
- `Dense(1, Sigmoid)`

Key techniques:

- **Class Weighting** → reduces false negatives on the minority pulsar class  
- **Dropout** → combats overfitting on noisy data  
- **Stratified splitting + scaling** → stable training

#### 🔧 Architecture Diagram (Figure 2)

![Model Architecture](figures/Model%20Architecture%20Diagram.png)

---

## 📊 Exploratory Data Analysis (EDA)

### 🔬 Correlation Matrix (Figure 3)

![Feature Correlation Matrix](figures/Feature%20Correlation%20Matrix.png)

### 📉 Loss & Accuracy Curves (Figure 4)

![Loss and Accuracy](figures/Model%20Loss-Accuracy.png)

### 🧩 Confusion Matrices (Figure 5)

![Confusion Matrices](figures/Confusion%20Matrices.png)

---

## 🏆 Results

| Metric               | Baseline MLP | Optimized MLP |
|----------------------|-------------:|--------------:|
| **Accuracy**         | 0.98         | 0.97          |
| **Recall (Pulsar)**  | 0.84         | **0.91**      |
| **Precision (Pulsar)**| 0.95        | 0.78          |
| **F1-score**         | 0.89         | 0.84          |
| **False Negatives**  | 53           | **29**        |

### 🌟 Impact

The optimized model correctly identifies **24 more true pulsars** compared to the baseline — a meaningful improvement in any astronomical survey pipeline, where missed detections are more costly than extra false positives.

---

## 🎯 Threshold Tuning

Default decision threshold: `0.50`  
Tuned decision threshold: `0.35` → increases Recall at the cost of more false positives.

| Threshold | Recall (Pulsar) | False Positives |
|-----------|-----------------|-----------------|
| 0.50      | 0.91            | Moderate        |
| 0.35      | **0.92**        | Higher          |

In scientific discovery tasks, this trade-off is often acceptable: **it is better to review extra candidates than to miss a real pulsar**.

---

## 🚀 How to Run

### 1️⃣ Clone the repository

```bash
git clone https://github.com/ilpeiris/Pulsar-Classification-Artificial-Neural-Networks.git
cd Pulsar-Classification-Artificial-Neural-Networks
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the script

```bash
python pulsar_classification.py
```

---

## 📁 Project Structure

```
Pulsar-Classification-Artificial-Neural-Networks/
├── data/
│   └── HTRU_2.csv
├── figures/
│   ├── class_distribution.png
│   ├── model_architecture.png
│   ├── feature_correlation_matrix.png
│   ├── Model_Loss-Accuracy.png
│   └── confusion_matrices.png
├── pulsar_classification.py
├── requirements.txt
└── README.md
```

---

## 🔮 Future Improvements

- Experiment with ensemble methods (Random Forest, XGBoost)
- Apply SMOTE or other resampling techniques
- Explore CNN-based models for time-series pulsar profiles
- Deploy as a web API for real-time classification

---

## 📚 References

- R. J. Lyon et al. (2016). "Fifty Years of Pulsar Candidate Selection: From simple filters to a new principled real-time classification approach"
- HTRU2 Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/HTRU2)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ilpeiris/Pulsar-Classification-Artificial-Neural-Networks/issues).

---

## 👨‍💻 Author

**Isuru Peiris**

- GitHub: [@ilpeiris](https://github.com/ilpeiris)

---


