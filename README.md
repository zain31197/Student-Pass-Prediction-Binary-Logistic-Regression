# Student Pass Prediction — Binary Logistic Regression

> A supervised machine learning pipeline that predicts academic outcomes using behavioral and performance features. Built with scikit-learn, pandas, and seaborn.

---

## Overview

This project builds a binary classification model to determine whether a student is likely to **pass or fail** based on five academic indicators. The pipeline covers everything from raw data ingestion to probability-based decision thresholding — making it a complete end-to-end ML workflow.

**Tech Stack:** Python · pandas · NumPy · scikit-learn · matplotlib · seaborn

---

## Dataset

| Feature | Description | Range |
|---|---|---|
| `Study_Hours` | Average daily study hours | Continuous |
| `Attendance` | Attendance percentage | 0 – 100 |
| `Assignment` | Assignment average | 0 – 10 |
| `Mid` | Midterm exam score | 0 – 30 |
| `Lab` | Lab performance score | 0 – 10 |
| `Pass` *(target)* | Outcome label | 0 = Fail, 1 = Pass |

**Dataset size:** 160 records · 6 columns · No missing values

---

## Project Structure

```
├── 23i2582_ZainShahid_DS6A_lab10.ipynb     # Main notebook
├── Binary_Logistic_Regression_...xlsx      # Source dataset
└── README.md
```

---

## Pipeline Breakdown

### 1. Data Loading & Inspection
- Load dataset with `pandas.read_excel()`
- Inspect shape, column names, data types, and null values
- Confirm dataset integrity before processing

### 2. Feature / Target Separation
- **X (features):** `Study_Hours`, `Attendance`, `Assignment`, `Mid`, `Lab`
- **y (target):** `Pass`

### 3. Train-Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility

### 4. Model Training
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```
After training, model coefficients are extracted to rank feature importance.

### 5. Prediction & Evaluation

| Metric | Result |
|---|---|
| Accuracy | High (see notebook output) |
| Precision | Evaluated per class |
| Recall | Evaluated per class |
| Confusion Matrix | TP / TN / FP / FN breakdown |

### 6. New Student Inference

Three unseen students are passed through the trained model:

| Student | Study | Attendance | Assignment | Mid | Lab | Prediction |
|---|---|---|---|---|---|---|
| S1 | 4 | 70 | 6 | 15 | 7 | See notebook |
| S2 | 2 | 55 | 4 | 10 | 5 | See notebook |
| S3 | 7 | 85 | 8 | 22 | 9 | See notebook |

Each prediction includes:
- Binary label (`PASS` / `FAIL`)
- Probability scores for both classes
- Risk interpretation (`High chance`, `Borderline`, `At risk`)

### 7. Feature Importance

Logistic regression coefficients reveal which features drive the prediction most. A bar chart visualizes positive (pass-driving) vs negative (fail-driving) coefficients.

### 8. Decision Threshold Analysis

| Threshold | Effect |
|---|---|
| `0.5` (default) | Balanced precision/recall |
| `0.6` (strict) | Fewer false positives · More false negatives · Accuracy may shift |

Raising the threshold makes the model more conservative — it only predicts PASS when it's fairly confident.

---

## Logistic Regression — How It Works

The model outputs a probability using the sigmoid function:

```
P(Y=1) = 1 / (1 + e^(-z))
where z = b0 + b1·Study_Hours + b2·Attendance + ... + b5·Lab
```

- Output is always between **0 and 1**
- Default decision boundary: `P > 0.5 → PASS`

---

## Getting Started

```bash
# Clone or download the repo
# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl

# Launch the notebook
jupyter notebook 23i2582_ZainShahid_DS6A_lab10.ipynb
```

Make sure `Binary_Logistic_Regression_Student_Dataset_160.xlsx` is in the **same directory** as the notebook.

---

## Key Findings

- **Most influential feature:** Mid-exam score (`Mid`) carries the highest coefficient weight
- **Threshold sensitivity:** Moving from 0.5 → 0.6 reduces false positives but increases missed PASS predictions
- **Model reliability:** Performs well on held-out test data with strong accuracy and precision

---

## Author

**Zain Shahid**  
*Data Science · Machine Learning · Predictive Modeling*
