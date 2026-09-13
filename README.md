<!-- anchor tag for back-to-top links -->
<a name="readme-top"></a>

<p align="center">
  <img src="https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/header-image.webp" alt="ML Credit Risk Engine" width="900">
</p>

# ML Credit Risk Engine

An end-to-end machine learning project for loan-default prediction and credit-risk assessment.

The workflow covers data preprocessing, feature engineering, exploratory analysis, model comparison, hyperparameter tuning, threshold optimization, evaluation, and reusable inference.

---

## Table of Contents

- [Summary](#summary)
- [Motivation](#motivation)
- [Data](#data)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Modeling](#modeling)
- [Final Model](#final-model)
- [Deployment](#deployment)
- [Testing](#testing)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Responsible Use](#responsible-use)
- [Project Structure](#project-structure)
- [Appendix](#appendix)
- [License](#license)

---

## Summary

Built an end-to-end machine learning solution for predicting loan defaults from applicant data.

The project includes:

- **Data preprocessing:** cleaning, duplicate checks, missing-value analysis, outlier analysis, scaling, categorical encoding, and train/validation/test splitting.
- **Feature engineering:** job-stability information, city-tier information, and state-level default-rate information.
- **Exploratory analysis:** numerical distributions, categorical frequencies, correlations, and numerical/categorical relationships.
- **Modeling:** eight baseline classifiers followed by hyperparameter tuning and decision-threshold optimization.
- **Evaluation:** AUC-PR, precision, recall, F1-score, accuracy, precision-recall curves, confusion matrix, overfitting diagnostics, and feature importance.
- **Inference:** reusable preprocessing + model pipeline suitable for application/API integration.

### Built With

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Computing-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-017CEE?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?style=flat-square&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

---

## Motivation

Loan-default prediction is an imbalanced binary-classification problem. Accuracy alone can obscure poor detection of default cases, so this project emphasizes precision-recall behavior and threshold selection.

### Objectives

- Predict loan default risk from applicant-level data.
- Compare multiple classification algorithms using PR-AUC and class-specific metrics.
- Identify variables associated with default risk.
- Tune model hyperparameters using cross-validation.
- Optimize the decision threshold using validation data.
- Evaluate generalization on an untouched hold-out test set.
- Package preprocessing and model inference together.

---

## Data

The project uses the public **Loan Prediction Based on Customer Behavior** dataset.

- **Records:** 252,000
- **Target:** `Risk Flag`
- **Default rate:** approximately 12.3%
- **Input variables:** 11

| Group | Variables |
|---|---|
| Demographic | Age, Married/Single, Profession |
| Financial | Income, House Ownership, Car Ownership |
| Employment | Experience, Current Job Years |
| Housing | Current House Years |
| Geographic | City, State |
| Target | Risk Flag |

---

## Data Preprocessing

The preprocessing workflow is designed to keep transformations reproducible between training and inference.

- Standardized column names and labels.
- Checked duplicate identifiers and duplicate rows.
- Checked missing values and data types.
- Performed deterministic train/validation/test splitting.
- Engineered job-stability information.
- Derived city-tier information.
- Derived state-level historical default-rate information.
- Analysed univariate and multivariate outliers.
- Scaled numerical variables using `StandardScaler`.
- Encoded categorical variables using scikit-learn encoders.
- Applied transformations through reusable pipeline components.

---

## Exploratory Data Analysis

### Numerical distributions

![Numerical Distributions](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/numerical_distributions_histograms.png)

### Categorical frequencies

![Categorical Frequencies](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/categorical_frequencies_barplots.png)

### Numerical-numerical relationships

![Numerical Relationships](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/numerical_relationships_scatterplots.png)

### Numerical-categorical relationships

![Numerical-Categorical Relationships](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/numerical_categorical_relationships_barplots.png)

### Categorical-categorical relationships

![Categorical-Categorical Relationships](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/categorical_relationships_groupedbarplots.png)

### Correlation heatmap

<img src="https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/correlation_heatmap.png" alt="Correlation Heatmap" width="850">

---

## Modeling

Eight baseline classifiers were evaluated:

1. Logistic Regression
2. Elastic Net
3. K-Nearest Neighbors
4. Support Vector Machine
5. Neural Network
6. Decision Tree
7. Random Forest
8. XGBoost

### Evaluation metrics

The primary metric is **Area Under the Precision-Recall Curve (AUC-PR)** because the default class is imbalanced.

Secondary metrics include class-1 recall, precision, F1-score, accuracy, and confusion-matrix analysis.

### Baseline comparison

![AUC-PR Comparison](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/aucpr_comparison_baseline.png)

![Baseline Precision-Recall Curves](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/precision_recall_curves_baseline.png)

The strongest candidates were Random Forest, XGBoost, KNN, and Decision Tree.

### Hyperparameter tuning

Performed randomized hyperparameter search with 5-fold cross-validation.

![Tuned Precision-Recall Curves](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/precision_recall_curves_tuned.png)

### Threshold optimization

The operating threshold was optimized on validation data to balance precision and recall.

![Random Forest Threshold Metrics](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/rf_metrics_by_threshold_tuned.png)

![XGBoost Threshold Metrics](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/xgb_metrics_by_threshold_tuned.png)

![Decision Tree Threshold Metrics](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/tree_metrics_by_threshold_tuned.png)

![KNN Threshold Metrics](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/knn_metrics_by_threshold_tuned.png)

### Overfitting diagnostics

![Overfitting Analysis](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/overfitting_tuned_thresholds.png)

---

## Final Model

**Random Forest with an optimized decision threshold** was selected based on predictive performance, thresholded precision/recall behavior, overfitting characteristics, and interpretability.

### Hyperparameters

```text
n_estimators       = 225
max_depth          = 26
min_samples_split  = 2
min_samples_leaf   = 1
max_features       = 0.13
class_weight       = "balanced"
```

### Training, validation, and test performance

| Data | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Training | 0.68 | 1.00 | 0.62 | 0.77 | 0.93 |
| Validation | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| Test | 0.59 | 0.79 | 0.51 | 0.62 | 0.88 |

### Test classification report

| Class | Precision | Recall | F1-Score | Samples |
| :--- | :--- | :--- | :--- | :--- |
| Non-Defaulter | 0.97 | 0.90 | 0.93 | 22,122 |
| Defaulter | 0.51 | 0.79 | 0.62 | 3,078 |
| Macro Avg | 0.74 | 0.84 | 0.78 | 25,200 |
| Weighted Avg | 0.91 | 0.88 | 0.89 | 25,200 |

<img src="https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/rf_confusion_matrix_test.png" alt="Final Random Forest Confusion Matrix" width="500">

### Feature importance

The strongest predictors include income, age, and state default rate.

![Random Forest Feature Importance](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/rf_feature_importance_final.png)

### Thresholded model comparison

| Model | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| K-Nearest Neighbors | 0.59 | 0.81 | 0.52 | 0.63 | 0.88 |
| Decision Tree | 0.52 | 0.81 | 0.49 | 0.61 | 0.87 |
| Random Forest | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| XGBoost | 0.61 | 0.80 | 0.54 | 0.64 | 0.89 |

---

## Deployment

The model is structured as an end-to-end inference pipeline so preprocessing and model inference remain coupled.

```text
Applicant input
      |
      v
Validation
      |
      v
Feature engineering
      |
      v
Encoding + scaling
      |
      v
Random Forest
      |
      v
Default probability
      |
      v
Optimized threshold
      |
      +----> Default / No Default
```

---

## Testing

Run the test suite with:

```bash
pytest
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- Docker Desktop (optional)

### Setup

```bash
git clone https://github.com/anjelli/ML-Credit-Risk-Engine.git
cd ML-Credit-Risk-Engine
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Training

```bash
python scripts/train_model.py --data data/training_data.csv
```

### Testing

```bash
pytest
```

---

## Usage

### Local inference

```python
import joblib
import pandas as pd

pipeline = joblib.load("artifacts/credit_risk_pipeline.joblib")

applicant_data = pd.DataFrame({
    "income": [300000],
    "age": [30],
    "experience": [3],
    "married": ["single"],
    "house_ownership": ["rented"],
    "car_ownership": ["no"],
    "profession": ["Artist"],
    "city": ["Sikar"],
    "state": ["Rajasthan"],
    "current_job_yrs": [3],
    "current_house_yrs": [11],
})

probability = pipeline.predict_proba(applicant_data)[:, 1]
print(f"Probability of default: {probability[0]:.3f}")
```

---

## Responsible Use

This project is intended for credit-risk decision support rather than autonomous lending.

Predictions should not be the sole basis for financial decisions. Production use would require additional validation, calibration, fairness analysis, drift monitoring, governance, explainability, and compliance review.

---

## Project Structure

```text
ML-Credit-Risk-Engine/
├── .github/workflows/
├── images/
├── scripts/
│   └── train_model.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── modeling.py
│   └── plots.py
├── tests/
│   └── test_modeling.py
├── loan_default_prediction.ipynb
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── requirements-notebook.txt
├── requirements-test.txt
├── pytest.ini
├── start.sh
├── .gitignore
└── LICENSE
```

---

## Appendix

### Descriptive statistics

| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Income | 201,600 | 5,000,741 | 2,880,711 | 10,310 | 2,499,018 | 5,004,535 | 7,489,827 | 9,999,938 |
| Age | 201,600 | 50.01 | 17.08 | 21 | 35 | 50 | 65 | 79 |
| Experience | 201,600 | 10.09 | 6.00 | 0 | 5 | 10 | 15 | 20 |
| Current Job Yrs | 201,600 | 6.33 | 3.65 | 0 | 3 | 6 | 9 | 14 |
| Current House Yrs | 201,600 | 12.00 | 1.40 | 10 | 11 | 12 | 13 | 14 |
| State Default Rate | 201,600 | 0.12 | 0.02 | 0.05 | 0.11 | 0.12 | 0.13 | 0.21 |

### Baseline models

| Model | AUC-PR | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.16 | 0.00 | 0.00 | 0.00 | 0.88 |
| Elastic Net | 0.16 | 0.00 | 0.00 | 0.00 | 0.88 |
| KNN | 0.53 | 0.50 | 0.56 | 0.53 | 0.89 |
| SVM | 0.13 | 0.28 | 0.13 | 0.18 | 0.67 |
| Neural Network | 0.32 | 0.10 | 0.51 | 0.17 | 0.88 |
| Decision Tree | 0.46 | 0.57 | 0.52 | 0.55 | 0.88 |
| Random Forest | 0.60 | 0.53 | 0.60 | 0.56 | 0.90 |
| XGBoost | 0.54 | 0.20 | 0.64 | 0.30 | 0.89 |

### Tuned models — optimized thresholds

| Model | AUC-PR | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| KNN | 0.59 | 0.81 | 0.52 | 0.63 | 0.88 |
| Decision Tree | 0.52 | 0.81 | 0.49 | 0.61 | 0.87 |
| Random Forest | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| XGBoost | 0.61 | 0.80 | 0.54 | 0.64 | 0.89 |

### Threshold diagnostics

![Random Forest Threshold Metrics](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/rf_metrics_by_threshold_tuned.png)

![XGBoost Threshold Metrics](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/xgb_metrics_by_threshold_tuned.png)

![Decision Tree Threshold Metrics](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/tree_metrics_by_threshold_tuned.png)

![KNN Threshold Metrics](https://raw.githubusercontent.com/JensBender/loan-default-prediction/main/images/knn_metrics_by_threshold_tuned.png)

---

## License

MIT License.
