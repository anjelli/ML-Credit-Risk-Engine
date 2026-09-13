<!-- anchor tag for back-to-top links -->
<a name="readme-top"></a>

<!-- HEADER -->
<p align="center">
  <img src="images/header-image.webp" alt="ML Credit Risk Engine" width="900">
</p>

# ML Credit Risk Engine

An end-to-end machine learning application for loan default prediction and credit risk assessment.

Engineered a complete workflow covering feature engineering, preprocessing, exploratory data analysis, model comparison, hyperparameter tuning, threshold optimization, evaluation, and deployable inference.

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

Built an end-to-end machine learning solution for predicting loan defaults using customer application data.

The project includes:

- **Data Preprocessing:** feature engineering, duplicate and missing-value checks, outlier analysis, numerical scaling, categorical encoding, and deterministic train/validation/test splitting.
- **Exploratory Data Analysis:** numerical distributions, categorical frequencies, correlations, and numerical/categorical relationship analysis.
- **Modeling:** eight baseline classifiers, followed by hyperparameter tuning and decision-threshold optimization.
- **Evaluation:** AUC-PR, class-1 precision, recall, F1-score, accuracy, precision-recall curves, confusion matrix, overfitting analysis, feature importance, and prediction examples.
- **Deployment:** reusable scikit-learn preprocessing/model pipeline with application and containerization support.

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

Predicting loan defaults is challenging because applicant risk depends on interacting demographic, financial, employment, behavioral, housing, and geographic attributes. Because defaults form the minority class, accuracy alone can hide poor detection of risky applicants.

### Objectives

- Develop a machine learning model for loan-default prediction.
- Compare multiple classification algorithms using a metric appropriate for class imbalance.
- Identify important variables associated with default risk.
- Tune model hyperparameters using cross-validation.
- Optimize the decision threshold against explicit precision/recall constraints.
- Evaluate the final model on an untouched hold-out test set.
- Package the preprocessing and model as a reusable inference pipeline.

---

## Data

The project uses the public **Loan Prediction Based on Customer Behavior** dataset.

Dataset statistics:

- **Records:** 252,000
- **Target:** `Risk Flag`
- **Default rate:** approximately 12.3%
- **Features:** 11 input variables

### Data overview

| Column | Description | Type |
| :--- | :--- | :--- |
| Risk Flag | Defaulted on loan (0: No, 1: Yes) | Binary |
| Income | Applicant income | Numerical |
| Age | Applicant age | Numerical |
| Experience | Work experience in years | Numerical |
| Profession | Applicant profession | Categorical |
| Married/Single | Marital status | Categorical |
| House_Ownership | Housing ownership | Categorical |
| Car_Ownership | Vehicle ownership | Categorical |
| CURRENT_JOB_YRS | Years in current job | Numerical |
| CURRENT_HOUSE_YRS | Years in current house | Numerical |
| City | City of residence | Categorical |
| State | State of residence | Categorical |

### Example records

| Risk Flag | Income | Age | Experience | Profession | Married | House Ownership | Car Ownership | Current Job Years | Current House Years | City | State |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 1,303,834 | 23 | 3 | Mechanical_engineer | single | rented | no | 3 | 13 | Rewa | Madhya_Pradesh |
| 1 | 6,256,451 | 41 | 2 | Software_Developer | single | rented | yes | 2 | 12 | Bangalore | Tamil_Nadu |
| 0 | 3,991,815 | 66 | 4 | Technical_writer | married | rented | no | 4 | 10 | Alappuzha | Kerala |

---

## Data Preprocessing

Used pandas and scikit-learn for loading, cleaning, transformation, feature engineering, and model preparation.

- Standardized column names and labels.
- Checked duplicate identifiers and complete duplicate rows.
- Checked missing values and data types.
- Performed train-validation-test splitting.
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

![Numerical Distributions](images/numerical_distributions_histograms.png)

### Categorical frequencies

![Categorical Frequencies](images/categorical_frequencies_barplots.png)

### Numerical-numerical relationships

![Numerical Relationships](images/numerical_relationships_scatterplots.png)

### Numerical-categorical relationships

![Numerical-Categorical Relationships](images/numerical_categorical_relationships_barplots.png)

### Categorical-categorical relationships

![Categorical-Categorical Relationships](images/categorical_relationships_groupedbarplots.png)

### Correlation heatmap

<img src="images/correlation_heatmap.png" alt="Correlation Heatmap" width="850">

---

## Modeling

Trained and evaluated eight baseline models using scikit-learn and XGBoost:

- Logistic Regression
- Elastic Net
- K-Nearest Neighbors
- Support Vector Machine
- Neural Network
- Decision Tree
- Random Forest
- XGBoost

### Evaluation metrics

The primary metric is **Area Under the Precision-Recall Curve (AUC-PR)** because the default class is imbalanced.

Secondary metrics:

- Class-1 recall
- Class-1 precision
- Class-1 F1-score
- Accuracy
- Confusion matrix

### Baseline models

![AUC-PR Comparison](images/aucpr_comparison_baseline.png)

![Baseline Precision-Recall Curves](images/precision_recall_curves_baseline.png)

The strongest baseline candidates were Random Forest, XGBoost, KNN, and Decision Tree.

### Hyperparameter tuning

Performed randomized hyperparameter search with 5-fold cross-validation and evaluated the best-performing candidates on validation data.

![Tuned Precision-Recall Curves](images/precision_recall_curves_tuned.png)

### Threshold optimization

Optimized decision thresholds to balance precision and recall according to the project's risk-detection objectives.

![Random Forest Threshold Metrics](images/rf_metrics_by_threshold_tuned.png)

![XGBoost Threshold Metrics](images/xgb_metrics_by_threshold_tuned.png)

![Decision Tree Threshold Metrics](images/tree_metrics_by_threshold_tuned.png)

![KNN Threshold Metrics](images/knn_metrics_by_threshold_tuned.png)

### Overfitting diagnostics

![Overfitting Analysis](images/overfitting_tuned_thresholds.png)

---

## Final Model

**Random Forest with an optimized decision threshold** was selected based on predictive performance, thresholded F1/recall/precision behavior, overfitting characteristics, and interpretability.

### Hyperparameters

```text
n_estimators      = 225
max_depth         = 26
min_samples_split = 2
min_samples_leaf  = 1
max_features      = 0.13
class_weight      = "balanced"
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

<img src="images/rf_confusion_matrix_test.png" alt="Final Random Forest Confusion Matrix" width="500">

### Feature importance

The most influential features include income, age, and state default rate, with employment and demographic variables providing additional signal.

![Random Forest Feature Importance](images/rf_feature_importance_final.png)

### Thresholded model comparison

| Model | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| K-Nearest Neighbors | 0.59 | 0.81 | 0.52 | 0.63 | 0.88 |
| Decision Tree | 0.52 | 0.81 | 0.49 | 0.61 | 0.87 |
| Random Forest | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| XGBoost | 0.61 | 0.80 | 0.54 | 0.64 | 0.89 |

---

## Deployment

The model is structured as an end-to-end pipeline so preprocessing and model inference remain coupled.

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

The repository includes Docker and application scaffolding for serving the trained model through an application/API layer.

---

## Testing

Unit tests cover the model-selection and evaluation utilities.

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

The model is intended as a credit-risk decision-support system rather than an autonomous lending mechanism.

Predictions should not be the sole basis for financial decisions. Production use would require additional validation, calibration, fairness analysis, monitoring, governance, explainability, and compliance review.

---

## Project Structure

```text
ML-Credit-Risk-Engine/
├── .github/
│   └── workflows/
│       └── quality.yml
├── images/
│   ├── aucpr_comparison_baseline.png
│   ├── categorical_frequencies_barplots.png
│   ├── categorical_relationships_groupedbarplots.png
│   ├── correlation_heatmap.png
│   ├── knn_metrics_by_threshold_tuned.png
│   ├── numerical_categorical_relationships_barplots.png
│   ├── numerical_distributions_histograms.png
│   ├── numerical_relationships_scatterplots.png
│   ├── overfitting_tuned_thresholds.png
│   ├── precision_recall_curves_baseline.png
│   ├── precision_recall_curves_tuned.png
│   ├── rf_confusion_matrix_test.png
│   ├── rf_feature_importance_final.png
│   ├── rf_metrics_by_threshold_tuned.png
│   ├── tree_metrics_by_threshold_tuned.png
│   └── xgb_metrics_by_threshold_tuned.png
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

### Descriptive statistics for numerical columns

| Column | Count | Mean | Std | Min | 25% | 50% | 75% | Max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Income | 201,600 | 5,000,741 | 2,880,711 | 10,310 | 2,499,018 | 5,004,535 | 7,489,827 | 9,999,938 |
| Age | 201,600 | 50.01 | 17.08 | 21 | 35 | 50 | 65 | 79 |
| Experience | 201,600 | 10.09 | 6.00 | 0 | 5 | 10 | 15 | 20 |
| Current Job Yrs | 201,600 | 6.33 | 3.65 | 0 | 3 | 6 | 9 | 14 |
| Current House Yrs | 201,600 | 12.00 | 1.40 | 10 | 11 | 12 | 13 | 14 |
| State Default Rate | 201,600 | 0.12 | 0.02 | 0.05 | 0.11 | 0.12 | 0.13 | 0.21 |

### Baseline model comparison

| Model | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.16 | 0.00 | 0.00 | 0.00 | 0.88 |
| Elastic Net | 0.16 | 0.00 | 0.00 | 0.00 | 0.88 |
| K-Nearest Neighbors | 0.53 | 0.50 | 0.56 | 0.53 | 0.89 |
| Support Vector Machine | 0.13 | 0.28 | 0.13 | 0.18 | 0.67 |
| Neural Network | 0.32 | 0.10 | 0.51 | 0.17 | 0.88 |
| Decision Tree | 0.46 | 0.57 | 0.52 | 0.55 | 0.88 |
| Random Forest | 0.60 | 0.53 | 0.60 | 0.56 | 0.90 |
| XGBoost | 0.54 | 0.20 | 0.64 | 0.30 | 0.89 |

### Hyperparameter-tuned models — default thresholds

| Model | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: |
| K-Nearest Neighbors | 0.59 | 0.54 | 0.59 | 0.56 | 0.89 |
| Decision Tree | 0.52 | 0.87 | 0.46 | 0.60 | 0.86 |
| Random Forest | 0.62 | 0.77 | 0.57 | 0.65 | 0.90 |
| XGBoost | 0.61 | 0.81 | 0.53 | 0.64 | 0.89 |

### Hyperparameter-tuned models — optimized thresholds

| Model | AUC-PR | Recall (Class 1) | Precision (Class 1) | F1-Score (Class 1) | Accuracy |
| :--- | ---: | ---: | ---: | ---: | ---: |
| K-Nearest Neighbors | 0.59 | 0.81 | 0.52 | 0.63 | 0.88 |
| Decision Tree | 0.52 | 0.81 | 0.49 | 0.61 | 0.87 |
| Random Forest | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| XGBoost | 0.61 | 0.80 | 0.54 | 0.64 | 0.89 |

### Threshold optimization

![Random Forest](images/rf_metrics_by_threshold_tuned.png)

![XGBoost](images/xgb_metrics_by_threshold_tuned.png)

![Decision Tree](images/tree_metrics_by_threshold_tuned.png)

![KNN](images/knn_metrics_by_threshold_tuned.png)

---

## License

MIT License.
