# ML Credit Risk Engine

A complete machine-learning workflow for loan-default prediction and credit-risk analysis, covering data preparation, exploratory analysis, model comparison, hyperparameter tuning, threshold optimization, evaluation, and reusable inference.

The project treats loan default prediction as an **imbalanced binary-classification problem** and therefore places particular emphasis on **PR-AUC, recall, precision, F1-score, and operating-threshold selection** rather than accuracy alone.

---

## Overview

The pipeline is organized as:

```text
Raw applicant data
        |
        v
Data validation & cleaning
        |
        v
Feature engineering
        |
        v
Train / validation / test split
        |
        v
Preprocessing
  - numerical scaling
  - categorical encoding
        |
        v
Model comparison
        |
        v
Hyperparameter tuning
        |
        v
Threshold optimization
        |
        v
Final model evaluation
        |
        v
Reusable inference pipeline
```

### What the project covers

- Data quality checks and preprocessing
- Feature engineering for financial, employment, housing, and geographic variables
- Exploratory Data Analysis across numerical and categorical features
- Eight baseline classification models
- Randomized hyperparameter search with cross-validation
- Precision-recall analysis for an imbalanced target
- Validation-set threshold optimization
- Overfitting diagnostics
- Confusion-matrix and feature-importance analysis
- Reusable scikit-learn inference pipeline
- Docker/application scaffolding
- Automated testing with `pytest`

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Computing-013243?style=flat-square&logo=numpy&logoColor=white)
![Scikit Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-017CEE?style=flat-square)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat-square)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-Containerization-2496ED?style=flat-square&logo=docker&logoColor=white)
![Pytest](https://img.shields.io/badge/pytest-Testing-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

---

## Problem Statement

Loan default prediction is a challenging classification problem because risk is influenced by interacting demographic, financial, employment, housing, and geographic variables.

The dataset is also substantially imbalanced: defaults are a minority outcome. A model can therefore achieve high overall accuracy while performing poorly at identifying actual defaulters.

This project addresses that problem by using **PR-AUC as the primary ranking metric** and by explicitly tuning the probability threshold used for the final classification decision.

### Objectives

- Predict whether a loan applicant is likely to default.
- Compare multiple classification algorithms under class imbalance.
- Identify the strongest predictive signals.
- Tune promising models using cross-validation.
- Optimize the classification threshold using validation data.
- Evaluate generalization on an untouched test set.
- Package preprocessing and inference into a reusable workflow.

---

## Dataset

The project uses the public **Loan Prediction Based on Customer Behavior** dataset.

### Dataset statistics

| Property | Value |
|---|---:|
| Records | 252,000 |
| Target | `Risk Flag` |
| Default rate | ~12.3% |
| Input variables | 11 |

### Feature groups

| Group | Variables |
|---|---|
| Demographic | Age, Married/Single, Profession |
| Financial | Income, House Ownership, Car Ownership |
| Employment | Experience, Current Job Years |
| Housing | Current House Years |
| Geographic | City, State |
| Target | Risk Flag |

### Data dictionary

| Column | Description | Type |
|---|---|---|
| `Risk Flag` | Loan default indicator | Binary |
| `Income` | Applicant income | Numerical |
| `Age` | Applicant age | Numerical |
| `Experience` | Work experience in years | Numerical |
| `Profession` | Applicant profession | Categorical |
| `Married/Single` | Marital status | Categorical |
| `House_Ownership` | Housing ownership status | Categorical |
| `Car_Ownership` | Vehicle ownership | Categorical |
| `CURRENT_JOB_YRS` | Years in current job | Numerical |
| `CURRENT_HOUSE_YRS` | Years in current house | Numerical |
| `City` | City of residence | Categorical |
| `State` | State of residence | Categorical |

### Example records

| Risk Flag | Income | Age | Experience | Profession | Married | House Ownership | Car Ownership | Current Job Years | Current House Years | City | State |
|---:|---:|---:|---:|---|---|---|---|---:|---:|---|---|
| 0 | 1,303,834 | 23 | 3 | Mechanical_engineer | single | rented | no | 3 | 13 | Rewa | Madhya_Pradesh |
| 1 | 6,256,451 | 41 | 2 | Software_Developer | single | rented | yes | 2 | 12 | Bangalore | Tamil_Nadu |
| 0 | 3,991,815 | 66 | 4 | Technical_writer | married | rented | no | 4 | 10 | Alappuzha | Kerala |

---

## Data Preprocessing

The preprocessing workflow is designed to keep training and inference transformations consistent.

### Cleaning and validation

- Standardized column names and category labels
- Checked duplicate identifiers and duplicate rows
- Checked missing values and data types
- Validated train/validation/test partitions
- Analysed univariate and multivariate outliers

### Feature engineering

Additional signals were derived to capture information that is not directly represented by a single raw column:

- **Job stability** from professional/employment information
- **City tier** from geographic information
- **State default rate** as a historical target-based geographic signal

### Transformation

- Numerical features standardized with `StandardScaler`
- Categorical features encoded using scikit-learn encoders
- Transformations organized into reusable pipeline components
- Final preprocessing kept coupled with model inference to reduce train/serve inconsistencies

---

# Exploratory Data Analysis

EDA was performed at both the univariate and bivariate levels to understand distributions, imbalance, feature relationships, and potential predictive structure.

## Numerical distributions

![Numerical Distributions](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/numerical_distributions_histograms.png)

The numerical distributions provide visibility into applicant age, income, employment experience, job tenure, and housing tenure.

## Categorical frequencies

![Categorical Frequencies](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/categorical_frequencies_barplots.png)

Categorical frequency analysis highlights the composition of the applicant population across professions, ownership variables, marital status, cities, and states.

## Numerical-numerical relationships

![Numerical Relationships](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/numerical_relationships_scatterplots.png)

Scatterplot analysis is used to inspect relationships and potential nonlinear structure between continuous variables.

## Numerical-categorical relationships

![Numerical-Categorical Relationships](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/numerical_categorical_relationships_barplots.png)

These comparisons examine how numerical variables vary across important categorical groups.

## Categorical-categorical relationships

![Categorical-Categorical Relationships](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/categorical_relationships_groupedbarplots.png)

Grouped categorical analysis is used to inspect relationships among categorical variables and their interaction with the target structure.

## Correlation heatmap

<img src="https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/correlation_heatmap.png" alt="Correlation Heatmap" width="850">

The correlation matrix helps identify strongly related numerical variables and potential redundancy before modeling.

---

# Modeling

Eight baseline classifiers were evaluated:

1. Logistic Regression
2. Elastic Net
3. K-Nearest Neighbors
4. Support Vector Machine
5. Neural Network
6. Decision Tree
7. Random Forest
8. XGBoost

## Evaluation strategy

Because the default class is imbalanced, **Area Under the Precision-Recall Curve (AUC-PR)** is the primary ranking metric.

Secondary metrics are:

- Class-1 recall
- Class-1 precision
- Class-1 F1-score
- Accuracy
- Confusion matrix

The test set is held out from model selection and threshold optimization.

---

## Baseline model comparison

![Baseline AUC-PR Comparison](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/aucpr_comparison_baseline.png)

![Baseline Precision-Recall Curves](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/precision_recall_curves_baseline.png)

The strongest baseline candidates were Random Forest, XGBoost, KNN, and Decision Tree.

### Baseline results

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

---

# Hyperparameter Tuning

The strongest candidate models were tuned using **RandomizedSearchCV with 5-fold cross-validation**.

![Tuned Precision-Recall Curves](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/precision_recall_curves_tuned.png)

The tuned models were subsequently evaluated on validation data before selecting the final operating threshold.

---

# Threshold Optimization

A probability score does not automatically define a useful business decision. The operating threshold determines how aggressively the system flags an applicant as a potential default.

Thresholds were therefore evaluated on validation data to balance recall and precision.

![Random Forest Threshold Metrics](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/rf_metrics_by_threshold_tuned.png)

![XGBoost Threshold Metrics](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/xgb_metrics_by_threshold_tuned.png)

![Decision Tree Threshold Metrics](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/tree_metrics_by_threshold_tuned.png)

![KNN Threshold Metrics](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/knn_metrics_by_threshold_tuned.png)

## Tuned-model comparison

| Model | AUC-PR | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| KNN | 0.59 | 0.81 | 0.52 | 0.63 | 0.88 |
| Decision Tree | 0.52 | 0.81 | 0.49 | 0.61 | 0.87 |
| Random Forest | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| XGBoost | 0.61 | 0.80 | 0.54 | 0.64 | 0.89 |

---

# Overfitting Analysis

Model selection should consider not only predictive performance but also the gap between training and validation behavior.

![Overfitting Analysis](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/overfitting_tuned_thresholds.png)

Random Forest showed the smallest train-to-validation AUC-PR gap among the tuned candidates, supporting its selection as the final model.

---

# Final Model

The final model is a **Random Forest classifier with an optimized decision threshold**.

### Selected hyperparameters

```text
n_estimators       = 225
max_depth          = 26
min_samples_split  = 2
min_samples_leaf   = 1
max_features       = 0.13
class_weight       = "balanced"
```

### Why Random Forest

Random Forest provided:

- Strong PR-AUC among the candidate models
- Competitive thresholded F1-score
- High recall at the selected operating point
- The lowest overfitting gap among the tuned candidates
- Good interpretability through feature importance

---

# Final Evaluation

### Training / validation / test performance

| Dataset | AUC-PR | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Training | 0.68 | 1.00 | 0.62 | 0.77 | 0.93 |
| Validation | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| Test | **0.59** | **0.79** | **0.51** | **0.62** | **0.88** |

### Test classification report

| Class | Precision | Recall | F1 | Samples |
|---|---:|---:|---:|---:|
| Non-default | 0.97 | 0.90 | 0.93 | 22,122 |
| Default | **0.51** | **0.79** | **0.62** | 3,078 |
| Macro average | 0.74 | 0.84 | 0.78 | 25,200 |
| Weighted average | 0.91 | 0.88 | 0.89 | 25,200 |

### Confusion matrix

<img src="https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/rf_confusion_matrix_test.png" alt="Random Forest Confusion Matrix" width="500">

### Feature importance

The strongest signals in the final model include income, age, and the engineered state-level default-rate feature, with employment-related variables also contributing meaningful signal.

![Random Forest Feature Importance](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/rf_feature_importance_final.png)

---

# Model Interpretation

The project explicitly separates two ideas:

1. **Risk estimation** — the classifier estimates a probability of default.
2. **Decision policy** — a threshold converts that probability into an operational prediction.

This separation makes the system easier to evaluate and modify without retraining the underlying model every time the operating policy changes.

The threshold is selected using validation data and then frozen before the final test-set evaluation.

---

# Deployment Architecture

The repository is structured so the trained preprocessing + model workflow can be served as a reusable inference component.

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

The repository also contains Docker and application scaffolding for extending the model into a deployable service.

---

# Getting Started

## Prerequisites

- Python 3.10+
- Git
- Docker Desktop (optional)

## Installation

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

## Train the model

```bash
python scripts/train_model.py --data data/training_data.csv
```

## Run tests

```bash
pytest
```

---

# Usage

## Local inference

```python
import joblib
import pandas as pd

pipeline = joblib.load("artifacts/credit_risk_pipeline.joblib")

applicant = pd.DataFrame({
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

probability = pipeline.predict_proba(applicant)[:, 1]
print(f"Probability of default: {probability[0]:.3f}")
```

The operating threshold used for classification is selected during validation and stored with the evaluation artifacts generated by training.

---

# Repository Structure

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

# Testing

The project includes a pytest-based testing layer for the model and evaluation utilities.

```bash
pytest
```

Testing focuses on reusable modeling components rather than relying only on notebook execution.

---

# Responsible Use

This project is intended as a **credit-risk decision-support system**, not an autonomous lending system.

Model predictions should not be the sole basis for financial decisions. A production-grade deployment would require additional work around:

- probability calibration
- fairness and bias evaluation
- model drift monitoring
- data quality monitoring
- explainability
- governance and auditability
- human review
- regulatory and legal compliance

Historical training data can contain structural and socioeconomic biases. Model performance should therefore be evaluated across relevant population segments before operational use.

---

# Results at a Glance

| Item | Result |
|---|---|
| Dataset size | 252,000 records |
| Default rate | ~12.3% |
| Models evaluated | 8 baseline classifiers |
| Primary metric | PR-AUC |
| Final model | Random Forest |
| Test PR-AUC | 0.59 |
| Test default recall | 0.79 |
| Test default precision | 0.51 |
| Test default F1 | 0.62 |
| Test accuracy | 0.88 |

The core result is not simply the final accuracy figure: the workflow demonstrates how model selection, class imbalance, validation-based thresholding, and test-set evaluation interact in a practical credit-risk pipeline.

---

# License

MIT License.
