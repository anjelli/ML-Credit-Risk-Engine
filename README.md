# ML Credit Risk Engine

An end-to-end machine learning application for loan default prediction and credit risk assessment.

## Exploratory Data Analysis

### Numerical distributions

![Numerical Distributions](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/numerical_distributions_histograms.png)

### Categorical frequencies

![Categorical Frequencies](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/categorical_frequencies_barplots.png)

### Numerical-numerical relationships

![Numerical Relationships](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/numerical_relationships_scatterplots.png)

### Numerical-categorical relationships

![Numerical-Categorical Relationships](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/numerical_categorical_relationships_barplots.png)

### Categorical-categorical relationships

![Categorical-Categorical Relationships](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/categorical_relationships_groupedbarplots.png)

### Correlation heatmap

<img src="https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/correlation_heatmap.png" alt="Correlation Heatmap" width="850">

## Modeling

### Baseline model comparison

![AUC-PR Comparison](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/aucpr_comparison_baseline.png)

![Baseline Precision-Recall Curves](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/precision_recall_curves_baseline.png)

### Hyperparameter tuning

![Tuned Precision-Recall Curves](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/precision_recall_curves_tuned.png)

### Threshold optimization

![Random Forest Threshold Metrics](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/rf_metrics_by_threshold_tuned.png)

![XGBoost Threshold Metrics](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/xgb_metrics_by_threshold_tuned.png)

![Decision Tree Threshold Metrics](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/tree_metrics_by_threshold_tuned.png)

![KNN Threshold Metrics](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/knn_metrics_by_threshold_tuned.png)

### Overfitting analysis

![Overfitting Analysis](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/overfitting_tuned_thresholds.png)

## Final Model Results

The final workflow uses a Random Forest classifier with an optimized decision threshold.

| Dataset | AUC-PR | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Training | 0.68 | 1.00 | 0.62 | 0.77 | 0.93 |
| Validation | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| Test | 0.59 | 0.79 | 0.51 | 0.62 | 0.88 |

### Test confusion matrix

![Random Forest Confusion Matrix](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/rf_confusion_matrix_test.png)

### Feature importance

![Random Forest Feature Importance](https://media.githubusercontent.com/media/JensBender/loan-default-prediction/main/images/rf_feature_importance_final.png)

## Project Overview

The project covers:

- data cleaning and validation
- feature engineering
- exploratory data analysis
- eight-model baseline comparison
- hyperparameter tuning
- precision-recall analysis
- threshold optimization
- final model evaluation
- reusable model inference
- Docker/application scaffolding
- automated testing

## Dataset

The project uses the public **Loan Prediction Based on Customer Behavior** dataset with 252,000 records and `Risk Flag` as the binary target.

## Model comparison

| Model | AUC-PR | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| KNN | 0.59 | 0.81 | 0.52 | 0.63 | 0.88 |
| Decision Tree | 0.52 | 0.81 | 0.49 | 0.61 | 0.87 |
| Random Forest | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| XGBoost | 0.61 | 0.80 | 0.54 | 0.64 | 0.89 |

## Getting Started

```bash
git clone https://github.com/anjelli/ML-Credit-Risk-Engine.git
cd ML-Credit-Risk-Engine
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train:

```bash
python scripts/train_model.py --data data/training_data.csv
```

Test:

```bash
pytest
```

## Repository Structure

```text
ML-Credit-Risk-Engine/
├── images/
├── scripts/
│   └── train_model.py
├── src/
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
└── README.md
```

## Responsible Use

This is a credit-risk decision-support project, not an autonomous lending system. Production use would require additional validation, calibration, fairness analysis, monitoring, governance, explainability, and compliance review.

## License

MIT License.