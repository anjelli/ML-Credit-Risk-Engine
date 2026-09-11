# Agentic Credit Risk Engine

End-to-end **credit-risk modeling and decision support** for predicting loan default from applicant-level data. The project combines supervised machine learning with a deployable inference stack: feature engineering, preprocessing, model comparison, threshold optimization, FastAPI serving, Gradio UI, Docker, automated deployment, and testing.

> **Current implementation:** a production-style ML inference application for loan-default prediction.
>
> **Agentic extension:** LangGraph orchestration can be layered on top of the model to coordinate validation, risk scoring, policy checks, and grounded explanations. The current repository should not be interpreted as already containing that LangGraph layer.

## Key results

The final model is a **Random Forest classifier** selected after comparing and tuning multiple candidate models.

| Metric | Test set |
|---|---:|
| PR-AUC | **0.59** |
| Default recall | **0.79** |
| Default precision | **0.51** |
| Default F1 | **0.62** |
| Accuracy | **0.88** |
| Test observations | **25,200** |

The dataset contains **252,000 loan applications**, with defaults representing approximately **12.3%** of observations.

## Results & analysis

The analysis below preserves the original model-development results and visualizations from the source implementation while reorganizing them around the current project.

### Exploratory analysis

The dataset contains demographic, financial, employment, housing, and geographic variables. The original analysis examined numerical distributions, categorical frequencies, pairwise relationships, and correlations.

**Correlation structure**

![Correlation Heatmap](https://github.com/JensBender/loan-default-prediction/blob/main/images/correlation_heatmap.png?raw=true)

**Numerical distributions**

![Numerical Distributions](https://github.com/JensBender/loan-default-prediction/blob/main/images/numerical_distributions_histograms.png?raw=true)

**Categorical frequencies**

![Categorical Frequencies](https://github.com/JensBender/loan-default-prediction/blob/main/images/categorical_frequencies_barplots.png?raw=true)

**Numerical relationships**

![Numerical Relationships](https://github.com/JensBender/loan-default-prediction/blob/main/images/numerical_relationships_scatterplots.png?raw=true)

**Numerical vs. categorical relationships**

![Numerical-Categorical Relationships](https://github.com/JensBender/loan-default-prediction/blob/main/images/numerical_categorical_relationships_barplots.png?raw=true)

**Categorical relationships**

![Categorical Relationships](https://github.com/JensBender/loan-default-prediction/blob/main/images/categorical_relationships_groupedbarplots.png?raw=true)

### Baseline model comparison

Eight baseline classifiers were evaluated using **PR-AUC** as the primary metric because defaults account for only 12.3% of observations. The compared models were Logistic Regression, Elastic Net, K-Nearest Neighbors, Support Vector Machine, Decision Tree, Random Forest, XGBoost, and Multi-Layer Perceptron.

The original analysis compared four outlier-handling strategies and proceeded without outlier handling because it did not meaningfully improve PR-AUC.

![Baseline AUC-PR Comparison](https://github.com/JensBender/loan-default-prediction/blob/main/images/aucpr_comparison_baseline.png?raw=true)

![Baseline Precision-Recall Curves](https://github.com/JensBender/loan-default-prediction/blob/main/images/precision_recall_curves_baseline.png?raw=true)

Top baseline performers selected for tuning were Random Forest (**PR-AUC 0.62**), XGBoost (**0.56**), K-Nearest Neighbors (**0.56**), and Decision Tree (**0.47**).

### Hyperparameter tuning and threshold optimization

Hyperparameters were tuned with randomized search and 5-fold cross-validation. On validation data, Random Forest achieved the highest PR-AUC (**0.62**), followed by XGBoost (**0.61**).

![Tuned Precision-Recall Curves](https://github.com/JensBender/loan-default-prediction/blob/main/images/precision_recall_curves_tuned.png?raw=true)

The operating threshold was then optimized for the minority class, targeting **recall ≥ 0.80** and **precision ≥ 0.40** while maximizing F1-score.

![Random Forest Metrics by Threshold](https://github.com/JensBender/loan-default-prediction/blob/main/images/rf_metrics_by_threshold_tuned.png?raw=true)

The tuned validation comparison was:

| Model | PR-AUC | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| K-Nearest Neighbors | 0.59 | 0.81 | 0.52 | 0.63 | 0.88 |
| Decision Tree | 0.52 | 0.81 | 0.49 | 0.61 | 0.87 |
| Random Forest | **0.62** | 0.80 | **0.54** | **0.64** | **0.89** |
| XGBoost | 0.61 | 0.80 | 0.54 | **0.64** | **0.89** |

Random Forest was selected over XGBoost because it combined strong validation performance with the smallest train-validation PR-AUC gap (**0.06** vs. **0.13** for XGBoost).

![Threshold-Optimized Overfitting Analysis](https://github.com/JensBender/loan-default-prediction/blob/main/images/overfitting_tuned_thresholds.png?raw=true)

### Final model

**Random Forest + optimized threshold = 0.29**

Key hyperparameters:

```text
n_estimators      = 225
max_depth         = 26
min_samples_split = 2
min_samples_leaf  = 1
max_features      = 0.13
class_weight      = balanced
```

Performance across splits:

| Dataset | PR-AUC | Recall | Precision | F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Training | 0.68 | 1.00 | 0.62 | 0.77 | 0.93 |
| Validation | 0.62 | 0.80 | 0.54 | 0.64 | 0.89 |
| Test | **0.59** | **0.79** | **0.51** | **0.62** | **0.88** |

Test-set classification report:

| Class | Precision | Recall | F1 | Samples |
|---|---:|---:|---:|---:|
| Non-default | 0.97 | 0.90 | 0.93 | 22,122 |
| Default | **0.51** | **0.79** | **0.62** | 3,078 |
| Accuracy |  |  | **0.88** | 25,200 |

**Confusion matrix**

![Random Forest Test Confusion Matrix](https://github.com/JensBender/loan-default-prediction/blob/main/images/rf_confusion_matrix_test.png?raw=true)

**Final feature importance**

The original analysis identified **income, age, and state default rate** as the strongest features, with experience and current job years contributing moderately.

![Random Forest Feature Importance](https://github.com/JensBender/loan-default-prediction/blob/main/images/rf_feature_importance_final.png?raw=true)

## Data

Source: **Loan Prediction Based on Customer Behavior** by Subham Jain.

- **252,000** records
- **12.3%** default rate
- **11 input features**
- Binary target: `Risk Flag`

Feature groups include demographic, financial, employment, housing, and geographic variables.

Dataset: [Kaggle](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)

## Machine-learning workflow

### Data preparation

- duplicate checks
- type normalization
- missing-value checks
- train / validation / test splitting
- numerical and categorical feature identification
- outlier analysis
- reusable preprocessing with `ColumnTransformer`
- numerical scaling with `StandardScaler`
- categorical encoding with `OneHotEncoder` / `OrdinalEncoder`

### Feature engineering

Derived features include job-stability indicators, city-tier information, and state-level historical default rate.

### Deployment stack

```text
                Gradio UI
                    |
                    v
               FastAPI API
                    |
                    v
        Serialized sklearn pipeline
                    |
                    v
          Default probability
```

### Stack

- Python
- pandas / NumPy
- scikit-learn
- FastAPI / Uvicorn
- Pydantic
- Gradio
- Docker
- Hugging Face Hub / Spaces
- pytest
- GitHub Actions

## Planned agentic decision layer

The intended agentic layer is a workflow around the verified ML model rather than an LLM replacing the model:

```text
Application
    |
    v
Input Validation
    |
    v
Risk Prediction ──────> Default Probability
    |
    v
Policy / Threshold Check
    |
    +----> APPROVE
    +----> REVIEW
    +----> REJECT
    |
    v
Grounded Explanation / Audit Output
```

The quantitative risk score remains deterministic and model-backed. LangGraph can orchestrate state transitions, policy checks, manual-review routing, and explanation generation.

## Repository structure

```text
.
├── loan_default_prediction.ipynb   # EDA, preprocessing, modeling and evaluation
├── README.md                       # Project documentation
├── Dockerfile                      # Container definition
├── start.sh                        # Application startup script
├── requirements.txt                # Runtime dependencies
├── requirements-notebook.txt       # Notebook dependencies
├── requirements-test.txt           # Test dependencies
├── pytest.ini                      # Test configuration
├── upload_to_huggingface.py        # Hugging Face deployment helper
├── README-hf-hub.md                # Model-card documentation
├── README-hf-space.md              # Deployment documentation
└── LICENSE                         # MIT license
```

## Running locally

### Model development

Open `loan_default_prediction.ipynb` to reproduce the EDA, preprocessing, modeling, evaluation, and threshold-optimization workflow.

### Container

```bash
docker build -t credit-risk-engine .
docker run -p 7860:7860 credit-risk-engine
```

## Model limitations

This is a **decision-support research project**, not a production lending policy.

- historical training data may contain socioeconomic or geographic bias
- current populations may differ from the training distribution
- performance can degrade under distribution shift
- reported metrics come from a hold-out test set
- threshold selection reflects project constraints, not a real institution's loss function
- the model should not be the sole basis for real-world lending decisions

A real deployment would require additional validation, fairness analysis, monitoring, governance, and human oversight.

## Attribution

The EDA, model-development workflow, visualizations, and reported performance results reproduced in this documentation originate from the public `JensBender/loan-default-prediction` implementation and its underlying Kaggle dataset. This repository reorganizes and extends that work; it should not imply that the original analysis or assets were independently re-created.

## License

See `LICENSE` for the license governing this repository and `README-hf-hub.md` for model-card information.
