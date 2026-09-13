# ML Credit Risk Engine

A production-style machine learning pipeline for **loan default prediction**, designed around reproducible preprocessing, class-imbalance-aware evaluation, threshold optimization, and auditable risk decisions.

## At a glance

| Component | Implementation |
|---|---|
| Task | Binary loan-default classification |
| Dataset | 252,000 applicant records |
| Target | `Risk Flag` |
| Primary metric | Average Precision / PR-AUC |
| Model | Random Forest |
| Preprocessing | `Pipeline` + `ColumnTransformer` |
| Evaluation | PR-AUC, precision, recall, F1, confusion matrix |
| Decisioning | Validation-set threshold optimization |
| Artifact | Serialized inference pipeline |
| Tests | `pytest` |

## Pipeline

```text
                    ┌─────────────────────┐
                    │ Raw Applicant Data  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Schema / Data Checks│
                    └──────────┬──────────┘
                               │
                               ▼
                 ┌────────────────────────────┐
                 │ Leakage-safe Preprocessing │
                 │                            │
                 │ Numeric → scaling          │
                 │ Categorical → one-hot      │
                 └─────────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Random Forest Model │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Default Probability │
                    └──────────┬──────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │ Validation Threshold     │
                  │ Optimization             │
                  └────────────┬─────────────┘
                               │
                    ┌──────────┴──────────┐
                    ▼                     ▼
              Risk score             Decision policy
                    │                     │
                    └──────────┬──────────┘
                               ▼
                    Evaluation + Artifacts
```

## Why this project is structured this way

Loan default is an imbalanced classification problem, so accuracy alone can hide poor detection of the default class. The pipeline therefore uses **Average Precision (PR-AUC)** as its main ranking metric and evaluates the operating threshold separately using precision, recall and F1.

The model produces a probability. The operating threshold is a separate decision layer. This makes the model easier to evaluate, tune and audit without embedding business policy inside the classifier.

## Dataset

The project uses the **Loan Prediction Based on Customer Behavior** dataset containing applicant demographic, financial, employment, housing and geographic attributes.

Core fields include:

- Income
- Age
- Experience
- Profession
- Marital status
- House ownership
- Car ownership
- Current job years
- Current house years
- City
- State
- `Risk Flag` — binary default indicator

The expected dataset location is:

```text
data/training_data.csv
```

The dataset is not committed to the repository by default because of its size and redistribution considerations.

## Modeling

### Preprocessing

All transformations are encapsulated inside the training pipeline rather than performed as disconnected notebook operations.

- Numeric features are standardized with `StandardScaler`.
- Categorical features are one-hot encoded with `handle_unknown="ignore"`.
- Train/validation/test splits are stratified and deterministic.
- The fitted preprocessing graph is serialized together with the classifier.

This ensures training and inference use the same transformations.

### Model

The baseline production candidate is a class-weighted `RandomForestClassifier` with a fixed random seed. The implementation is intentionally modular so alternative classifiers can be benchmarked without rewriting the preprocessing layer.

### Threshold optimization

The classifier's probability output is not treated as a decision by itself. The operating threshold is selected using the validation set against explicit precision/recall constraints, then frozen before the test set is evaluated.

This prevents the test set from influencing the decision rule.

## Results and visual analysis

The repository is organized to retain the complete model-analysis workflow rather than only a single score.

### Model comparison

![Baseline model comparison](images/aucpr_comparison_baseline.png)

### Precision-recall analysis

![Precision recall curves](images/precision_recall_curves_baseline.png)

![Tuned precision recall curves](images/precision_recall_curves_tuned.png)

### Threshold analysis

![Threshold metrics](images/overfitting_tuned_thresholds.png)

### Final model diagnostics

![Confusion matrix](images/rf_confusion_matrix_test.png)

![Feature importance](images/rf_feature_importance_final.png)

### Exploratory data analysis

![Correlation heatmap](images/correlation_heatmap.png)

![Numerical distributions](images/numerical_distributions_histograms.png)

![Categorical frequencies](images/categorical_frequencies_barplots.png)

![Numerical relationships](images/numerical_relationships_scatterplots.png)

![Numerical categorical relationships](images/numerical_categorical_relationships_barplots.png)

![Categorical relationships](images/categorical_relationships_groupedbarplots.png)

The image set covers the main analytical stages: distributional analysis, relationships between variables, correlation structure, model comparison, precision-recall behavior, threshold selection, confusion-matrix diagnostics and feature importance.

## Reproducibility

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Install development/test dependencies:

```bash
pip install -r requirements-dev.txt
```

Place the dataset at `data/training_data.csv`, then run:

```bash
python scripts/train_model.py --data data/training_data.csv
```

Generated artifacts are written to:

```text
artifacts/
├── credit_risk_pipeline.joblib
├── metrics.json
└── figures/
    ├── precision_recall_curve.png
    ├── confusion_matrix.png
    └── threshold_metrics.png
```

Run the test suite:

```bash
pytest
```

## Repository structure

```text
ML-Credit-Risk-Engine/
│
├── data/                         # Local dataset; ignored by Git
├── images/                       # EDA + model-result visualizations
│
├── src/
│   ├── __init__.py
│   ├── data.py                   # Data loading and validation
│   ├── modeling.py               # Pipeline, model and evaluation logic
│   └── plots.py                  # Reusable evaluation plots
│
├── scripts/
│   └── train_model.py            # Reproducible training entry point
│
├── tests/
│   └── test_modeling.py          # Unit tests
│
├── loan_default_prediction.ipynb # Exploratory analysis notebook
├── requirements.txt
├── requirements-dev.txt
├── requirements-test.txt
├── requirements-notebook.txt
├── pytest.ini
├── Dockerfile
├── start.sh
├── .github/workflows/quality.yml
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

## Engineering decisions

### Leakage control

Preprocessing is fitted only within the training pipeline. Threshold selection is performed on validation data, while the test set remains a final unseen evaluation set.

### Imbalanced classification

PR-AUC is prioritized over raw accuracy because the default class is substantially smaller than the non-default class.

### Reusable components

Training, preprocessing, evaluation and plotting logic live in Python modules under `src/`. The notebook is retained for exploratory work rather than being the only executable representation of the project.

### Auditability

Model probability, decision threshold and classification outcome are conceptually separated. This makes changes to operating policy traceable without retraining the underlying classifier.

## Limitations

This is a credit-risk modeling project, not an autonomous lending system. Historical applicant data can encode socioeconomic and geographic biases. Any real deployment would require calibration, fairness testing, drift monitoring, governance controls, explainability review and human oversight.

## License

MIT License.
