# ML Credit Risk Engine

A reproducible machine-learning pipeline for loan-default prediction and credit-risk analysis.

The project separates **risk estimation** from **decision policy** so the model produces a probability of default while a separate threshold layer determines how that score is operationalized.

## Overview

```text
Raw applicant data
       |
       v
Schema validation
       |
       v
Train / validation / test split
       |
       v
Preprocessing
  - numeric scaling
  - categorical encoding
       |
       v
Random Forest classifier
       |
       v
Probability of default
       |
       v
Validation-based threshold selection
       |
       +----> approve / review / reject policy
       |
       v
Metrics + plots + audit-ready artifacts
```

## What is implemented

- Deterministic, stratified train/validation/test splitting.
- Leakage-safe preprocessing with `ColumnTransformer`.
- Numeric standardization and categorical one-hot encoding.
- Class-imbalance-aware Random Forest modeling.
- PR-AUC as the primary ranking metric.
- Threshold selection against explicit precision/recall constraints.
- Test-set precision, recall, F1 and confusion-matrix evaluation.
- Reusable plotting utilities for PR curves, threshold trade-offs and confusion matrices.
- Serialized inference pipeline via `joblib`.
- Unit tests for the evaluation and threshold-selection layer.

## Dataset

The project targets the public **Loan Prediction Based on Customer Behavior** dataset. The data contains applicant-level demographic, financial, employment, housing and geographic variables with `Risk Flag` as the binary target. The reference dataset contains 252,000 records and an imbalanced default class, making precision-recall analysis more informative than accuracy alone. fileciteturn5file0

Expected raw columns:

`Income`, `Age`, `Experience`, `Profession`, `Married/Single`, `House_Ownership`, `Car_Ownership`, `CURRENT_JOB_YRS`, `CURRENT_HOUSE_YRS`, `City`, `State`, `Risk Flag`

Place the CSV at:

```text
data/training_data.csv
```

## Modeling approach

### Preprocessing

The feature pipeline keeps transformation inside a scikit-learn `Pipeline` / `ColumnTransformer`, preventing preprocessing logic from being accidentally applied differently at training and inference time.

Numerical variables are standardized. Categorical variables use one-hot encoding with `handle_unknown="ignore"`, allowing inference on previously unseen categories without breaking the pipeline.

### Classifier

The current implementation uses a class-weighted `RandomForestClassifier` with a fixed random seed for reproducibility. The classifier is intentionally wrapped with preprocessing and saved as one inference artifact.

### Evaluation

The main evaluation metric is **Average Precision / PR-AUC**, appropriate for an imbalanced binary default target. Threshold-dependent metrics include:

| Metric | Purpose |
| --- | --- |
| PR-AUC | Overall ranking quality for the positive/default class |
| Precision | Share of flagged applications that are actual defaults |
| Recall | Share of actual defaults detected |
| F1 | Balance between precision and recall |
| Confusion matrix | Error decomposition at the operating threshold |

The decision threshold is selected on validation data and then frozen before test evaluation.

## Results

The repository includes the evaluation framework required to generate the result figures below. The README intentionally does not hard-code model performance that has not been reproduced from the current source tree.

After training, the generated artifacts are written to `artifacts/`:

```text
artifacts/
├── credit_risk_pipeline.joblib
├── metrics.json
└── figures/
    ├── precision_recall_curve.png
    ├── confusion_matrix.png
    └── threshold_metrics.png
```

### Evaluation visuals

#### Precision-recall performance

![Precision-Recall Curve](artifacts/figures/precision_recall_curve.png)

#### Confusion matrix at selected threshold

![Confusion Matrix](artifacts/figures/confusion_matrix.png)

#### Threshold trade-offs

![Threshold Metrics](artifacts/figures/threshold_metrics.png)

### Additional analysis to retain with the project

The project structure is designed to accommodate the broader analysis expected of a serious credit-risk workflow: model comparison, threshold optimization, feature importance, correlation analysis, categorical distributions, numerical distributions, and numerical/categorical relationship plots. The reference implementation demonstrates these analysis categories and reports PR-AUC, class-1 precision/recall/F1, confusion matrix and feature importance as its core outputs. fileciteturn5file0

## Reproduce the model

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run training:

```bash
python scripts/train_model.py --data data/training_data.csv
```

Run tests:

```bash
pytest
```

## Repository structure

```text
ML-Credit-Risk-Engine/
├── README.md
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
├── requirements-notebook.txt
├── requirements-test.txt
├── pytest.ini
├── loan_default_prediction.ipynb
├── scripts/
│   └── train_model.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── modeling.py
│   └── plots.py
├── tests/
│   └── test_modeling.py
├── images/
│   └── README.md
└── LICENSE
```

## Design principles

### Model score is not the policy

The classifier estimates default probability. A separate policy layer determines what score range maps to review or other operational actions. This prevents business rules from being silently buried inside the model.

### Reproducibility over notebook-only execution

The notebook remains useful for exploration, but the core modeling logic now lives in importable Python modules and a command-line training script.

### Metrics before claims

Performance values belong in the README only after the current implementation has produced them on a clearly defined validation/test split. This avoids stale or copied numbers surviving after code changes.

## Limitations and responsible use

This is a machine-learning research / portfolio pipeline, not an autonomous lending system. Credit decisions are high-impact decisions and should not be made from a model score alone. Any production use would require additional validation, calibration, fairness analysis, monitoring, governance, and review against applicable laws and institutional policy.

## License

MIT License.
