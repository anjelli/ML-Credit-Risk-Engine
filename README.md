# Credit Risk Engine

A credit-risk decision-support project built around a supervised loan-default model and a deterministic policy layer.

> **Status:** the repository is being rebuilt around original implementation work. The earlier notebook and result plots from `JensBender/loan-default-prediction` have been removed from the project codebase; they are not used as evidence of results produced by this project.

## What this project is

The engine is designed as a small, auditable pipeline:

```text
Applicant input
      |
      v
Input validation
      |
      v
Feature engineering / preprocessing
      |
      v
Credit-risk model
      |
      v
Probability of default
      |
      v
Policy / threshold layer
      |
      +----> approve
      +----> review
      +----> reject
      |
      v
Structured explanation + audit record
```

The model is responsible for quantitative risk estimation. The policy layer is responsible for converting that score into an action. An LLM, if added later, can explain verified outputs but does not generate the underlying credit score.

## Dataset

The intended dataset is **Loan Prediction Based on Customer Behavior** by Subham Jain, publicly available through Kaggle.

The dataset contains applicant-level demographic, financial, employment, housing, and geographic variables with `Risk Flag` as the binary target.

Dataset: https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior

**No performance numbers are reported here yet.** Metrics will be added only after the model is trained and evaluated from this repository's own implementation.

## Planned implementation

### 1. Data pipeline

- load and validate the public dataset
- deterministic train / validation / test split
- missing-value and duplicate checks
- categorical encoding and numerical scaling
- leakage-safe feature engineering

### 2. Model layer

Candidate models will include interpretable and tree-based classifiers. Model selection will use PR-AUC because the target is imbalanced, with recall, precision, F1, calibration, and confusion-matrix analysis as secondary diagnostics.

### 3. Decision layer

The probability output will be separated from the operating policy. Thresholds will be selected on validation data against explicitly documented constraints rather than copied from another implementation.

### 4. Agentic orchestration

The eventual LangGraph workflow will orchestrate deterministic components:

```text
Validation -> Risk scoring -> Policy check -> Review routing -> Explanation / audit
```

The agentic component will add workflow/state management rather than pretending an LLM is a credit-risk model.

## Engineering goals

- reproducible preprocessing and inference
- explicit separation of model score and policy decision
- testable validation and policy components
- containerized serving
- structured logging / audit output
- no unsupported claims about model performance or deployment

## Repository structure

```text
.
├── README.md
├── Dockerfile
├── start.sh
├── requirements.txt
├── requirements-notebook.txt
├── requirements-test.txt
├── pytest.ini
├── upload_to_huggingface.py
├── README-hf-hub.md
├── README-hf-space.md
└── LICENSE
```

The application source directories referenced by the old Dockerfile are not currently present. The deployment configuration will be repaired when the application implementation is added rather than leaving a Docker build that points to missing files.

## Responsible use

This is a research / portfolio project and is not a production lending system. A real credit-risk deployment would require representative data, leakage and bias analysis, calibration, fairness evaluation, monitoring, governance, regulatory review, and human oversight.

## Attribution

The earlier repository state incorporated material from `JensBender/loan-default-prediction`. That material is no longer used as this project's model implementation or performance evidence. The public Kaggle dataset remains an appropriate source dataset for an independently implemented model.

## License

See `LICENSE` for the repository license.
