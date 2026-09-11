# Agentic Credit Risk Engine

End-to-end **credit-risk modeling and decision support** for predicting loan default from applicant-level data. The project combines supervised machine learning with a deployable inference stack: feature engineering, preprocessing, model comparison, threshold optimization, FastAPI serving, Gradio UI, Docker, automated deployment, and testing.

> **Current implementation:** a production-style ML inference application for loan-default prediction.
>
> **Agentic extension:** LangGraph orchestration can be layered on top of the model to coordinate validation, risk scoring, policy checks, and grounded explanations. The current repository should not be interpreted as already containing that LangGraph layer.

## Why this project

Credit-risk modeling is an imbalanced classification problem where the cost of missing a likely defaulter can differ substantially from the cost of reviewing or rejecting a low-risk applicant.

This project focuses on three practical questions:

1. Can default risk be predicted reliably?
2. Can the decision threshold be tuned for the minority class rather than relying on accuracy alone?
3. Can the resulting model be exposed as a reusable inference service instead of remaining notebook-only?

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

The operating threshold was optimized to prioritize default detection while maintaining minimum precision/recall constraints.

The dataset contains **252,000 loan applications**, with defaults representing approximately **12.3%** of observations.

## System overview

```text
                    Applicant Data
                         |
                         v
                Data Validation / Cleaning
                         |
                         v
                 Feature Engineering
                         |
                         v
             Preprocessing Pipeline
        (encoding + scaling + transforms)
                         |
                         v
                  Risk Model
             Random Forest Classifier
                         |
                         v
               Default Probability
                         |
                         v
             Optimized Decision Threshold
                         |
                         v
               API / Gradio Interface
```

### Planned agentic decision layer

The model is designed to become one component of a lightweight LangGraph workflow rather than being replaced by an LLM:

```text
Application
    |
    v
Validation
    |
    v
Credit-Risk Model ──────> Default Probability
    |
    v
Policy / Threshold Check
    |
    +----> APPROVE
    +----> REVIEW
    +----> REJECT
    |
    v
Grounded Explanation
```

The quantitative decision remains model-driven; the agentic layer is used for workflow orchestration rather than inventing credit scores.

## Data

Source: **Loan Prediction Based on Customer Behavior** dataset by Subham Jain.

- **252,000** records
- **12.3%** default rate
- **11 input features**
- Binary target: `Risk Flag`

Feature groups include:

- **Demographic:** age, marital status, profession
- **Financial:** income, house ownership, car ownership
- **Employment:** experience, current job years
- **Location:** city, state
- **Housing:** current house years

Dataset: [Kaggle](https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior)

## Machine-learning workflow

### Data preparation

The modeling notebook implements:

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

Derived features include:

- job-stability indicators
- city-tier information
- state-level historical default rate

### Model comparison

Candidate classifiers include:

- Logistic Regression
- Elastic Net
- K-Nearest Neighbors
- Support Vector Machine
- Decision Tree
- Random Forest
- XGBoost
- Multi-Layer Perceptron

Hyperparameters were tuned using randomized search with cross-validation.

### Evaluation

Because the default class is the minority class, the project emphasizes **PR-AUC** rather than accuracy alone.

Secondary metrics include:

- precision
- recall
- F1-score
- confusion matrix
- precision-recall curves

### Threshold optimization

The classifier outputs a probability rather than relying blindly on the default `0.50` cutoff. The operating threshold is selected to balance default recall and precision under explicit constraints.

```text
Model training  →  estimate probability of default
Decision policy  →  convert probability into an action
```

## Deployment

The complete preprocessing + model pipeline is exposed through a lightweight application stack:

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

- **Python**
- **pandas / NumPy**
- **scikit-learn**
- **FastAPI / Uvicorn**
- **Pydantic**
- **Gradio**
- **Docker**
- **Hugging Face Hub / Spaces**
- **pytest**
- **GitHub Actions**

Pinned dependencies are provided in `requirements.txt`.

## Repository structure

```text
.
├── loan_default_prediction.ipynb   # EDA, preprocessing, modeling and evaluation
├── README.md                       # Project documentation
├── Dockerfile                      # Container definition
├── start.sh                        # Application startup script
├── requirements.txt                # Runtime dependencies
├── requirements-notebook.txt      # Notebook dependencies
├── requirements-test.txt          # Test dependencies
├── pytest.ini                      # Test configuration
├── upload_to_huggingface.py        # Hugging Face deployment helper
├── README-hf-hub.md                # Model-card documentation
├── README-hf-space.md              # Deployment documentation
└── LICENSE                         # MIT license
```

The repository currently centers the reproducible modeling notebook and deployment configuration.

## Running locally

### Install

```bash
git clone https://github.com/anjelli/Agentic-Credit-Risk-Engine.git
cd Agentic-Credit-Risk-Engine

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

### Model development

Open `loan_default_prediction.ipynb` to run the EDA, preprocessing, modeling, evaluation, and threshold-optimization workflow.

### Application

The deployment stack is containerized:

```bash
docker build -t credit-risk-engine .
docker run -p 7860:7860 credit-risk-engine
```

## API contract

The inference service accepts applicant attributes and returns a default prediction together with model probabilities.

Example payload:

```json
{
  "income": 300000,
  "age": 30,
  "experience": 3,
  "married": "single",
  "house_ownership": "rented",
  "car_ownership": "no",
  "profession": "artist",
  "city": "sikar",
  "state": "rajasthan",
  "current_job_yrs": 3,
  "current_house_yrs": 11
}
```

Example response fields:

```json
{
  "prediction": "No Default",
  "probabilities": {
    "No Default": 0.84,
    "Default": 0.16
  }
}
```

The deployed model uses an optimized decision threshold rather than assuming `0.50` is always appropriate.

## Testing

The repository includes a test configuration for the model-serving stack and application behavior.

```text
Unit tests
    ↓
Component / API integration tests
    ↓
End-to-end application tests
```

This matters in credit-risk systems because preprocessing, serialization, thresholding, and API behavior need to remain consistent between training and inference.

## Model limitations

This is a **decision-support research project**, not a production lending policy.

Important limitations include:

- the training data may contain historical socioeconomic or geographic bias
- the dataset may not represent current lending populations
- performance can degrade under distribution shift
- reported metrics come from a hold-out test set
- threshold selection reflects project constraints rather than an institution's actual loss function
- the model should not be used as the sole basis for real-world lending decisions

A real deployment would require additional validation, fairness analysis, monitoring, governance, and human oversight.

## Agentic extension

The intended LangGraph layer is deliberately small. Instead of asking an LLM to decide whether a borrower is creditworthy, LangGraph can orchestrate deterministic and model-backed components:

```text
                +------------------+
                | Application Input|
                +--------+---------+
                         |
                         v
                +------------------+
                | Input Validation |
                +--------+---------+
                         |
                         v
                +------------------+
                |  Risk Prediction |
                |  sklearn model   |
                +--------+---------+
                         |
                         v
                +------------------+
                | Policy / Rules   |
                +---+----------+---+
                    |          |
                  review     decision
                    |          |
                    +----+-----+
                         |
                         v
                +------------------+
                | Grounded Output  |
                +------------------+
```

This keeps the architecture simple and defensible:

- **ML model:** predicts default probability
- **Decision layer:** applies explicit policy and thresholds
- **LangGraph:** orchestrates workflow and state transitions
- **LLM, if used:** explains verified model outputs rather than generating unsupported risk assessments

## Reproducibility

Preprocessing and inference are kept together in the serialized pipeline to reduce training-serving inconsistencies. Dependency versions are pinned, and the application is containerized for repeatable deployment.

## License

Source code in this repository is released under the **MIT License**.

See `LICENSE` for details.

## References

- [scikit-learn](https://scikit-learn.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Gradio](https://www.gradio.app/)
- [Hugging Face](https://huggingface.co/)
- [LangGraph](https://langchain-ai.github.io/langgraph/)
