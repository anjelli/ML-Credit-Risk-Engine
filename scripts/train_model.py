"""Train and evaluate the credit-risk model from the command line.

Usage:
    python scripts/train_model.py --data data/training_data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import TARGET, load_training_data
from src.modeling import evaluate_binary_classifier, select_threshold
from src.plots import save_confusion_matrix, save_precision_recall_curve, save_threshold_curve


NUMERIC_COLUMNS = [
    "Income",
    "Age",
    "Experience",
    "CURRENT_JOB_YRS",
    "CURRENT_HOUSE_YRS",
]
CATEGORICAL_COLUMNS = [
    "Profession",
    "Married/Single",
    "House_Ownership",
    "Car_Ownership",
    "City",
    "State",
]


def build_pipeline() -> Pipeline:
    """Build a leakage-safe preprocessing + classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to Training Data.csv")
    parser.add_argument("--artifacts", default="artifacts", help="Directory for model, metrics and figures")
    args = parser.parse_args()

    artifacts = Path(args.artifacts)
    (artifacts / "figures").mkdir(parents=True, exist_ok=True)

    frame = load_training_data(args.data)
    X = frame[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS]
    y = frame[TARGET].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]
    threshold_result = select_threshold(y_val, val_probabilities, min_recall=0.75, min_precision=0.50)

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary_classifier(y_test, test_probabilities, threshold_result.threshold)
    metrics["selected_threshold"] = threshold_result.threshold
    metrics["threshold_precision"] = threshold_result.precision
    metrics["threshold_recall"] = threshold_result.recall
    metrics["threshold_f1"] = threshold_result.f1

    joblib.dump(pipeline, artifacts / "credit_risk_pipeline.joblib")
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_precision_recall_curve(y_test, test_probabilities, artifacts / "figures" / "precision_recall_curve.png")
    save_confusion_matrix(y_test, test_probabilities, threshold_result.threshold, artifacts / "figures" / "confusion_matrix.png")
    save_threshold_curve(y_test, test_probabilities, artifacts / "figures" / "threshold_metrics.png")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
