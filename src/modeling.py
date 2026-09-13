"""Model construction, evaluation and threshold optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Risk Flag"

NUMERIC_FEATURES = [
    "Income",
    "Age",
    "Experience",
    "CURRENT_JOB_YRS",
    "CURRENT_HOUSE_YRS",
]

CATEGORICAL_FEATURES = [
    "Profession",
    "Married",
    "House_Ownership",
    "Car_Ownership",
    "City",
    "State",
]


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    y_test: pd.Series


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.10,
    validation_size: float = 0.10,
    random_state: int = 42,
) -> SplitData:
    """Create deterministic stratified train/validation/test partitions."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size + validation_size, stratify=y, random_state=random_state
    )
    relative_test = test_size / (test_size + validation_size)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=random_state
    )
    return SplitData(X_train, X_valid, X_test, y_train, y_valid, y_test)


def build_pipeline(random_state: int = 42) -> Pipeline:
    """Build the complete preprocessing + classifier pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def optimize_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
    min_precision: float = 0.50,
    min_recall: float = 0.75,
) -> tuple[float, dict[str, float]]:
    """Select the highest-F1 threshold satisfying the operating constraints."""
    best: tuple[float, dict[str, float]] | None = None
    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = probabilities >= threshold
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        f1 = f1_score(y_true, predictions, zero_division=0)
        if precision >= min_precision and recall >= min_recall:
            metrics = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
            if best is None or f1 > best[1]["f1"]:
                best = (float(threshold), metrics)

    if best is None:
        # Fall back to the F1-optimal threshold if constraints cannot be met.
        thresholds = np.linspace(0.05, 0.95, 181)
        scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
        index = int(np.argmax(scores))
        threshold = float(thresholds[index])
        predictions = probabilities >= threshold
        best = (
            threshold,
            {
                "precision": float(precision_score(y_true, predictions, zero_division=0)),
                "recall": float(recall_score(y_true, predictions, zero_division=0)),
                "f1": float(scores[index]),
            },
        )
    return best


def evaluate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate ranking and threshold-dependent classification performance."""
    probabilities = model.predict_proba(X)[:, 1]
    predictions = probabilities >= threshold
    return {
        "average_precision": float(average_precision_score(y, probabilities)),
        "precision": float(precision_score(y, predictions, zero_division=0)),
        "recall": float(recall_score(y, predictions, zero_division=0)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(y, predictions, output_dict=True, zero_division=0),
    }
