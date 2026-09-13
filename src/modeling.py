"""Reusable modeling, evaluation, and threshold-selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    precision: float
    recall: float
    f1: float


def evaluate_binary_classifier(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """Return PR-AUC plus thresholded classification metrics."""
    y_true_array = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true_array, probabilities)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_true_array, predictions, zero_division=0)),
        "f1": float(f1_score(y_true_array, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_array, predictions).tolist(),
        "classification_report": classification_report(y_true_array, predictions, zero_division=0),
    }


def select_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    min_recall: float = 0.75,
    min_precision: float = 0.50,
) -> ThresholdResult:
    """Select the highest-F1 threshold satisfying operating constraints."""
    y_true_array = np.asarray(y_true)
    probabilities = np.asarray(probabilities)
    thresholds = np.unique(np.clip(probabilities, 0.0, 1.0))

    best: ThresholdResult | None = None
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true_array, predictions, zero_division=0)
        recall = recall_score(y_true_array, predictions, zero_division=0)
        f1 = f1_score(y_true_array, predictions, zero_division=0)
        if precision >= min_precision and recall >= min_recall:
            candidate = ThresholdResult(float(threshold), float(precision), float(recall), float(f1))
            if best is None or candidate.f1 > best.f1:
                best = candidate

    if best is None:
        # Fall back to the best F1 operating point when constraints are infeasible.
        best_threshold = 0.5
        best_f1 = -1.0
        best_precision = 0.0
        best_recall = 0.0
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            precision = precision_score(y_true_array, predictions, zero_division=0)
            recall = recall_score(y_true_array, predictions, zero_division=0)
            f1 = f1_score(y_true_array, predictions, zero_division=0)
            if f1 > best_f1:
                best_threshold, best_f1 = float(threshold), float(f1)
                best_precision, best_recall = float(precision), float(recall)
        best = ThresholdResult(best_threshold, best_precision, best_recall, best_f1)

    return best


def compare_models(models: dict[str, Any], X_train, y_train, X_eval, y_eval) -> pd.DataFrame:
    """Fit cloned models and return a compact PR-AUC comparison table."""
    rows: list[dict[str, float | str]] = []
    for name, estimator in models.items():
        fitted = clone(estimator)
        fitted.fit(X_train, y_train)
        probabilities = fitted.predict_proba(X_eval)[:, 1]
        metrics = evaluate_binary_classifier(y_eval, probabilities)
        rows.append({"model": name, "pr_auc": metrics["pr_auc"]})
    return pd.DataFrame(rows).sort_values("pr_auc", ascending=False, ignore_index=True)
