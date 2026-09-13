import numpy as np

from src.modeling import evaluate_binary_classifier, select_threshold


def test_evaluate_binary_classifier_returns_core_metrics():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.05, 0.2, 0.8, 0.9])
    metrics = evaluate_binary_classifier(y_true, probabilities, threshold=0.5)

    assert 0.0 <= metrics["pr_auc"] <= 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_threshold_selection_honors_constraints_when_feasible():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.02, 0.1, 0.2, 0.7, 0.85, 0.95])
    result = select_threshold(y_true, probabilities, min_recall=0.75, min_precision=0.50)

    assert result.recall >= 0.75
    assert result.precision >= 0.50
    assert 0.0 <= result.threshold <= 1.0
