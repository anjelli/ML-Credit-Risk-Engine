"""Plotting utilities for credit-risk model evaluation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve


def save_evaluation_plots(model, X_test, y_test, threshold: float, output_dir: Path) -> None:
    """Generate compact, publication-ready evaluation figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = model.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=180)
    plt.close(fig)

    predictions = probabilities >= threshold
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, ax=ax, values_format=",d")
    ax.set_title(f"Confusion Matrix (threshold = {threshold:.3f})")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    candidate_thresholds = thresholds[:: max(1, len(thresholds) // 100)]
    if len(candidate_thresholds) == 0:
        candidate_thresholds = [threshold]

    precisions, recalls, f1s = [], [], []
    for t in candidate_thresholds:
        pred = probabilities >= t
        tp = ((pred == 1) & (y_test.to_numpy() == 1)).sum()
        fp = ((pred == 1) & (y_test.to_numpy() == 0)).sum()
        fn = ((pred == 0) & (y_test.to_numpy() == 1)).sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(candidate_thresholds, precisions, label="Precision")
    ax.plot(candidate_thresholds, recalls, label="Recall")
    ax.plot(candidate_thresholds, f1s, label="F1")
    ax.axvline(threshold, linestyle="--", label=f"Selected = {threshold:.3f}")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Threshold Trade-offs")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "threshold_metrics.png", dpi=180)
    plt.close(fig)
