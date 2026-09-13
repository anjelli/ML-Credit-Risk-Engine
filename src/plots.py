"""Publication-ready evaluation plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, precision_recall_curve


def save_precision_recall_curve(y_true, probabilities, output_path: str | Path) -> None:
    """Save a precision-recall curve."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, probabilities, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_matrix(y_true, probabilities, threshold: float, output_path: str | Path) -> None:
    """Save a thresholded confusion matrix."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_true, predictions, display_labels=["No Default", "Default"], ax=ax)
    ax.set_title(f"Confusion Matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_threshold_curve(y_true, probabilities, output_path: str | Path) -> None:
    """Save precision/recall/F1 as a function of threshold."""
    from sklearn.metrics import f1_score, precision_score, recall_score

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds = np.linspace(0.01, 0.99, 99)
    precision, recall, f1 = [], [], []
    for threshold in thresholds:
        predictions = (np.asarray(probabilities) >= threshold).astype(int)
        precision.append(precision_score(y_true, predictions, zero_division=0))
        recall.append(recall_score(y_true, predictions, zero_division=0))
        f1.append(f1_score(y_true, predictions, zero_division=0))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, precision, label="Precision")
    ax.plot(thresholds, recall, label="Recall")
    ax.plot(thresholds, f1, label="F1")
    ax.set(xlabel="Decision threshold", ylabel="Score", title="Metrics by Decision Threshold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
