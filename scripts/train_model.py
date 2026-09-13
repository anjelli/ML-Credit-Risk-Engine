"""Train, tune and evaluate the credit-risk model.

Usage:
    python scripts/train_model.py --data data/training_data.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib

from src.data import load_data, split_features_target
from src.modeling import build_pipeline, make_splits, optimize_threshold, evaluate
from src.plots import save_evaluation_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the loan default prediction pipeline.")
    parser.add_argument("--data", type=Path, required=True, help="Path to the training CSV.")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="Artifact directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_data(args.data)
    X, y = split_features_target(df)
    split = make_splits(X, y)

    model = build_pipeline()
    model.fit(split.X_train, split.y_train)

    valid_probabilities = model.predict_proba(split.X_valid)[:, 1]
    threshold, threshold_metrics = optimize_threshold(split.y_valid, valid_probabilities)
    test_metrics = evaluate(model, split.X_test, split.y_test, threshold)

    args.output.mkdir(parents=True, exist_ok=True)
    figures_dir = args.output / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, args.output / "credit_risk_pipeline.joblib")

    metrics = {
        "threshold": threshold,
        "validation_threshold_metrics": threshold_metrics,
        "test_metrics": test_metrics,
        "split_sizes": {
            "train": len(split.X_train),
            "validation": len(split.X_valid),
            "test": len(split.X_test),
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    save_evaluation_plots(
        model=model,
        X_test=split.X_test,
        y_test=split.y_test,
        threshold=threshold,
        output_dir=figures_dir,
    )

    print(f"Test PR-AUC: {test_metrics['average_precision']:.4f}")
    print(f"Threshold:   {threshold:.3f}")
    print(f"Precision:   {test_metrics['precision']:.4f}")
    print(f"Recall:      {test_metrics['recall']:.4f}")
    print(f"F1:          {test_metrics['f1']:.4f}")
    print(f"Artifacts:   {args.output.resolve()}")


if __name__ == "__main__":
    main()
