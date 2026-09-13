"""Data loading and validation utilities for the credit-risk pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

TARGET = "Risk Flag"
EXPECTED_COLUMNS = {
    "Income",
    "Age",
    "Experience",
    "Profession",
    "Married/Single",
    "House_Ownership",
    "Car_Ownership",
    "CURRENT_JOB_YRS",
    "CURRENT_HOUSE_YRS",
    "City",
    "State",
    TARGET,
}


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the loan dataset and validate its basic schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, received: {path.suffix}")

    df = pd.read_csv(path)
    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame, required: Iterable[str] = EXPECTED_COLUMNS) -> None:
    """Raise a clear error when required columns or target values are invalid."""
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    target_values = set(df[TARGET].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{TARGET!r} must contain only 0/1 values; found {target_values}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common source column names to the names used by the pipeline."""
    renamed = df.rename(
        columns={
            "Married/Single": "Married",
            "House_Ownership": "House_Ownership",
            "Car_Ownership": "Car_Ownership",
        }
    ).copy()
    return renamed


def split_features_target(df: pd.DataFrame):
    """Return feature matrix and binary target."""
    df = clean_column_names(df)
    return df.drop(columns=[TARGET]), df[TARGET].astype(int)
