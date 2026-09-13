"""Data loading and validation helpers for the credit-risk pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TARGET = "Risk Flag"
REQUIRED_COLUMNS = {
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


def load_training_data(path: str | Path) -> pd.DataFrame:
    """Load the raw loan dataset and validate its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame[TARGET].isna().any():
        raise ValueError(f"Target column {TARGET!r} contains missing values")
    return frame


def clean_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized column names."""
    cleaned = frame.copy()
    cleaned.columns = (
        cleaned.columns.astype(str)
        .str.strip()
        .str.replace("/", "_", regex=False)
        .str.replace(" ", "_", regex=False)
        .str.lower()
    )
    return cleaned
