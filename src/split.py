"""Chronological train / validation / test split by calendar year."""

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

TEST_YEAR = 2025
VAL_YEAR = TEST_YEAR - 1


def split_by_year(
    df: pd.DataFrame, val_year: int = VAL_YEAR, test_year: int = TEST_YEAR
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train is every year before val_year, validation is val_year, test is test_year."""
    year = df["timestamp"].dt.year
    train = df[year < val_year]
    val = df[year == val_year]
    test = df[year == test_year]
    return train, val, test


def time_series_cv(n_splits: int = 5) -> TimeSeriesSplit:
    """Expanding-window CV for tuning on the training years (rows must be time-sorted)."""
    return TimeSeriesSplit(n_splits=n_splits)
