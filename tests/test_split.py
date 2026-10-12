import numpy as np
import pandas as pd

from src.split import split_by_year, time_series_cv


def hourly_frame(start="2018-01-01", end="2025-12-31 23:00"):
    ts = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"timestamp": ts, "y": np.arange(len(ts))})


def test_split_is_chronological_and_disjoint():
    train, val, test = split_by_year(hourly_frame(), val_year=2024, test_year=2025)
    assert train["timestamp"].max() < val["timestamp"].min()
    assert val["timestamp"].max() < test["timestamp"].min()
    assert set(train.index).isdisjoint(val.index)
    assert set(val.index).isdisjoint(test.index)
    assert (train["timestamp"].dt.year <= 2023).all()
    assert (val["timestamp"].dt.year == 2024).all()
    assert (test["timestamp"].dt.year == 2025).all()


def test_time_series_cv_never_trains_on_the_future():
    train, _, _ = split_by_year(hourly_frame(), val_year=2024, test_year=2025)
    ts = train["timestamp"].to_numpy()
    for fit_idx, eval_idx in time_series_cv(5).split(train):
        assert ts[fit_idx].max() < ts[eval_idx].min()
