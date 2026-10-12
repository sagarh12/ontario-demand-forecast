"""Feature engineering for day-ahead hourly forecasting.

The forecast for every hour of day D+1 is issued at the end of day D, i.e. after
hour D 23:00 is known. Every demand-derived feature for a target hour on D+1 uses
only demand up to and including D 23:00. Weather is the actual value at the target
hour, standing in for a day-ahead weather forecast.
"""

import numpy as np
import pandas as pd

from src.data import PROCESSED_PATH, ontario_holidays

TARGET = "ontario_demand"
HDD_CDD_BASE_C = 18.0

CALENDAR_FEATURES = [
    "hour", "day_of_week", "month", "day_of_year", "is_weekend",
    "is_holiday", "is_day_before_holiday", "is_day_after_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
]
LAG_FEATURES = ["lag_24", "lag_48", "lag_168"]
ROLLING_FEATURES = ["prev_day_mean", "prev_day_std", "prev_7d_mean", "prev_7d_std"]
WEATHER_FEATURES = ["temp", "humidity", "wind", "hdd", "cdd"]
FEATURES = CALENDAR_FEATURES + LAG_FEATURES + ROLLING_FEATURES + WEATHER_FEATURES


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    dates = ts.dt.date
    hols = ontario_holidays(ts.dt.year.min(), ts.dt.year.max())
    one_day = pd.Timedelta(days=1)
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_holiday"] = dates.isin(hols).astype(int)
    df["is_day_before_holiday"] = (ts + one_day).dt.date.isin(hols).astype(int)
    df["is_day_after_holiday"] = (ts - one_day).dt.date.isin(hols).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    return df


def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags of at least 24h and daily statistics of days before the target day.

    Missing demand hours are forward-filled for feature building only, so a filled
    value never depends on a later hour. The target column is left as NaN.
    """
    demand = df[TARGET].ffill()
    df["lag_24"] = demand.shift(24)
    df["lag_48"] = demand.shift(48)
    df["lag_168"] = demand.shift(168)

    # Rolling windows read at 23:00 of day D (the issue time) cover day D and
    # days D-6..D, and are attached to every hour of target day D+1.
    issue = df["timestamp"].dt.hour == 23
    at_issue = pd.DataFrame({
        "prev_day_mean": demand.rolling(24).mean(),
        "prev_day_std": demand.rolling(24).std(),
        "prev_7d_mean": demand.rolling(168).mean(),
        "prev_7d_std": demand.rolling(168).std(),
    })[issue.to_numpy()]
    at_issue.index = df.loc[issue, "timestamp"].dt.normalize() + pd.Timedelta(days=1)
    return df.join(at_issue, on=df["timestamp"].dt.normalize())


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df["hdd"] = (HDD_CDD_BASE_C - df["temp"]).clip(lower=0)
    df["cdd"] = (df["temp"] - HDD_CDD_BASE_C).clip(lower=0)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a complete hourly frame with all FEATURES added.

    Rows whose features cannot be computed (the first week) are dropped.
    Rows with a missing target are kept so callers decide how to treat them.
    """
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    expected = pd.date_range(df["timestamp"].iloc[0], periods=len(df), freq="h")
    if not (df["timestamp"].to_numpy() == expected.to_numpy()).all() or expected[0].hour != 0:
        raise ValueError("Input must be a gap-free hourly series starting at midnight")
    df = add_calendar_features(df)
    df = add_demand_features(df)
    df = add_weather_features(df)
    return df.dropna(subset=FEATURES).reset_index(drop=True)


def load_features() -> pd.DataFrame:
    return build_features(pd.read_parquet(PROCESSED_PATH))
