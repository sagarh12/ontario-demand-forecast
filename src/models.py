"""Naive baselines and scikit-learn model definitions."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FEATURES

ONE_HOT = ["hour", "day_of_week", "month"]
SEED = 42


def same_hour_yesterday(df: pd.DataFrame) -> pd.Series:
    return df["lag_24"]


def same_hour_last_week(df: pd.DataFrame) -> pd.Series:
    return df["lag_168"]


def hybrid(df: pd.DataFrame) -> pd.Series:
    """Last week's value on Sat, Sun and Mon (yesterday was a different kind of day), yesterday's otherwise."""
    use_last_week = df["day_of_week"].isin([5, 6, 0])
    return df["lag_168"].where(use_last_week, df["lag_24"])


BASELINES = {
    "baseline_yesterday": same_hour_yesterday,
    "baseline_last_week": same_hour_last_week,
    "baseline_hybrid": hybrid,
}


def make_ridge() -> Pipeline:
    """Ridge with one-hot hour/day/month and standardised numeric features.

    One-hot encoding lets a linear model learn a separate level per hour and weekday,
    which a single numeric 'hour' column cannot express.
    """
    numeric = [f for f in FEATURES if f not in ONE_HOT]
    pre = ColumnTransformer([
        ("onehot", OneHotEncoder(handle_unknown="ignore"), ONE_HOT),
        ("scale", StandardScaler(), numeric),
    ])
    return Pipeline([("pre", pre), ("model", Ridge())])


def make_random_forest() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=SEED)


def make_hist_gb() -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(early_stopping=False, random_state=SEED)


# Small grids: tuning is deliberately light.
MODELS = {
    "ridge": (make_ridge, {"model__alpha": [0.1, 1.0, 10.0, 100.0]}),
    "random_forest": (make_random_forest, {
        "min_samples_leaf": [1, 5],
        "max_features": [0.33, 0.6],
    }),
    "hist_gradient_boosting": (make_hist_gb, {
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [31, 63],
        "max_iter": [300, 600],
    }),
}
