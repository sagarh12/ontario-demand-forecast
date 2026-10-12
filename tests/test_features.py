import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES, TARGET, build_features


def synthetic_hourly(days: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=days * 24, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        TARGET: 15000 + 2000 * np.sin(2 * np.pi * ts.hour / 24) + rng.normal(0, 300, len(ts)),
        "market_demand": 0.0,
        "temp": rng.normal(5, 8, len(ts)),
        "humidity": rng.uniform(40, 90, len(ts)),
        "wind": rng.uniform(0, 30, len(ts)),
    })


@pytest.mark.parametrize("target_day", ["2024-01-15", "2024-01-29", "2024-02-08"])
def test_features_ignore_demand_after_issue_time(target_day):
    """Scrambling every demand value from the start of D+1 onward must not change
    any feature for D+1: the forecast is issued at the end of day D."""
    raw = synthetic_hourly()
    target_start = pd.Timestamp(target_day)

    tampered = raw.copy()
    future = tampered["timestamp"] >= target_start
    tampered.loc[future, TARGET] = np.random.default_rng(1).uniform(0, 1e6, future.sum())

    def day_rows(df):
        feats = build_features(df)
        return feats[feats["timestamp"].dt.normalize() == target_start][FEATURES].reset_index(drop=True)

    pd.testing.assert_frame_equal(day_rows(raw), day_rows(tampered))


def test_target_is_not_a_feature():
    assert TARGET not in FEATURES
    assert "market_demand" not in FEATURES


def test_lags_point_at_the_right_hours():
    feats = build_features(synthetic_hourly()).set_index("timestamp")
    raw = synthetic_hourly().set_index("timestamp")[TARGET]
    t = pd.Timestamp("2024-01-20 13:00")
    assert feats.loc[t, "lag_24"] == raw[t - pd.Timedelta(hours=24)]
    assert feats.loc[t, "lag_168"] == raw[t - pd.Timedelta(hours=168)]
    assert feats.loc[t, "prev_day_mean"] == pytest.approx(raw["2024-01-19"].mean())
    assert feats.loc[t, "prev_7d_mean"] == pytest.approx(raw["2024-01-13":"2024-01-19"].mean())


def test_missing_demand_is_filled_only_from_the_past():
    raw = synthetic_hourly()
    gap = pd.Timestamp("2024-01-20 23:00")
    raw.loc[raw["timestamp"] == gap, TARGET] = np.nan
    feats = build_features(raw).set_index("timestamp")
    before = raw.set_index("timestamp").loc[gap - pd.Timedelta(hours=1), TARGET]
    assert feats.loc[gap + pd.Timedelta(hours=24), "lag_24"] == before
    assert np.isnan(feats.loc[gap, TARGET])


def test_holiday_flags():
    feats = build_features(synthetic_hourly()).set_index("timestamp")
    assert feats.loc["2024-02-19 12:00", "is_holiday"] == 1  # Family Day (Ontario)
    assert feats.loc["2024-02-18 12:00", "is_day_before_holiday"] == 1
    assert feats.loc["2024-02-20 12:00", "is_day_after_holiday"] == 1
