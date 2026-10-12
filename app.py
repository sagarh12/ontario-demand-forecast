"""Streamlit dashboard. Reads saved results only; nothing is trained here."""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS = Path(__file__).parent / "results"
LABELS = {
    "baseline_yesterday": "Baseline: same hour yesterday",
    "baseline_last_week": "Baseline: same hour last week",
    "baseline_hybrid": "Baseline: hybrid",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}

st.set_page_config(page_title="Ontario demand forecast", layout="wide")


@st.cache_data
def load():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    preds = pd.read_parquet(RESULTS / "predictions.parquet")
    return metrics, preds


metrics, preds = load()
models = list(metrics["overall"])
learned = [m for m in models if not m.startswith("baseline")]
baselines = [m for m in models if m.startswith("baseline")]

with st.sidebar:
    st.header("What this shows")
    st.markdown(
        f"Day-ahead forecasts of Ontario's hourly electricity demand (IESO) for the test year "
        f"**{metrics['test_year']}**, issued at the end of the previous day. Models were tuned on "
        f"earlier years and refit on {metrics['fit_years']} before this one evaluation.\n\n"
        "**Limits**\n"
        "- Uses *actual* weather as a stand-in for a weather forecast, so errors are optimistic.\n"
        "- One province-wide series; no outage, price, or generation data.\n"
        "- Weather is a simple average of Toronto, Ottawa and Windsor.\n"
        "- Timestamps are hour-starting, Eastern Standard Time (no daylight saving)."
    )
    model = st.selectbox("Model", learned, index=learned.index(metrics["best_model"]),
                         format_func=LABELS.get)
    baseline = st.selectbox("Baseline", baselines, index=baselines.index(metrics["best_baseline"]),
                            format_func=LABELS.get)

st.title("Ontario day-ahead electricity demand forecast")

first, last = preds["timestamp"].min().date(), preds["timestamp"].max().date()
picked = st.date_input("Date range (test year)", value=(pd.Timestamp("2025-07-07").date(),
                       pd.Timestamp("2025-07-20").date()), min_value=first, max_value=last)
if len(picked) != 2:
    st.info("Pick an end date.")
    st.stop()
start, end = picked
window = preds[(preds["timestamp"].dt.date >= start) & (preds["timestamp"].dt.date <= end)]
chart = window.set_index("timestamp")[["actual", model, baseline]].rename(
    columns={"actual": "Actual", model: LABELS[model], baseline: LABELS[baseline]})
st.line_chart(chart, y_label="MW")

st.subheader(f"Test-year error, {metrics['test_year']}")
table = pd.DataFrame(metrics["overall"]).T[["mae", "rmse", "mape"]]
table["skill_vs_best_baseline_%"] = pd.Series(metrics["skill"]) * 100
table.index = table.index.map(LABELS)
st.dataframe(table.sort_values("mae").style.format("{:.1f}"), width="stretch")

left, right = st.columns(2)
with left:
    st.subheader("MAE by hour of day")
    by_hour = pd.DataFrame({LABELS[m]: metrics["by_hour"][m] for m in [model, baseline]})
    by_hour.index = by_hour.index.astype(int)
    st.line_chart(by_hour.sort_index(), y_label="MAE (MW)")
with right:
    st.subheader("MAE by month")
    by_month = pd.DataFrame({LABELS[m]: metrics["by_month"][m] for m in [model, baseline]})
    by_month.index = by_month.index.astype(int)
    st.bar_chart(by_month.sort_index(), stack=False, y_label="MAE (MW)")

st.subheader(f"Permutation importance ({LABELS[metrics['best_model']]})")
imp = pd.Series({k: v["mae_increase"] for k, v in metrics["permutation_importance"].items()})
st.bar_chart(imp.head(15).sort_values(), horizontal=True, x_label="MAE increase when shuffled (MW)")
