"""Final, one-time evaluation on the test year.

Refits each tuned model on train + validation years, predicts the test year, and writes
results/metrics.json, results/predictions.parquet, results/summary.md and
results/figures/*.png. The README results block is regenerated from metrics.json.
"""

import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features import FEATURES, TARGET, load_features
from src.metrics import regression_metrics
from src.models import BASELINES, MODELS, SEED
from src.split import TEST_YEAR, split_by_year
from src.train import RESULTS_DIR, TUNING_PATH

FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_PATH = RESULTS_DIR / "metrics.json"
PREDICTIONS_PATH = RESULTS_DIR / "predictions.parquet"
SUMMARY_PATH = RESULTS_DIR / "summary.md"
README_PATH = RESULTS_DIR.parent / "README.md"
N_EXTREME_DAYS = 10

LABELS = {
    "baseline_yesterday": "Baseline: same hour yesterday",
    "baseline_last_week": "Baseline: same hour last week",
    "baseline_hybrid": "Baseline: hybrid",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "HistGradientBoosting",
}


def predict_test(fit_df, test, tuning) -> tuple[pd.DataFrame, dict]:
    """Predictions for every baseline and refitted model on the test year."""
    preds = test[["timestamp", TARGET, "hour", "month", "day_of_week",
                  "is_weekend", "is_holiday", "temp"]].rename(columns={TARGET: "actual"})
    for name, fn in BASELINES.items():
        preds[name] = fn(test).to_numpy()
    fitted = {}
    for name, (factory, _) in MODELS.items():
        model = factory().set_params(**tuning["chosen"][name]["params"])
        model.fit(fit_df[FEATURES], fit_df[TARGET])
        preds[name] = model.predict(test[FEATURES])
        fitted[name] = model
    return preds.reset_index(drop=True), fitted


def mae_by(preds: pd.DataFrame, models: list, key) -> dict:
    scored = preds.dropna(subset=["actual"])
    grouped = scored.groupby(key)
    return {
        m: {str(k): float(v) for k, v in grouped.apply(lambda g: (g[m] - g["actual"]).abs().mean()).items()}
        for m in models
    }


def extreme_days(preds: pd.DataFrame, models: list) -> dict:
    """Error on the hottest and coldest test days, with bias (mean of predicted - actual)."""
    daily = preds.assign(date=preds["timestamp"].dt.date).groupby("date")["temp"].agg(["max", "min"])
    hot = set(daily["max"].nlargest(N_EXTREME_DAYS).index)
    cold = set(daily["min"].nsmallest(N_EXTREME_DAYS).index)
    out = {}
    for label, days in [("heat_wave_days", hot), ("cold_snap_days", cold)]:
        sub = preds[preds["timestamp"].dt.date.isin(days)].dropna(subset=["actual"])
        out[label] = {
            "dates": sorted(str(d) for d in days),
            "mean_temp_c": float(sub["temp"].mean()),
            "mae": {m: float((sub[m] - sub["actual"]).abs().mean()) for m in models},
            "bias": {m: float((sub[m] - sub["actual"]).mean()) for m in models},
            "peak_actual_mw": float(sub["actual"].max()),
            "peak_predicted_mw": {m: float(sub[m].max()) for m in models},
        }
    return out


def worst_days(preds: pd.DataFrame, model: str, n: int = 5) -> list:
    scored = preds.dropna(subset=["actual"]).assign(date=lambda d: d["timestamp"].dt.date)
    err = scored.assign(abs_err=(scored[model] - scored["actual"]).abs(), err=scored[model] - scored["actual"])
    daily = err.groupby("date").agg(mae=("abs_err", "mean"), bias=("err", "mean"),
                                   max_temp=("temp", "max"), min_temp=("temp", "min"))
    return [{"date": str(d), **{k: float(v) for k, v in row.items()}}
            for d, row in daily.nlargest(n, "mae").iterrows()]


def importance(model, test: pd.DataFrame) -> dict:
    scored = test.dropna(subset=[TARGET])
    result = permutation_importance(model, scored[FEATURES], scored[TARGET],
                                    scoring="neg_mean_absolute_error", n_repeats=5,
                                    random_state=SEED, n_jobs=1)
    order = np.argsort(-result.importances_mean)
    return {FEATURES[i]: {"mae_increase": float(result.importances_mean[i]),
                          "std": float(result.importances_std[i])} for i in order}


def plot_week(preds, start, models, title, path):
    week = preds[(preds["timestamp"] >= start) & (preds["timestamp"] < start + pd.Timedelta(days=7))]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(week["timestamp"], week["actual"], color="black", lw=2, label="Actual")
    for m in models:
        ax.plot(week["timestamp"], week[m], lw=1.3, label=LABELS[m])
    ax.set(title=title, ylabel="Ontario demand (MW)")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def make_figures(preds, metrics, best, best_baseline):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    daily = preds.assign(date=preds["timestamp"].dt.normalize()).groupby("date")["temp"].agg(["min", "max"])
    for label, day in [("winter", daily["min"].idxmin()), ("summer", daily["max"].idxmax())]:
        start = day - pd.Timedelta(days=int(day.dayofweek))
        plot_week(preds, start, [best, best_baseline],
                  f"{label.title()} week of {start.date()} (contains the {'coldest' if label == 'winter' else 'hottest'} test day)",
                  FIGURES_DIR / f"week_{label}.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    for m in [best, best_baseline]:
        hours = metrics["by_hour"][m]
        ax.plot([int(h) for h in hours], list(hours.values()), marker="o", label=LABELS[m])
    ax.set(title=f"Test MAE by hour of day ({TEST_YEAR}, hour starting, EST)", xlabel="Hour", ylabel="MAE (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "error_by_hour.png", dpi=120)
    plt.close(fig)

    imp = metrics["permutation_importance"]
    top = list(imp)[:15][::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top, [imp[f]["mae_increase"] for f in top], xerr=[imp[f]["std"] for f in top])
    ax.set(title=f"Permutation importance, {LABELS[best]} (test {TEST_YEAR})", xlabel="MAE increase when shuffled (MW)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)

    scored = preds.dropna(subset=["actual"])
    fig, ax = plt.subplots(figsize=(8, 4))
    for m in [best_baseline, best]:
        ax.hist(scored[m] - scored["actual"], bins=80, alpha=0.55, label=LABELS[m])
    ax.axvline(0, color="black", lw=1)
    ax.set(title=f"Residuals, predicted - actual (test {TEST_YEAR})", xlabel="MW", ylabel="Hours")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "residual_histogram.png", dpi=120)
    plt.close(fig)


def summary_table(metrics) -> str:
    rows = ["| Model | MAE (MW) | RMSE (MW) | MAPE (%) | Skill vs best baseline |", "|---|---:|---:|---:|---:|"]
    for m, s in sorted(metrics["overall"].items(), key=lambda kv: kv[1]["mae"]):
        name = LABELS[m] + (" (selected on validation)" if m == metrics["best_model"] else "")
        rows.append(f"| {name} | {s['mae']:.0f} | {s['rmse']:.0f} | {s['mape']:.2f} | {metrics['skill'][m] * 100:+.1f}% |")
    note = (f"\nTest year {TEST_YEAR}, {metrics['overall'][metrics['best_model']]['n']} hours. "
            f"Best baseline on the test year: {LABELS[metrics['best_baseline']]}. "
            "Skill = 1 - MAE_model / MAE_best_baseline.\n")
    return "\n".join(rows) + "\n" + note


def details_tables(metrics) -> str:
    """Breakdown, hardest-day and importance tables for the README."""
    best, base = metrics["best_model"], metrics["best_baseline"]
    shown = [best, "random_forest" if best != "random_forest" else "hist_gradient_boosting", "ridge", base]
    shown = list(dict.fromkeys(shown))
    head = "| | " + " | ".join(LABELS[m] for m in shown) + " |\n|---|" + "---:|" * len(shown) + "\n"

    def row(label, values):
        return f"| {label} | " + " | ".join(values) + " |\n"

    seg = head + row("Mean bias, whole year (pred - actual)", [f"{metrics['overall'][m]['bias']:+.0f}" for m in shown])
    for key, groups in [("weekday_vs_weekend", ["weekday", "weekend"]),
                        ("holiday_vs_non_holiday", ["non_holiday", "holiday"])]:
        for g in groups:
            seg += row(f"MAE, {g.replace('_', '-')}", [f"{metrics[key][m][g]:.0f}" for m in shown])
    hour = metrics["by_hour"][best]
    month = metrics["by_month"][best]

    ext = head
    for label, key in [("Heat-wave days", "heat_wave_days"), ("Cold-snap days", "cold_snap_days")]:
        e = metrics["extreme_days"][key]
        ext += row(f"{label}: MAE", [f"{e['mae'][m]:.0f}" for m in shown])
        ext += row(f"{label}: bias (pred - actual)", [f"{e['bias'][m]:+.0f}" for m in shown])
        ext += row(f"{label}: highest prediction (actual peak {e['peak_actual_mw']:.0f})",
                   [f"{e['peak_predicted_mw'][m]:.0f}" for m in shown])
    hot, cold = metrics["extreme_days"]["heat_wave_days"], metrics["extreme_days"]["cold_snap_days"]

    worst = "| Date | MAE | Bias | Max temp (°C) |\n|---|---:|---:|---:|\n" + "".join(
        f"| {d['date']} | {d['mae']:.0f} | {d['bias']:+.0f} | {d['max_temp']:.1f} |\n"
        for d in metrics["worst_days_best_model"])

    imp = "| Feature | MAE increase when shuffled (MW) |\n|---|---:|\n" + "".join(
        f"| `{f}` | {v['mae_increase']:.0f} ± {v['std']:.0f} |\n"
        for f, v in list(metrics["permutation_importance"].items())[:10])

    val = "| Model | Validation MAE (2024) | Test MAE (2025) |\n|---|---:|---:|\n" + "".join(
        f"| {LABELS[m]} | {metrics['validation'][m]['mae']:.0f} | {metrics['overall'][m]['mae']:.0f} |\n"
        for m in metrics["overall"])

    return (
        f"**Validation vs test.** Models were selected on 2024 and scored once on 2025.\n\n{val}\n"
        f"**By segment** (test MAE, MW):\n\n{seg}\n"
        f"By hour, {LABELS[best]} is best at hour {min(hour, key=hour.get)} ({min(hour.values()):.0f} MW) and worst at "
        f"hour {max(hour, key=hour.get)} ({max(hour.values()):.0f} MW). By month it is best in month "
        f"{min(month, key=month.get)} ({min(month.values()):.0f} MW) and worst in month "
        f"{max(month, key=month.get)} ({max(month.values()):.0f} MW).\n\n"
        f"**Hardest days.** The {N_EXTREME_DAYS} hottest test days by daily max temperature "
        f"(mean {hot['mean_temp_c']:.1f} °C) and the {N_EXTREME_DAYS} coldest by daily min temperature "
        f"(mean {cold['mean_temp_c']:.1f} °C):\n\n{ext}\n"
        f"Worst five days for {LABELS[best]}:\n\n{worst}\n"
        f"**Permutation importance** ({LABELS[best]}, test year, top 10):\n\n{imp}"
    )


def update_readme(table: str, details: str):
    if not README_PATH.exists():
        return
    text = README_PATH.read_text()
    for tag, body in [("RESULTS", table), ("DETAILS", details)]:
        block = f"<!-- {tag}:START -->\n{body}<!-- {tag}:END -->"
        text = re.sub(rf"<!-- {tag}:START -->.*?<!-- {tag}:END -->", lambda _: block, text, flags=re.S)
    README_PATH.write_text(text)


def main():
    tuning = json.loads(TUNING_PATH.read_text())
    train, val, test = split_by_year(load_features())
    preds, fitted = predict_test(pd.concat([train, val]), test, tuning)
    models = list(BASELINES) + list(MODELS)

    overall = {m: regression_metrics(preds["actual"], preds[m]) for m in models}
    best_baseline = min(BASELINES, key=lambda m: overall[m]["mae"])
    best = tuning["best_model_on_validation"]
    metrics = {
        "test_year": TEST_YEAR,
        "fit_years": f"{train['timestamp'].dt.year.min()}-{val['timestamp'].dt.year.max()}",
        "best_model": best,
        "best_baseline": best_baseline,
        "overall": overall,
        "skill": {m: 1 - overall[m]["mae"] / overall[best_baseline]["mae"] for m in models},
        "validation": tuning["validation"],
        "hyperparameters": {m: tuning["chosen"][m]["params"] for m in MODELS},
        "by_hour": mae_by(preds, models, "hour"),
        "by_month": mae_by(preds, models, "month"),
        "weekday_vs_weekend": mae_by(preds, models, preds["is_weekend"].map({0: "weekday", 1: "weekend"})),
        "holiday_vs_non_holiday": mae_by(preds, models, preds["is_holiday"].map({0: "non_holiday", 1: "holiday"})),
        "extreme_days": extreme_days(preds, models),
        "worst_days_best_model": worst_days(preds, best),
        "permutation_importance": importance(fitted[best], test),
    }

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    preds.to_parquet(PREDICTIONS_PATH, index=False)
    make_figures(preds, metrics, best, best_baseline)
    table = summary_table(metrics)
    details = details_tables(metrics)
    SUMMARY_PATH.write_text(table + "\n" + details)
    update_readme(table, details)
    print(table)


if __name__ == "__main__":
    main()
