"""Tune each model with TimeSeriesSplit on the training years, then compare on validation.

Writes results/tuning.json with the chosen hyperparameters and validation metrics.
The test year is never loaded into a model here.
"""

import json
import time
from pathlib import Path

from sklearn.model_selection import GridSearchCV

from src.features import FEATURES, TARGET, load_features
from src.metrics import regression_metrics
from src.models import BASELINES, MODELS
from src.split import split_by_year, time_series_cv

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TUNING_PATH = RESULTS_DIR / "tuning.json"
CV_SPLITS = 3


def tune(train) -> dict:
    """Grid-search every model on the training years with expanding-window CV."""
    chosen = {}
    for name, (factory, grid) in MODELS.items():
        start = time.time()
        search = GridSearchCV(
            factory(), grid, cv=time_series_cv(CV_SPLITS),
            scoring="neg_mean_absolute_error", n_jobs=1,
        )
        search.fit(train[FEATURES], train[TARGET])
        chosen[name] = {
            "params": search.best_params_,
            "cv_mae": -search.best_score_,
            "seconds": round(time.time() - start, 1),
        }
        print(f"{name}: {search.best_params_} cv MAE={-search.best_score_:.1f}")
    return chosen


def validate(train, val, chosen: dict) -> dict:
    """Fit tuned models on the training years and score every model on the validation year."""
    scores = {name: regression_metrics(val[TARGET], fn(val)) for name, fn in BASELINES.items()}
    for name, (factory, _) in MODELS.items():
        model = factory().set_params(**chosen[name]["params"])
        model.fit(train[FEATURES], train[TARGET])
        scores[name] = regression_metrics(val[TARGET], model.predict(val[FEATURES]))
    return scores


def main():
    train, val, _ = split_by_year(load_features())
    chosen = tune(train)
    val_scores = validate(train, val, chosen)
    for name, m in val_scores.items():
        print(f"{name:24s} val MAE={m['mae']:.1f} RMSE={m['rmse']:.1f} MAPE={m['mape']:.2f}%")
    best = min(MODELS, key=lambda n: val_scores[n]["mae"])
    RESULTS_DIR.mkdir(exist_ok=True)
    TUNING_PATH.write_text(json.dumps(
        {"chosen": chosen, "validation": val_scores, "best_model_on_validation": best}, indent=2))
    print(f"best learned model on validation: {best}")


if __name__ == "__main__":
    main()
