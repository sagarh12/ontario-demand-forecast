"""Error metrics shared by training and evaluation."""

import numpy as np


def regression_metrics(actual, predicted) -> dict:
    """MAE, RMSE and bias (mean of predicted - actual) in MW and MAPE in percent, ignoring missing actuals."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    keep = ~np.isnan(actual)
    err = predicted[keep] - actual[keep]
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(np.mean(np.abs(err) / actual[keep]) * 100),
        "bias": float(np.mean(err)),
        "n": int(keep.sum()),
    }
