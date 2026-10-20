"""Evaluation metrics.

Copied from precursor2cof-capacitance/modules/training_and_evaluation/evaluation_metrics.py.
`pairwise_accuracy` is a vectorised version of `average_ranking_score` with the same result.
"""

import numpy as np
import sklearn.metrics


def pairwise_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of ordered pairs with different true values whose predicted order matches.

    Same as the original: for every ordered pair (i, j) with y_true[i] != y_true[j],
    the pair is correct when (y_true[i] > y_true[j]) == (y_pred[i] > y_pred[j]).
    """
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    true_gt = y_true[:, None] > y_true[None, :]
    pred_gt = y_pred[:, None] > y_pred[None, :]
    valid = y_true[:, None] != y_true[None, :]
    return float((true_gt == pred_gt)[valid].mean())


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), 1e-10)
    return float(np.mean(2 * np.abs(y_pred - y_true) / denom))


def male(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute log error: exp(male) is the typical multiplicative error."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(np.log(np.maximum(y_pred, 1e-3)) - np.log(y_true))))


def mape_ge(threshold: float):
    """MAPE on samples with y_true >= threshold (the 8 targets below 20 dominate plain MAPE)."""

    def score(y_true, y_pred):
        y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        keep = y_true >= threshold
        return sklearn.metrics.mean_absolute_percentage_error(y_true[keep], y_pred[keep])

    return score


METRICS = {
    "mape": lambda t, p: sklearn.metrics.mean_absolute_percentage_error(t, p),
    "pa": pairwise_accuracy,
    "rmse": lambda t, p: sklearn.metrics.root_mean_squared_error(t, p),
    "smape": smape,
    "mae": lambda t, p: sklearn.metrics.mean_absolute_error(t, p),
    "r2": lambda t, p: sklearn.metrics.r2_score(t, p),
    "male": male,
    "mape_ge20": mape_ge(20),
}
# True when a higher value is better.
HIGHER_IS_BETTER = {"mape": False, "pa": True, "rmse": False, "smape": False, "mae": False, "r2": True, "male": False, "mape_ge20": False}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, metrics=tuple(METRICS)) -> dict:
    return {m: METRICS[m](y_true, y_pred) for m in metrics}
