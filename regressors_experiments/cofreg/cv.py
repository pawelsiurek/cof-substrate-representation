"""Cross-validation folds and the evaluation loop.

`old_protocol_folds` copies `custom_data_kfold` from
precursor2cof-capacitance/modules/data/utils.py (10 folds, seed 42, quantile bins).
"""

import os
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from cofreg.data import RESULTS_DIR
from cofreg.metrics import METRICS, evaluate

N_REPEATS = 5
N_FOLDS = 10
N_BINS = 10


def target_bins(y: np.ndarray, num_bins: int = N_BINS) -> np.ndarray:
    return pd.qcut(pd.Series(y), q=num_bins, labels=False).to_numpy()


def old_protocol_folds(y: np.ndarray) -> list:
    """The single 10-fold split used in precursor2cof-capacitance."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    return list(skf.split(np.zeros(len(y)), target_bins(y)))


def repeated_folds(y: np.ndarray, n_repeats: int = N_REPEATS) -> list[list]:
    """`n_repeats` stratified 10-fold splits with seeds 0..n_repeats-1."""
    return [
        list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed).split(np.zeros(len(y)), target_bins(y)))
        for seed in range(n_repeats)
    ]


def _fit_predict(model, X: np.ndarray, y: np.ndarray, train_idx, test_idx) -> np.ndarray:
    model = clone(model)
    model.fit(X[train_idx], y[train_idx])
    return np.asarray(model.predict(X[test_idx])).ravel()


def cross_val_oof(model, X, y: np.ndarray, repeats: list[list], n_jobs: int = -1) -> np.ndarray:
    """Out-of-fold predictions, shape (n_repeats, n_samples)."""
    X = np.asarray(X, dtype=float)
    jobs = [(r, tr, te) for r, folds in enumerate(repeats) for tr, te in folds]
    preds = Parallel(n_jobs=n_jobs)(delayed(_fit_predict)(model, X, y, tr, te) for _, tr, te in jobs)
    oof = np.full((len(repeats), len(y)), np.nan)
    for (r, _, te), p in zip(jobs, preds):
        oof[r, te] = p
    return oof


def pooled_scores(y: np.ndarray, oof: np.ndarray) -> pd.DataFrame:
    """Metrics on the pooled OOF predictions of each repeat (one row per repeat)."""
    return pd.DataFrame([evaluate(y, p) for p in oof])


def per_fold_scores(y: np.ndarray, oof: np.ndarray, folds: list) -> pd.DataFrame:
    """Metrics per test fold, as in the old pipeline (one row per fold)."""
    return pd.DataFrame([evaluate(y[te], oof[te]) for _, te in folds])


def summarize(scores: pd.DataFrame) -> dict:
    out = {}
    for m in METRICS:
        out[f"{m}_mean"] = scores[m].mean()
        out[f"{m}_std"] = scores[m].std(ddof=1) if len(scores) > 1 else 0.0
    return out


def run_experiment(
    name: str,
    model,
    X,
    y: np.ndarray,
    repeats: list[list],
    experiment: str,
    features: str,
    save: bool = True,
    n_jobs: int = -1,
) -> dict:
    """Run repeated CV, save OOF predictions to results/<experiment>/<name>.csv, return a summary row."""
    start = time.time()
    oof = cross_val_oof(model, X, y, repeats, n_jobs=n_jobs)
    row = {"experiment": experiment, "model": name, "features": features, "n_features": np.asarray(X).shape[1]}
    row.update(summarize(pooled_scores(y, oof)))
    row["seconds"] = round(time.time() - start, 1)
    if save:
        out_dir = os.path.join(RESULTS_DIR, experiment)
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(oof.T, columns=[f"repeat_{r}" for r in range(len(oof))]).assign(y_true=y).to_csv(
            os.path.join(out_dir, f"{name}.csv"), index=False
        )
    return row


def run_many(configs: list[tuple[str, object, object, str]], y: np.ndarray, repeats: list[list], experiment: str, **kwargs) -> pd.DataFrame:
    """configs: list of (name, model, X, feature_label)."""
    rows = []
    for name, model, X, feats in configs:
        rows.append(run_experiment(name, model, X, y, repeats, experiment, feats, **kwargs))
        r = rows[-1]
        print(f"{name:45s} MAPE {r['mape_mean']:.3f}±{r['mape_std']:.3f}  PA {r['pa_mean']:.3f}±{r['pa_std']:.3f}  RMSE {r['rmse_mean']:.1f}±{r['rmse_std']:.1f}  MALE {r['male_mean']:.3f}  ({r['seconds']}s)")
    return pd.DataFrame(rows)


def load_oof(experiment: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Saved (y_true, oof) for a run."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, experiment, f"{name}.csv"))
    return df["y_true"].to_numpy(), df.drop(columns="y_true").to_numpy().T

