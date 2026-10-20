"""Models and the old tuning protocol.

`OLD_PARAM_GRIDS` and `OldProtocolTuned` copy `model_factory.py` and `SklearnTrainingPipeline.tune_model`
from precursor2cof-capacitance: Optuna (QMC sampler, seed 42, 100 trials) minimising RMSE on one
stratified 70/30 split of the training data, then a refit on the whole training data.
"""

import ast
import warnings

import numpy as np
import optuna
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from cofreg.cv import target_bins

optuna.logging.set_verbosity(optuna.logging.ERROR)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

OLD_PARAM_GRIDS = {
    "xgboost": {
        "n_estimators": list(range(5, 101, 5)),
        "learning_rate": [0.05, 0.10, 0.15],
        "max_depth": list(range(5, 16)) + [None],
        "min_child_weight": [3, 5, 7],
        "gamma": [0.0, 0.1, 0.2],
        "colsample_bytree": [0.2, 0.3, 0.4],
    },
    "random_forest": {
        "n_estimators": list(range(5, 101, 5)),
        "max_depth": list(range(5, 16)) + [None],
        "min_samples_split": (2, 5),
        "min_samples_leaf": (1, 5),
        "bootstrap": [True, False],
    },
    "bayes": {
        "alpha_1": [1e-6, 1e-5, 1e-4],
        "alpha_2": [1e-6, 1e-5, 1e-4],
        "lambda_1": [1e-6, 1e-5, 1e-4],
        "lambda_2": [1e-6, 1e-5, 1e-4],
    },
}


def old_base_model(name: str):
    return {
        "xgboost": XGBRegressor(random_state=42, n_jobs=1),
        "random_forest": RandomForestRegressor(random_state=42, n_jobs=1),
        "bayes": BayesianRidge(),
    }[name]


def _suggest(trial: optuna.Trial, grid: dict) -> dict:
    params = {}
    for key, value in grid.items():
        if isinstance(value, list):
            params[key] = trial.suggest_categorical(key, value)
            if key == "hidden_layer_sizes":
                params[key] = ast.literal_eval(params[key])
        elif isinstance(value, tuple):
            if isinstance(value[0], int):
                params[key] = trial.suggest_int(key, value[0], value[1])
            else:
                params[key] = trial.suggest_float(key, value[0], value[1])
        else:
            raise ValueError(f"Unsupported parameter type: {type(value)}")
    return params


class OldProtocolTuned(BaseEstimator, RegressorMixin):
    """Standard scaling + the old Optuna tuning on a single 70/30 split."""

    def __init__(self, model_name: str = "random_forest", n_trials: int = 100, num_bins: int = 10):
        self.model_name = model_name
        self.n_trials = n_trials
        self.num_bins = num_bins

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
        grid = OLD_PARAM_GRIDS[self.model_name]
        tr, va = train_test_split(np.arange(len(y)), train_size=0.7, random_state=42, shuffle=True, stratify=target_bins(y, self.num_bins))
        scaler = StandardScaler().fit(X[tr])
        X_tr, X_va = scaler.transform(X[tr]), scaler.transform(X[va])
        base = old_base_model(self.model_name)

        def objective(trial):
            model = clone(base).set_params(**_suggest(trial, grid))
            try:
                model.fit(X_tr, y[tr])
                return root_mean_squared_error(y[va], model.predict(X_va))
            except ValueError:
                return float("inf")

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.QMCSampler(seed=42))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params_ = study.best_params
        self.model_ = make_pipeline(StandardScaler(), clone(base).set_params(**self.best_params_)).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


class MedianBaseline(BaseEstimator, RegressorMixin):
    """Predicts the training-set median (the baseline used in the old pipeline)."""

    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_)


class TargetTransformed(BaseEstimator, RegressorMixin):
    """Fit `regressor` on a transformed target and map predictions back.

    transform: "log", "sqrt", "rank" (normal scores of the training ranks, inverted by interpolation)
    or "none". Predictions in transformed space are clipped to the training range, which stops
    linear models from producing huge values through exp().
    """

    def __init__(self, regressor=None, transform: str = "log"):
        self.regressor = regressor
        self.transform = transform

    def _forward(self, y):
        if self.transform == "log":
            return np.log(y)
        if self.transform == "sqrt":
            return np.sqrt(y)
        if self.transform == "rank":
            return np.interp(y, self.y_sorted_, self.z_sorted_)
        return y

    def _inverse(self, z):
        if self.transform == "log":
            return np.exp(z)
        if self.transform == "sqrt":
            return np.square(z)
        if self.transform == "rank":
            return np.interp(z, self.z_sorted_, self.y_sorted_)
        return z

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        if self.transform == "rank":
            from scipy.stats import norm, rankdata

            self.y_sorted_ = np.sort(y)
            self.z_sorted_ = np.sort(norm.ppf((rankdata(y) - 0.5) / len(y)))
        z = self._forward(y)
        self.z_min_, self.z_max_ = z.min(), z.max()
        self.regressor_ = clone(self.regressor).fit(X, z)
        return self

    def predict(self, X):
        z = np.asarray(self.regressor_.predict(X), dtype=float).ravel()
        return self._inverse(np.clip(z, self.z_min_, self.z_max_))


class InnerCVTuned(BaseEstimator, RegressorMixin):
    """Optuna (TPE) tuning with stratified inner K-fold CV on the training data, then refit.

    space: {param: ("int", lo, hi) | ("float", lo, hi) | ("logfloat", lo, hi) | ("cat", [choices])}.
    The objective is the mean absolute log error, so it fits a log-target model and also suits MAPE/PA.
    """

    def __init__(self, estimator=None, space: dict | None = None, n_trials: int = 30, inner_folds: int = 5, seed: int = 0):
        self.estimator = estimator
        self.space = space
        self.n_trials = n_trials
        self.inner_folds = inner_folds
        self.seed = seed

    def _params(self, trial):
        params = {}
        for key, spec in self.space.items():
            kind = spec[0]
            if kind == "int":
                params[key] = trial.suggest_int(key, spec[1], spec[2])
            elif kind == "float":
                params[key] = trial.suggest_float(key, spec[1], spec[2])
            elif kind == "logfloat":
                params[key] = trial.suggest_float(key, spec[1], spec[2], log=True)
            else:
                params[key] = trial.suggest_categorical(key, spec[1])
        return params

    def fit(self, X, y):
        from sklearn.model_selection import StratifiedKFold

        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float).ravel()
        folds = list(StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed).split(X, target_bins(y, 5)))

        def objective(trial):
            model = clone(self.estimator).set_params(**self._params(trial))
            errs = []
            for tr, va in folds:
                pred = np.maximum(model.fit(X[tr], y[tr]).predict(X[va]), 1e-3)
                errs.append(np.mean(np.abs(np.log(pred) - np.log(y[va]))))
            return float(np.mean(errs))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.seed))
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        self.best_params_ = study.best_params
        self.model_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))
