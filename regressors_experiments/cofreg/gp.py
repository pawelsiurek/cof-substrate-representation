"""Gaussian-process kernels for fingerprints and mixed fingerprint + descriptor inputs.

Both kernels work on a column slice of X, so one GP can combine a Tanimoto kernel on
fingerprint columns with an RBF kernel on descriptor columns.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Kernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


def _cols(X, cols):
    return X if cols is None else X[:, cols[0] : cols[1]]


class Tanimoto(Kernel):
    """MinMax (generalised Tanimoto) kernel for non-negative count vectors. No hyperparameters."""

    def __init__(self, cols: tuple[int, int] | None = None):
        self.cols = cols

    def __call__(self, X, Y=None, eval_gradient=False):
        A = _cols(np.atleast_2d(X), self.cols)
        B = A if Y is None else _cols(np.atleast_2d(Y), self.cols)
        num = np.minimum(A[:, None, :], B[None, :, :]).sum(-1)
        den = np.maximum(A[:, None, :], B[None, :, :]).sum(-1)
        K = np.where(den > 0, num / np.where(den > 0, den, 1), 1.0)
        if eval_gradient:
            return K, np.empty((K.shape[0], K.shape[1], 0))
        return K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False


class SliceRBF(RBF):
    """RBF kernel applied to a column slice."""

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-2, 1e3), cols: tuple[int, int] | None = None):
        super().__init__(length_scale=length_scale, length_scale_bounds=length_scale_bounds)
        self.cols = cols

    def __call__(self, X, Y=None, eval_gradient=False):
        return super().__call__(_cols(X, self.cols), None if Y is None else _cols(Y, self.cols), eval_gradient)

    def diag(self, X):
        return np.ones(X.shape[0])


class TanimotoGP(BaseEstimator, RegressorMixin):
    """GP on [fingerprint counts | descriptors].

    The first `n_fp` columns go to a Tanimoto kernel; the remaining columns (if any) are standardised
    and go to an RBF kernel (ARD when `ard`). The target is standardised inside the GP.
    """

    def __init__(self, n_fp: int | None = None, ard: bool = False, n_restarts: int = 3, random_state: int = 0):
        self.n_fp = n_fp
        self.ard = ard
        self.n_restarts = n_restarts
        self.random_state = random_state

    def _kernel(self, n_cols: int):
        n_fp = n_cols if self.n_fp is None else self.n_fp
        kernel = ConstantKernel(1.0, (1e-3, 1e2)) * Tanimoto(cols=(0, n_fp)) if n_fp > 0 else None
        if n_cols > n_fp:
            ls = np.ones(n_cols - n_fp) if self.ard else 1.0
            rbf = ConstantKernel(1.0, (1e-3, 1e2)) * SliceRBF(length_scale=ls, cols=(n_fp, n_cols))
            kernel = rbf if kernel is None else kernel + rbf
        return kernel + WhiteKernel(0.1, (1e-4, 1e1))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        n_fp = X.shape[1] if self.n_fp is None else self.n_fp
        self.scaler_ = StandardScaler().fit(X[:, n_fp:]) if X.shape[1] > n_fp else None
        self.gp_ = GaussianProcessRegressor(
            self._kernel(X.shape[1]), normalize_y=True, n_restarts_optimizer=self.n_restarts, random_state=self.random_state
        ).fit(self._prep(X), y)
        return self

    def _prep(self, X):
        X = np.asarray(X, dtype=float)
        if self.scaler_ is None:
            return X
        n_fp = X.shape[1] - self.scaler_.n_features_in_
        return np.hstack([X[:, :n_fp], self.scaler_.transform(X[:, n_fp:])])

    def predict(self, X, return_std=False):
        return self.gp_.predict(self._prep(X), return_std=return_std)


class TanimotoKernelModel(BaseEstimator, RegressorMixin):
    """Any sklearn model with kernel="precomputed" (SVR, KernelRidge) on a Tanimoto kernel of X."""

    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        from sklearn.base import clone

        self.X_ = np.asarray(X, dtype=float)
        self.model_ = clone(self.model).fit(Tanimoto()(self.X_), y)
        return self

    def predict(self, X):
        return self.model_.predict(Tanimoto()(np.asarray(X, dtype=float), self.X_))
