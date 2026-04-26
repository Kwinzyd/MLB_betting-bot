"""Lightweight Poisson GLM for per-prop count projections.

Log-link Poisson regression with L2 regularization, fit via L-BFGS-B. No heavy
deps — numpy + scipy only. An `exposure` term (projected IP or PA) enters as a
log-offset, so coefficients are interpreted as rates per unit of exposure.

Usage:
    glm = PoissonGLM(feature_names=[...])
    glm.fit(X_train, y_train, exposure_train)
    mean = glm.predict_mean(X_new_row, exposure=projected_ip)
    glm.save("models/mlb_poisson_pitcher_strikeouts.pkl")
    glm = PoissonGLM.load("models/mlb_poisson_pitcher_strikeouts.pkl")
"""
import os
import pickle
from dataclasses import dataclass, field
from typing import List

import numpy as np
from scipy.optimize import minimize


@dataclass
class PoissonGLM:
    feature_names: List[str] = field(default_factory=list)
    coef_: np.ndarray = None              # shape (n_features + 1,) — first entry is intercept
    feature_mean_: np.ndarray = None      # shape (n_features,)
    feature_std_: np.ndarray = None       # shape (n_features,)
    l2: float = 0.01
    n_iter_: int = 0
    converged_: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, exposure: np.ndarray = None,
            max_iter: int = 500):
        """
        Fit the GLM. X is (n_samples, n_features); y is integer counts.
        exposure is (n_samples,) of positive offsets — defaults to ones.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if exposure is None:
            exposure = np.ones_like(y)
        exposure = np.asarray(exposure, dtype=np.float64)
        # Guard against zero/negative exposure
        exposure = np.clip(exposure, 1e-3, None)

        # Standardize features so regularization is balanced across scales
        self.feature_mean_ = X.mean(axis=0)
        self.feature_std_ = X.std(axis=0)
        self.feature_std_[self.feature_std_ < 1e-8] = 1.0
        Xs = (X - self.feature_mean_) / self.feature_std_

        n_features = Xs.shape[1]
        # Design matrix with intercept column
        Xd = np.hstack([np.ones((Xs.shape[0], 1)), Xs])

        log_exposure = np.log(exposure)

        def nll_and_grad(beta):
            eta = Xd @ beta + log_exposure
            # Numerical-safe exp
            eta_clipped = np.clip(eta, -50.0, 50.0)
            mu = np.exp(eta_clipped)
            # Negative log-likelihood (drop constant log(y!))
            nll = np.sum(mu - y * eta_clipped)
            # L2 on non-intercept coefficients only
            reg = 0.5 * self.l2 * np.sum(beta[1:] ** 2)
            loss = nll + reg

            grad = Xd.T @ (mu - y)
            grad[1:] += self.l2 * beta[1:]
            return loss, grad

        beta0 = np.zeros(n_features + 1)
        # Warm-start intercept at log mean rate (per unit exposure)
        pos = y > 0
        if pos.any():
            beta0[0] = float(np.log(max(1e-3, (y.sum() / exposure.sum()))))

        result = minimize(
            nll_and_grad, beta0, jac=True, method="L-BFGS-B",
            options={"maxiter": max_iter, "disp": False},
        )
        self.coef_ = result.x
        self.n_iter_ = result.nit
        self.converged_ = bool(result.success)
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_mean(self, X, exposure: float | np.ndarray = 1.0) -> float | np.ndarray:
        """
        Predict expected count for a single row (1D array) or a batch (2D).
        Exposure can be a scalar or an array matching the number of samples in X.
        """
        if self.coef_ is None:
            raise RuntimeError("PoissonGLM is not fitted")
        X = np.asarray(X, dtype=np.float64)
        single = (X.ndim == 1)
        if single:
            X = X.reshape(1, -1)

        Xs = (X - self.feature_mean_) / self.feature_std_
        Xd = np.hstack([np.ones((Xs.shape[0], 1)), Xs])
        log_exposure = np.log(np.clip(np.asarray(exposure), 1e-3, None))
        eta = Xd @ self.coef_ + log_exposure
        mu = np.exp(np.clip(eta, -50.0, 50.0))
        return float(mu[0]) if single else mu

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "feature_names": self.feature_names,
                "coef_": self.coef_,
                "feature_mean_": self.feature_mean_,
                "feature_std_": self.feature_std_,
                "l2": self.l2,
                "n_iter_": self.n_iter_,
                "converged_": self.converged_,
            }, f)

    @classmethod
    def load(cls, path: str) -> "PoissonGLM":
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj = cls(feature_names=state["feature_names"], l2=state.get("l2", 0.01))
        obj.coef_ = state["coef_"]
        obj.feature_mean_ = state["feature_mean_"]
        obj.feature_std_ = state["feature_std_"]
        obj.n_iter_ = state.get("n_iter_", 0)
        obj.converged_ = state.get("converged_", True)
        return obj
