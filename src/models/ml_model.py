"""ML models for per-prop count projections.

Two model classes:
  PoissonGLM   — lightweight log-link Poisson GLM, numpy+scipy only, used as
                 fallback when LightGBM is unavailable or data is too sparse.
  LGBMPoissonModel — LightGBM with Poisson objective; richer non-linear interactions,
                 champion model when lightgbm is installed. Exposes SHAP values
                 for feature attribution.

Both classes implement the same interface:
    model.fit(X, y, exposure)
    mean = model.predict_mean(x_row, exposure=ip_or_pa)
    model.save(path)
    model = ModelClass.load(path)
"""
import os
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

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
            options={"maxiter": max_iter},
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


# ---------------------------------------------------------------------------
# LightGBM Poisson model (champion model when lightgbm is installed)
# ---------------------------------------------------------------------------

class LGBMPoissonModel:
    """LightGBM with Poisson objective for count projections.

    Advantages over PoissonGLM:
      - Non-linear feature interactions (e.g. whiff_pct × park_factor)
      - Built-in feature importance via SHAP TreeExplainer
      - Native Poisson log-loss; log-offset handled via init_score

    The exposure (IP or PA) enters as an additive log-offset on the raw score,
    mirroring the GLM's log-link design: predicted_mean = exp(f(x) + log(exposure)).

    Falls back to PoissonGLM transparently if lightgbm is not installed.
    """

    def __init__(self, feature_names: List[str] = None,
                 n_estimators: int = 300,
                 max_depth: int = 6,
                 learning_rate: float = 0.05,
                 num_leaves: int = 31,
                 min_child_samples: int = 20,
                 reg_lambda: float = 1.0):
        self.feature_names = feature_names or []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self._model = None          # lgb.Booster after fit
        self.converged_: bool = False
        self.n_iter_: int = 0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, exposure: Optional[np.ndarray] = None,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            exposure_val: Optional[np.ndarray] = None) -> "LGBMPoissonModel":
        try:
            import lightgbm as lgb
        except ImportError:
            raise RuntimeError(
                "lightgbm is not installed. Run: pip install lightgbm"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if exposure is None:
            exposure = np.ones(len(y))
        exposure = np.clip(np.asarray(exposure, dtype=np.float64), 1e-3, None)

        # Log-offset: shift the initial score so LightGBM's Poisson loss sees
        # rate = count / exposure rather than raw count.
        init_score = np.log(exposure)

        params = {
            "objective":         "poisson",
            "metric":            "poisson",
            "max_depth":         self.max_depth,
            "num_leaves":        self.num_leaves,
            "learning_rate":     self.learning_rate,
            "min_child_samples": self.min_child_samples,
            "reg_lambda":        self.reg_lambda,
            "n_jobs":            -1,
            "verbose":           -1,
            "seed":              42,
        }

        train_ds = lgb.Dataset(
            X, label=y, init_score=init_score,
            feature_name=self.feature_names or "auto",
        )

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
        valid_sets = [train_ds]
        if X_val is not None and y_val is not None:
            if exposure_val is None:
                exposure_val = np.ones(len(y_val))
            exposure_val = np.clip(np.asarray(exposure_val, dtype=np.float64), 1e-3, None)
            val_ds = lgb.Dataset(
                X_val, label=y_val, init_score=np.log(exposure_val), reference=train_ds,
            )
            valid_sets = [train_ds, val_ds]

        self._model = lgb.train(
            params,
            train_ds,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            callbacks=callbacks,
        )
        self.n_iter_ = self._model.num_trees()
        self.converged_ = True
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_mean(self, X, exposure: float | np.ndarray = 1.0) -> float | np.ndarray:
        if self._model is None:
            raise RuntimeError("LGBMPoissonModel is not fitted")
        X = np.asarray(X, dtype=np.float64)
        single = (X.ndim == 1)
        if single:
            X = X.reshape(1, -1)
        exposure_arr = np.clip(np.asarray(exposure, dtype=np.float64).ravel(), 1e-3, None)
        if len(exposure_arr) == 1:
            exposure_arr = np.full(len(X), exposure_arr[0])
        # raw_score is log(rate); multiply rate by exposure to get count
        raw = self._model.predict(X, raw_score=True)
        mu = np.exp(raw) * exposure_arr
        return float(mu[0]) if single else mu

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        """Return SHAP values array (n_samples, n_features). Requires shap package."""
        try:
            import shap
        except ImportError:
            raise RuntimeError("shap is not installed. Run: pip install shap")
        if self._model is None:
            raise RuntimeError("LGBMPoissonModel is not fitted")
        explainer = shap.TreeExplainer(self._model)
        return explainer.shap_values(np.asarray(X, dtype=np.float64))

    # ------------------------------------------------------------------
    # Persistence — LightGBM native format (not pickle)
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        import lightgbm as lgb  # noqa: F401 (import just for type check)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model_path = path.replace(".pkl", ".lgb")
        self._model.save_model(model_path)
        meta = {
            "feature_names":   self.feature_names,
            "n_estimators":    self.n_estimators,
            "max_depth":       self.max_depth,
            "learning_rate":   self.learning_rate,
            "num_leaves":      self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "reg_lambda":      self.reg_lambda,
            "n_iter_":         self.n_iter_,
            "model_path":      model_path,
        }
        with open(path, "wb") as f:
            pickle.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> "LGBMPoissonModel":
        import lightgbm as lgb
        with open(path, "rb") as f:
            meta = pickle.load(f)
        obj = cls(
            feature_names=meta.get("feature_names", []),
            n_estimators=meta.get("n_estimators", 300),
            max_depth=meta.get("max_depth", 6),
            learning_rate=meta.get("learning_rate", 0.05),
            num_leaves=meta.get("num_leaves", 31),
            min_child_samples=meta.get("min_child_samples", 20),
            reg_lambda=meta.get("reg_lambda", 1.0),
        )
        model_path = meta.get("model_path", path.replace(".pkl", ".lgb"))
        obj._model = lgb.Booster(model_file=model_path)
        obj.n_iter_ = meta.get("n_iter_", 0)
        obj.converged_ = True
        return obj

    @staticmethod
    def available() -> bool:
        """True if lightgbm is importable."""
        try:
            import lightgbm  # noqa: F401
            return True
        except ImportError:
            return False
