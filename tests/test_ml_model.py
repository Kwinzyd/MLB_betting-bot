import os
import tempfile
import numpy as np
import pytest

from src.models.ml_model import PoissonGLM


def _synthetic_poisson_data(n=2000, n_features=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    beta_true = np.array([0.3, -0.2, 0.1, 0.05])
    exposure = rng.uniform(3.0, 7.0, size=n)  # e.g. innings pitched
    eta = X @ beta_true + np.log(exposure)
    mu = np.exp(eta)
    y = rng.poisson(mu)
    return X, y, exposure, beta_true


def test_fit_recovers_rate():
    X, y, exposure, _ = _synthetic_poisson_data()
    glm = PoissonGLM(feature_names=[f"f{i}" for i in range(X.shape[1])], l2=0.001)
    glm.fit(X, y, exposure=exposure)
    assert glm.converged_
    # Predicted means should be correlated with actuals
    preds = np.array([glm.predict_mean(X[i], exposure=exposure[i]) for i in range(len(y))])
    # Pearson corr as a sanity check
    corr = np.corrcoef(preds, y)[0, 1]
    assert corr > 0.3


def test_predict_single_and_batch_consistency():
    X, y, exposure, _ = _synthetic_poisson_data(n=500)
    glm = PoissonGLM().fit(X, y, exposure=exposure)
    single = glm.predict_mean(X[0], exposure=exposure[0])
    batch = glm.predict_mean(X[:5], exposure=exposure[0])
    assert abs(single - float(batch[0])) < 1e-9


def test_save_and_load_roundtrip(tmp_path):
    X, y, exposure, _ = _synthetic_poisson_data(n=500)
    glm = PoissonGLM(feature_names=["a", "b", "c", "d"]).fit(X, y, exposure=exposure)
    path = str(tmp_path / "glm.pkl")
    glm.save(path)
    loaded = PoissonGLM.load(path)
    p1 = glm.predict_mean(X[0], exposure=4.0)
    p2 = loaded.predict_mean(X[0], exposure=4.0)
    assert abs(p1 - p2) < 1e-9
    assert loaded.feature_names == ["a", "b", "c", "d"]


def test_zero_exposure_is_guarded():
    X, y, exposure, _ = _synthetic_poisson_data(n=200)
    # Some zero exposures should not crash the fit
    exposure[0] = 0.0
    exposure[1] = -1.0
    glm = PoissonGLM().fit(X, y, exposure=exposure)
    mu = glm.predict_mean(X[0], exposure=0.0)
    assert mu >= 0.0 and np.isfinite(mu)


def test_predict_before_fit_raises():
    glm = PoissonGLM()
    with pytest.raises(RuntimeError):
        glm.predict_mean(np.zeros(4), exposure=1.0)
