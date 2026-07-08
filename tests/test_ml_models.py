"""Model wrappers: recovery of planted signal, guards, ensemble math."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import ModelError
from quantlab.ml import (
    EnsembleModel,
    GradientBoostingModel,
    LinearModel,
    TreeModel,
)


def _xy(n=400, seed=0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n)},
    )
    y = 2.0 * X["f1"] - 1.0 * X["f2"]
    return X, y.rename("y")


class ConstantModel:
    """Minimal ModelWrapper-compatible stub predicting a constant."""

    def __init__(self, value: float) -> None:
        self.value = value

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), self.value)

    @property
    def feature_importances_(self):
        return None


# ------------------------------------------------------------ LinearModel


def test_linear_model_recovers_planted_coefficients():
    X, y = _xy()
    model = LinearModel(alpha=0.0).fit(X, y)
    assert model.coef_["f1"] == pytest.approx(2.0, abs=1e-10)
    assert model.coef_["f2"] == pytest.approx(-1.0, abs=1e-10)
    np.testing.assert_allclose(model.predict(X), y.to_numpy(), atol=1e-10)


def test_linear_model_importances_are_absolute_coefficients():
    X, y = _xy()
    imp = LinearModel(alpha=0.0).fit(X, y).feature_importances_
    assert imp is not None
    assert imp["f1"] == pytest.approx(2.0, abs=1e-10)
    assert imp["f2"] == pytest.approx(1.0, abs=1e-10)


def test_linear_model_ridge_shrinks_coefficients():
    X, y = _xy()
    loose = LinearModel(alpha=0.0).fit(X, y).coef_["f1"]
    tight = LinearModel(alpha=1000.0).fit(X, y).coef_["f1"]
    assert abs(tight) < abs(loose)


def test_linear_model_rejects_negative_alpha():
    with pytest.raises(ModelError, match="alpha"):
        LinearModel(alpha=-1.0)


# ------------------------------------------------------------ guards


def test_predict_before_fit_raises():
    X, _ = _xy(n=10)
    with pytest.raises(ModelError, match="not fitted"):
        LinearModel().predict(X)


def test_predict_with_different_columns_raises():
    X, y = _xy()
    model = LinearModel().fit(X, y)
    with pytest.raises(ModelError, match="do not match"):
        model.predict(X.rename(columns={"f1": "other"}))


def test_fit_rejects_mismatched_lengths():
    X, y = _xy()
    with pytest.raises(ModelError, match="rows"):
        LinearModel().fit(X, y.iloc[:-5])


def test_fit_rejects_empty_frame():
    X, y = _xy()
    with pytest.raises(ModelError, match="empty"):
        LinearModel().fit(X.iloc[:0], y.iloc[:0])


# ------------------------------------------------------------ TreeModel


def test_tree_model_fits_signal_and_ranks_informative_feature():
    X, y = _xy()
    model = TreeModel(n_estimators=50, random_state=0).fit(X, y)
    preds = model.predict(X)
    assert np.corrcoef(preds, y.to_numpy())[0, 1] > 0.9
    imp = model.feature_importances_
    assert imp is not None
    assert imp["f1"] > imp["f2"]  # f1 has twice the effect


def test_tree_model_deterministic_given_seed():
    X, y = _xy()
    a = TreeModel(n_estimators=20, random_state=3).fit(X, y).predict(X)
    b = TreeModel(n_estimators=20, random_state=3).fit(X, y).predict(X)
    np.testing.assert_array_equal(a, b)


# ------------------------------------------------------------ GradientBoostingModel


def test_gbm_backend_resolved_at_fit_time():
    X, y = _xy()
    model = GradientBoostingModel(n_estimators=30, random_state=0)
    assert model.backend is None  # nothing imported or chosen yet
    model.fit(X, y)
    assert model.backend in ("lightgbm", "sklearn")


def test_gbm_learns_planted_signal_with_importances():
    X, y = _xy()
    model = GradientBoostingModel(n_estimators=100, random_state=0).fit(X, y)
    assert np.corrcoef(model.predict(X), y.to_numpy())[0, 1] > 0.95
    imp = model.feature_importances_
    assert imp is not None
    assert imp["f1"] > imp["f2"]


def test_gbm_never_imported_at_module_load():
    import sys

    # Importing the ml package must not pull lightgbm in.
    assert "quantlab.ml.models" in sys.modules
    fresh = {m for m in sys.modules if m.startswith("lightgbm")}
    if fresh:
        pytest.skip("lightgbm imported by another test in this session")
    assert not fresh


# ------------------------------------------------------------ EnsembleModel


def test_ensemble_is_exact_weighted_average():
    X, y = _xy(n=20)
    ensemble = EnsembleModel(
        [ConstantModel(1.0), ConstantModel(3.0)], weights=[3.0, 1.0]
    ).fit(X, y)
    # Normalized weights (0.75, 0.25): 0.75*1 + 0.25*3 = 1.5.
    np.testing.assert_allclose(ensemble.predict(X), 1.5)


def test_ensemble_defaults_to_equal_weights():
    X, y = _xy(n=20)
    ensemble = EnsembleModel([ConstantModel(0.0), ConstantModel(2.0)]).fit(X, y)
    np.testing.assert_allclose(ensemble.predict(X), 1.0)


def test_ensemble_importances_none_when_any_member_lacks_them():
    X, y = _xy()
    ensemble = EnsembleModel([LinearModel(), ConstantModel(1.0)]).fit(X, y)
    assert ensemble.feature_importances_ is None


def test_ensemble_importances_blend_members():
    X, y = _xy()
    ensemble = EnsembleModel(
        [LinearModel(alpha=0.0), LinearModel(alpha=0.0)], weights=[1.0, 1.0]
    ).fit(X, y)
    imp = ensemble.feature_importances_
    assert imp is not None
    assert imp["f1"] == pytest.approx(2.0, abs=1e-10)


@pytest.mark.parametrize(
    ("models", "weights", "match"),
    [
        ([], None, "at least one"),
        ([ConstantModel(1.0)], [0.5, 0.5], "weights for"),
        ([ConstantModel(1.0)], [-1.0], "non-negative"),
    ],
)
def test_ensemble_validation(models, weights, match):
    with pytest.raises(ModelError, match=match):
        EnsembleModel(models, weights=weights)


def test_ensemble_predict_before_fit_raises():
    X, _ = _xy(n=5)
    with pytest.raises(ModelError, match="not fitted"):
        EnsembleModel([ConstantModel(1.0)]).predict(X)
