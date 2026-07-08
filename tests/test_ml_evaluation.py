"""Evaluation: exact metric cases, rank IC wiring, permutation importance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import ModelError
from quantlab.ml import (
    LinearModel,
    permutation_importance,
    prediction_rank_ic,
    regression_metrics,
)
from quantlab.research.evaluation import forward_returns


def test_regression_metrics_hand_case():
    y_true = pd.Series([1.0, 2.0, 3.0])
    y_pred = pd.Series([1.0, 2.0, 4.0])
    m = regression_metrics(y_true, y_pred)
    assert m["mse"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["r2"] == pytest.approx(1 - (1 / 3) / (2 / 3))
    assert m["correlation"] == pytest.approx(float(y_true.corr(y_pred)))
    assert m["n"] == 3


def test_regression_metrics_perfect_prediction():
    y = pd.Series([0.1, -0.2, 0.05, 0.0])
    m = regression_metrics(y, y.copy())
    assert m["mse"] == 0.0
    assert m["r2"] == pytest.approx(1.0)


def test_regression_metrics_drop_unaligned_rows():
    y_true = pd.Series([1.0, 2.0, 3.0, np.nan], index=[0, 1, 2, 3])
    y_pred = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    m = regression_metrics(y_true, y_pred)
    assert m["n"] == 2  # only indexes 1 and 2 are complete


def test_regression_metrics_reject_tiny_overlap():
    with pytest.raises(ModelError, match="aligned"):
        regression_metrics(pd.Series([1.0]), pd.Series([1.0]))


def test_prediction_rank_ic_is_one_for_perfect_predictor():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2022-01-03", periods=30)
    prices = pd.DataFrame(
        {s: 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 30))) for s in "ABCDE"},
        index=idx,
    )
    horizon = 3
    predictions = forward_returns(prices, horizon)  # cheating on purpose
    ic = prediction_rank_ic(predictions, prices, horizon)
    assert ic.dropna().mean() == pytest.approx(1.0)


def test_permutation_importance_separates_signal_from_noise():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {"signal": rng.normal(size=500), "noise": rng.normal(size=500)}
    )
    y = 2.0 * X["signal"] + rng.normal(0, 0.1, 500)
    model = LinearModel(alpha=0.0).fit(X, y)
    imp = permutation_importance(model, X, y, n_repeats=5, random_state=0)
    assert imp["signal"] > 10 * max(imp["noise"], 1e-12)
    assert imp["noise"] == pytest.approx(0.0, abs=0.05)


def test_permutation_importance_validation():
    X = pd.DataFrame({"f": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    model = LinearModel().fit(X, y)
    with pytest.raises(ModelError, match="n_repeats"):
        permutation_importance(model, X, y, n_repeats=0)
    with pytest.raises(ModelError, match="complete rows"):
        permutation_importance(model, X.iloc[:2], y.iloc[:2])
