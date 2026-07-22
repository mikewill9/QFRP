"""Gradient boosting challenger (Checkpoint 11).

Pins the pieces the notebook's ``ML_MODEL = "gbm"`` path relies on:
the ``subsample`` knob, walk-forward training in both window modes,
purged k-fold compatibility, and ``importance_history`` as the
tree-model analogue of ``coefficient_history``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import ModelError
from quantlab.ml import (
    GradientBoostingModel,
    PanelDataset,
    PurgedKFold,
    WalkForward,
    WalkForwardTrainer,
    importance_history,
)

# The checkpoint's challenger configuration (set in the notebook config cell).
CHALLENGER = {
    "n_estimators": 100,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
}


def _xy(n=400, seed=0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"f1": rng.normal(size=n), "f2": rng.normal(size=n)},
    )
    y = 2.0 * X["f1"] - 1.0 * X["f2"]
    return X, y.rename("y")


def _planted_panel(
    n_dates=120, symbols=("A", "B", "C", "D", "E"), beta=0.3, noise=0.002, seed=5
) -> PanelDataset:
    """Prices whose next-bar log return is ``beta * factor_t`` plus noise."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n_dates)
    factor = pd.DataFrame(
        rng.normal(size=(n_dates, len(symbols))), index=idx, columns=list(symbols)
    )
    drivers = beta * factor.to_numpy() + rng.normal(0, noise, factor.shape)
    log_prices = np.log(100.0) + np.vstack(
        [np.zeros(len(symbols)), np.cumsum(drivers[:-1], axis=0)]
    )
    prices = pd.DataFrame(np.exp(log_prices), index=idx, columns=list(symbols))
    return PanelDataset.from_factors({"f": factor}, prices, horizon=1)


def _skip_unless_sklearn(model: GradientBoostingModel) -> None:
    # Sum-to-one importances are a sklearn contract; LightGBM (if installed)
    # reports raw split counts instead.
    if model.backend != "sklearn":
        pytest.skip("normalized importances require the sklearn backend")


# ---------------------------------------------------------------- subsample


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_subsample_out_of_range_raises(bad):
    with pytest.raises(ModelError, match="subsample"):
        GradientBoostingModel(subsample=bad)


def test_subsample_default_matches_explicit_full_sample():
    # The new kwarg's default must reproduce the pre-Checkpoint-11 model
    # exactly, so existing results are unchanged.
    X, y = _xy()
    old = GradientBoostingModel(n_estimators=30, random_state=0).fit(X, y)
    new = GradientBoostingModel(n_estimators=30, subsample=1.0, random_state=0).fit(X, y)
    np.testing.assert_array_equal(old.predict(X), new.predict(X))


def test_subsample_deterministic_given_seed():
    X, y = _xy()
    a = GradientBoostingModel(n_estimators=30, subsample=0.8, random_state=3)
    b = GradientBoostingModel(n_estimators=30, subsample=0.8, random_state=3)
    np.testing.assert_array_equal(a.fit(X, y).predict(X), b.fit(X, y).predict(X))


# ----------------------------------------------------- challenger behavior


def test_challenger_learns_planted_signal_with_finite_predictions():
    X, y = _xy()
    model = GradientBoostingModel(**CHALLENGER, random_state=0).fit(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert np.corrcoef(preds, y.to_numpy())[0, 1] > 0.9


def test_challenger_importances_normalized_and_ranked():
    X, y = _xy()
    model = GradientBoostingModel(**CHALLENGER, random_state=0).fit(X, y)
    _skip_unless_sklearn(model)
    imp = model.feature_importances_
    assert imp is not None
    assert (imp >= 0).all()
    assert imp.sum() == pytest.approx(1.0, abs=1e-9)
    assert imp["f1"] > imp["f2"]  # f1 has twice the effect


def test_constant_target_predicts_exactly_the_constant():
    # Hand-computed case: with y constant, boosting's initial estimate is
    # the mean and every tree fits zero residuals, so predictions are
    # exactly that constant.
    X, _ = _xy(n=50)
    y = pd.Series(np.full(len(X), 3.7), name="y")
    model = GradientBoostingModel(n_estimators=10, random_state=0).fit(X, y)
    _skip_unless_sklearn(model)
    np.testing.assert_allclose(model.predict(X), 3.7)


# ------------------------------------------------ walk-forward, both modes


@pytest.mark.parametrize("mode", ["rolling", "expanding"])
def test_walk_forward_runs_in_both_window_modes(mode):
    panel = _planted_panel()
    signal = WalkForwardTrainer(
        lambda: GradientBoostingModel(n_estimators=30, subsample=0.8, random_state=0),
        WalkForward(train_periods=40, test_periods=20, mode=mode),
    ).run(panel)
    assert signal.n_folds > 1
    stacked = signal.predictions.stack(future_stack=True).dropna()
    assert len(stacked) > 0
    assert np.isfinite(stacked.to_numpy()).all()
    for _, fold in signal.folds.iterrows():
        assert fold["train_end"] < fold["test_start"]  # strictly causal


def test_purged_kfold_runs_with_gbm():
    # The purge/embargo machinery is model-agnostic; pin that the GBM
    # challenger goes through it exactly as Ridge does.
    panel = _planted_panel(n_dates=80)
    signal = WalkForwardTrainer(
        lambda: GradientBoostingModel(n_estimators=20, subsample=0.8, random_state=0),
        PurgedKFold(n_splits=4, horizon=1),
    ).run(panel)
    assert signal.n_folds == 4
    # K-fold predicts every date exactly once.
    assert len(signal.predictions.index) == len(panel.dates)


# -------------------------------------------------------- importance_history


def test_importance_history_frame_contract():
    panel = _planted_panel()
    signal = WalkForwardTrainer(
        lambda: GradientBoostingModel(n_estimators=20, random_state=0),
        WalkForward(train_periods=40, test_periods=20),
    ).run(panel)
    history = importance_history(signal)
    assert list(history.columns) == signal.feature_names
    assert len(history) == signal.n_folds
    assert history.index.name == "test_start"
    assert (history.index == pd.DatetimeIndex(signal.folds["test_start"])).all()


def test_importance_history_rows_are_normalized_per_fold():
    panel = _planted_panel()
    signal = WalkForwardTrainer(
        lambda: GradientBoostingModel(n_estimators=20, random_state=0),
        WalkForward(train_periods=40, test_periods=20),
    ).run(panel)
    _skip_unless_sklearn(signal.models[0])
    history = importance_history(signal)
    assert (history.to_numpy() >= 0).all()
    np.testing.assert_allclose(history.sum(axis=1).to_numpy(), 1.0, atol=1e-9)


def test_importance_history_rejects_models_without_importances():
    class ImportanceFreeModel:
        """ModelWrapper-compatible stub with no notion of importance."""

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X))

        @property
        def feature_importances_(self):
            return None

    panel = _planted_panel()
    signal = WalkForwardTrainer(
        ImportanceFreeModel, WalkForward(train_periods=40, test_periods=20)
    ).run(panel)
    with pytest.raises(ModelError, match="feature importances"):
        importance_history(signal)
