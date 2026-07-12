"""Model comparison on shared purged folds: ranking, determinism, contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import ModelError
from quantlab.ml import LinearModel, PanelDataset, PurgedKFold, compare_models


class _MeanModel:
    """Dummy wrapper predicting the training-target mean (zero skill)."""

    def __init__(self) -> None:
        self._mean = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _MeanModel:
        self._mean = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._mean)

    @property
    def feature_importances_(self) -> pd.Series | None:
        return None


def _planted_panel(n_dates=60, symbols=("A", "B", "C", "D", "E", "F"), seed=3):
    """Hand-built panel where y = 0.5 * feature + small noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    index = pd.MultiIndex.from_product(
        [dates, list(symbols)], names=["date", "symbol"]
    )
    feature = rng.normal(size=len(index))
    X = pd.DataFrame({"f": feature}, index=index)
    y = pd.Series(
        0.5 * feature + rng.normal(scale=0.05, size=len(index)),
        index=index,
        name="forward_return",
    )
    groups = index.get_level_values("date").to_series(index=index).rename("date")
    return PanelDataset(X=X, y=y, groups=groups, horizon=1)


# The mean model's constant predictions make Pearson correlation undefined
# inside regression_metrics; numpy warns on the NaN divide, which is expected.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compare_models_ranks_signal_over_noise():
    panel = _planted_panel()
    table = compare_models(
        panel,
        {"ridge": lambda: LinearModel(alpha=1.0), "mean": _MeanModel},
        PurgedKFold(n_splits=4, horizon=1),
    )
    assert table.index[0] == "ridge"
    assert table.loc["ridge", "mean_rank_ic"] > 0.5
    mean_ic = table.loc["mean", "mean_rank_ic"]
    # Constant predictions have no rank ordering: NaN IC (or, at best, ~0).
    assert np.isnan(mean_ic) or mean_ic < table.loc["ridge", "mean_rank_ic"]


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compare_models_is_deterministic():
    panel = _planted_panel()
    factories = {"ridge": lambda: LinearModel(alpha=1.0), "mean": _MeanModel}
    first = compare_models(panel, factories, PurgedKFold(n_splits=4, horizon=1))
    second = compare_models(panel, factories, PurgedKFold(n_splits=4, horizon=1))
    pd.testing.assert_frame_equal(first, second)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_compare_models_frame_contract():
    panel = _planted_panel()
    splitter = PurgedKFold(n_splits=5, horizon=1)
    n_folds = len(list(splitter.split(panel)))
    table = compare_models(
        panel,
        {"ridge": lambda: LinearModel(alpha=1.0), "mean": _MeanModel},
        splitter,
    )
    assert list(table.columns) == [
        "mean_r2",
        "std_r2",
        "mean_mse",
        "mean_rank_ic",
        "n_folds",
    ]
    assert table.index.name == "model"
    assert set(table.index) == {"ridge", "mean"}
    assert (table["n_folds"] == n_folds).all()


def test_compare_models_empty_factories_raises():
    panel = _planted_panel()
    with pytest.raises(ModelError, match="empty"):
        compare_models(panel, {}, PurgedKFold(n_splits=4, horizon=1))
