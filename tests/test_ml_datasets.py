"""Panel construction: alignment, lagging, NaN handling, validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import DataError
from quantlab.ml import PanelDataset


def _prices(n=15, symbols=("A", "B", "C"), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    data = {
        s: 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.02, n))) for s in symbols
    }
    return pd.DataFrame(data, index=idx)


def test_target_is_forward_return_over_t_to_t_plus_h():
    prices = _prices()
    factor = prices * 0.0  # all-zero factor, always valid
    panel = PanelDataset.from_factors({"f": factor}, prices, horizon=3)
    t = prices.index[2]
    expected = np.log(prices.loc[prices.index[5], "B"] / prices.loc[t, "B"])
    assert panel.y.loc[(t, "B")] == pytest.approx(expected)


def test_features_hold_factor_value_at_t():
    prices = _prices()
    factor = pd.DataFrame(
        np.arange(prices.size, dtype=float).reshape(prices.shape),
        index=prices.index,
        columns=prices.columns,
    )
    panel = PanelDataset.from_factors({"f": factor}, prices, horizon=2)
    t = prices.index[4]
    assert panel.X.loc[(t, "A"), "f"] == factor.loc[t, "A"]


def test_last_horizon_dates_have_no_rows():
    prices = _prices()
    panel = PanelDataset.from_factors({"f": prices * 0.0}, prices, horizon=4)
    assert panel.dates.max() == prices.index[-5]
    assert len(panel.dates) == len(prices.index) - 4


def test_feature_lag_shifts_features_back():
    prices = _prices()
    factor = pd.DataFrame(
        np.arange(prices.size, dtype=float).reshape(prices.shape),
        index=prices.index,
        columns=prices.columns,
    )
    lagged = PanelDataset.from_factors(
        {"f": factor}, prices, horizon=2, feature_lag=1
    )
    t = prices.index[4]
    assert lagged.X.loc[(t, "A"), "f"] == factor.loc[prices.index[3], "A"]


def test_nan_feature_drops_only_that_row():
    prices = _prices()
    factor = prices * 0.0
    t = prices.index[3]
    factor.loc[t, "B"] = np.nan
    panel = PanelDataset.from_factors({"f": factor}, prices, horizon=2)
    assert (t, "B") not in panel.X.index
    assert (t, "A") in panel.X.index


def test_inf_feature_treated_as_missing():
    prices = _prices()
    factor = prices * 0.0
    factor.loc[prices.index[3], "B"] = np.inf
    panel = PanelDataset.from_factors({"f": factor}, prices, horizon=2)
    assert (prices.index[3], "B") not in panel.X.index


def test_groups_match_date_level_and_are_sorted():
    prices = _prices()
    panel = PanelDataset.from_factors({"f": prices * 0.0}, prices, horizon=1)
    assert (panel.groups == panel.X.index.get_level_values("date")).all()
    assert panel.groups.is_monotonic_increasing


def test_alignment_uses_intersection_of_dates_and_symbols():
    prices = _prices(symbols=("A", "B", "C"))
    factor = (prices * 0.0).iloc[2:][["A", "B"]]  # shorter, fewer symbols
    panel = PanelDataset.from_factors({"f": factor}, prices, horizon=1)
    assert set(panel.X.index.get_level_values("symbol")) == {"A", "B"}
    assert panel.dates.min() == prices.index[2]


def test_multiple_factors_become_columns():
    prices = _prices()
    panel = PanelDataset.from_factors(
        {"f1": prices * 0.0, "f2": prices * 0.0 + 1.0}, prices, horizon=1
    )
    assert panel.feature_names == ["f1", "f2"]
    assert (panel.X["f2"] == 1.0).all()


@pytest.mark.parametrize(
    ("factors", "horizon", "match"),
    [
        ({}, 1, "empty"),
        ("prices_zero", 0, "horizon"),
        ("prices_zero", 100, "too little overlap"),
    ],
)
def test_from_factors_validation(factors, horizon, match):
    prices = _prices()
    if factors == "prices_zero":
        factors = {"f": prices * 0.0}
    with pytest.raises(DataError, match=match):
        PanelDataset.from_factors(factors, prices, horizon=horizon)


def test_all_nan_factor_raises():
    prices = _prices()
    with pytest.raises(DataError, match="No complete rows"):
        PanelDataset.from_factors({"f": prices * np.nan}, prices, horizon=1)
