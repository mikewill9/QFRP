"""Attribution: exact reconstruction, planted-factor recovery, validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.analytics.attribution import asset_contributions, factor_attribution
from quantlab.exceptions import DataError


def _frame(data: dict[str, list[float]]) -> pd.DataFrame:
    n = len(next(iter(data.values())))
    return pd.DataFrame(data, index=pd.bdate_range("2022-01-03", periods=n))


# ------------------------------------------------------------ asset_contributions


def test_constant_weights_contributions_sum_to_portfolio_return():
    idx = pd.bdate_range("2022-01-03", periods=6)
    weights = pd.DataFrame({"A": 0.6, "B": 0.4}, index=idx)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {"A": rng.normal(0, 0.01, 6), "B": rng.normal(0, 0.01, 6)}, index=idx
    )
    contributions = asset_contributions(weights, returns)
    portfolio_return = 0.6 * returns["A"] + 0.4 * returns["B"]
    total = contributions.sum(axis=1, skipna=False)
    # First row is NaN: no prior-bar weight exists yet.
    assert total.iloc[1:].to_numpy() == pytest.approx(portfolio_return.iloc[1:].to_numpy())
    assert pd.isna(total.iloc[0])


def test_contribution_uses_prior_bar_weight():
    idx = pd.bdate_range("2022-01-03", periods=3)
    weights = pd.DataFrame({"A": [0.2, 0.8, 0.8]}, index=idx)
    returns = pd.DataFrame({"A": [0.0, 0.05, -0.02]}, index=idx)
    contributions = asset_contributions(weights, returns)
    # Bar 1's contribution uses bar 0's weight (0.2), not bar 1's (0.8).
    assert contributions["A"].iloc[1] == pytest.approx(0.2 * 0.05)
    assert contributions["A"].iloc[2] == pytest.approx(0.8 * -0.02)


def test_asset_contributions_aligns_on_intersection():
    idx = pd.bdate_range("2022-01-03", periods=4)
    weights = pd.DataFrame({"A": 0.5, "B": 0.5, "C": 0.0}, index=idx)
    returns = pd.DataFrame({"A": [0.01] * 3, "B": [0.02] * 3}, index=idx[1:])
    out = asset_contributions(weights, returns)
    assert set(out.columns) == {"A", "B"}
    assert len(out) == 3


def test_asset_contributions_rejects_no_overlap():
    weights = _frame({"A": [0.5, 0.5]})
    returns = pd.DataFrame({"B": [0.01, 0.02]}, index=pd.bdate_range("2030-01-01", periods=2))
    with pytest.raises(DataError, match="too little overlap"):
        asset_contributions(weights, returns)


# ------------------------------------------------------------ factor_attribution


def test_factor_attribution_recovers_planted_alpha_and_betas():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2020-01-01", periods=500)
    factors = pd.DataFrame(
        {"mkt": rng.normal(0, 0.01, 500), "size": rng.normal(0, 0.005, 500)},
        index=idx,
    )
    returns = 0.0003 + 1.2 * factors["mkt"] - 0.4 * factors["size"]
    result = factor_attribution(returns, factors)
    assert result.alpha == pytest.approx(0.0003, abs=1e-12)
    assert result.betas["mkt"] == pytest.approx(1.2)
    assert result.betas["size"] == pytest.approx(-0.4)
    assert result.r_squared == pytest.approx(1.0, abs=1e-10)


def test_factor_attribution_contributions_reconstruct_returns_exactly():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2021-01-04", periods=200)
    factors = pd.DataFrame({"f1": rng.normal(0, 0.01, 200)}, index=idx)
    returns = 0.5 * factors["f1"] + rng.normal(0, 0.002, 200)
    returns = pd.Series(returns, index=idx)
    result = factor_attribution(returns, factors)
    reconstructed = result.contributions.sum(axis=1)
    np.testing.assert_allclose(reconstructed.to_numpy(), returns.to_numpy(), atol=1e-12)
    assert list(result.contributions.columns) == ["alpha", "f1", "residual"]


def test_factor_attribution_noisy_data_gives_partial_r_squared():
    rng = np.random.default_rng(6)
    idx = pd.bdate_range("2020-01-01", periods=1000)
    factors = pd.DataFrame({"mkt": rng.normal(0, 0.01, 1000)}, index=idx)
    returns = 0.5 * factors["mkt"] + rng.normal(0, 0.01, 1000)
    result = factor_attribution(pd.Series(returns.to_numpy(), index=idx), factors)
    assert 0.0 < result.r_squared < 1.0


def test_factor_attribution_rejects_collinear_factors():
    idx = pd.bdate_range("2020-01-01", periods=100)
    base = pd.Series(np.random.default_rng(0).normal(0, 0.01, 100), index=idx)
    factors = pd.DataFrame({"f1": base, "f2": 2.0 * base})
    with pytest.raises(DataError, match="collinear"):
        factor_attribution(base, factors)


def test_factor_attribution_rejects_short_overlap():
    idx = pd.bdate_range("2020-01-01", periods=3)
    factors = pd.DataFrame({"f1": [0.01, 0.02, 0.01], "f2": [0.0, 0.01, 0.02]}, index=idx)
    returns = pd.Series([0.01, 0.0, 0.01], index=idx)
    with pytest.raises(DataError, match="overlapping"):
        factor_attribution(returns, factors)


def test_factor_attribution_rejects_no_factors():
    idx = pd.bdate_range("2020-01-01", periods=10)
    returns = pd.Series(np.zeros(10), index=idx)
    with pytest.raises(DataError, match="no columns"):
        factor_attribution(returns, pd.DataFrame(index=idx))
