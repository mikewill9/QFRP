"""Analytics: hand-checked performance statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.analytics import (
    cagr,
    calmar_ratio,
    hit_rate,
    max_drawdown,
    performance_summary,
    periods_per_year,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    turnover,
)
from quantlab.exceptions import DataError


def _daily(values) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values)))


def test_periods_per_year_inference():
    daily = pd.bdate_range("2020-01-01", periods=100)
    weekly = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    monthly = pd.date_range("2020-01-31", periods=36, freq="BME")
    assert periods_per_year(daily) == 252.0
    assert periods_per_year(weekly) == 52.0
    assert periods_per_year(monthly) == 12.0


def test_total_return_and_cagr():
    # One year of a constant per-bar return that exactly doubles the book.
    r = (2.0 ** (1 / 252)) - 1.0
    rets = _daily([r] * 252)
    assert total_return(rets) == pytest.approx(1.0)
    assert cagr(rets) == pytest.approx(1.0)  # doubled in one year


def test_sharpe_constant_series():
    rets = _daily(np.full(300, 0.001))
    rets.iloc[0] = 0.0011  # break zero variance
    expected = rets.mean() / rets.std() * np.sqrt(252)
    assert sharpe_ratio(rets) == pytest.approx(expected)


def test_sharpe_with_risk_free():
    rets = _daily(np.full(252, 0.02 / 252))
    # Excess return exactly zero at rf=2% annual.
    assert abs(sharpe_ratio(rets, rf=0.02)) < 1e-8 or np.isnan(sharpe_ratio(rets, rf=0.02))


def test_sortino_ignores_upside_vol():
    calm = _daily([0.001, 0.001, -0.001] * 100)
    spiky_up = _daily([0.001, 0.05, -0.001] * 100)  # huge upside spikes
    assert sortino_ratio(spiky_up) > sortino_ratio(calm)
    all_positive = _daily(np.full(50, 0.001))
    assert sortino_ratio(all_positive) == np.inf


def test_max_drawdown_hand_case():
    # 100 -> 120 -> 90 -> 130: worst drawdown = 90/120 - 1 = -25%.
    prices = [100.0, 120.0, 90.0, 130.0]
    rets = _daily([prices[i] / prices[i - 1] - 1 for i in range(1, 4)])
    assert max_drawdown(rets) == pytest.approx(-0.25)
    assert calmar_ratio(rets) == pytest.approx(cagr(rets) / 0.25)


def test_hit_rate():
    rets = _daily([0.01, -0.01, 0.02, 0.0, 0.03])
    assert hit_rate(rets) == pytest.approx(3 / 5)


def test_turnover_hand_case():
    idx = pd.bdate_range("2022-01-03", periods=3)
    w = pd.DataFrame(
        {"A": [0.5, 0.0, 0.5], "B": [0.5, 1.0, 0.5]}, index=idx
    )
    to = turnover(w)
    assert to.iloc[1] == pytest.approx(0.5)  # sold half the book into B
    assert to.iloc[2] == pytest.approx(0.5)


def test_rolling_sharpe_shape():
    rng = np.random.default_rng(1)
    rets = _daily(rng.normal(0.0005, 0.01, 400))
    rs = rolling_sharpe(rets, window=126)
    assert rs.iloc[:125].isna().all()
    assert np.isfinite(rs.iloc[126:]).all()


def test_performance_summary_keys_and_consistency():
    rng = np.random.default_rng(2)
    rets = _daily(rng.normal(0.0004, 0.008, 500))
    idx = rets.index
    w = pd.DataFrame({"A": 0.6, "B": 0.4}, index=idx)
    s = performance_summary(rets, weights=w)
    expected = {
        "total_return", "cagr", "annualized_vol", "sharpe", "sortino",
        "calmar", "max_drawdown", "hit_rate", "skew", "excess_kurtosis",
        "n_periods", "annualized_turnover",
    }
    assert set(s.index) == expected
    assert s["sharpe"] == pytest.approx(sharpe_ratio(rets))
    assert s["annualized_turnover"] == pytest.approx(0.0)  # frozen weights
    with pytest.raises(DataError):
        performance_summary(pd.Series(dtype=float))
