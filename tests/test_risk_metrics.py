"""Risk metrics and drawdown anatomy: exact hand cases and properties."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from quantlab.exceptions import DataError
from quantlab.risk import (
    annualized_vol,
    beta,
    drawdown_episodes,
    drawdown_series,
    expected_shortfall,
    max_drawdown,
    rolling_vol,
    ulcer_index,
    value_at_risk,
)


def _series(values: list[float]) -> pd.Series:
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)))


def _normal_sample(n=20_000, mean=0.0005, std=0.01, seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return _series(list(rng.normal(mean, std, n)))


# ------------------------------------------------------------ volatility


def test_annualized_vol_explicit_periods():
    r = _series([0.01, -0.01, 0.02, 0.00])
    assert annualized_vol(r, periods=252) == pytest.approx(
        float(r.std()) * np.sqrt(252)
    )


def test_rolling_vol_last_window():
    r = _series([0.01, -0.02, 0.03, 0.01, -0.01])
    out = rolling_vol(r, window=3, periods=252)
    expected = float(r.iloc[-3:].std()) * np.sqrt(252)
    assert out.iloc[-1] == pytest.approx(expected)
    assert out.iloc[:2].isna().all()


def test_rolling_vol_rejects_tiny_window():
    with pytest.raises(DataError, match="window"):
        rolling_vol(_series([0.01, 0.02, 0.03]), window=1)


# ------------------------------------------------------------ VaR


def test_historical_var_hand_case():
    r = _series([-0.04, -0.02, 0.01, 0.03])
    # tail=0.25: linear-interpolated quantile between -0.04 and -0.02.
    assert value_at_risk(r, level=0.75) == pytest.approx(0.025)


def test_normal_var_matches_formula():
    r = _series([0.01, -0.02, 0.005, 0.03, -0.015])
    z = stats.norm.ppf(0.05)
    expected = -(float(r.mean()) + z * float(r.std()))
    assert value_at_risk(r, level=0.95, method="normal") == pytest.approx(expected)


def test_var_methods_agree_on_gaussian_data():
    r = _normal_sample()
    hist = value_at_risk(r, 0.95, "historical")
    norm = value_at_risk(r, 0.95, "normal")
    cf = value_at_risk(r, 0.95, "cornish_fisher")
    assert hist == pytest.approx(norm, rel=0.05)
    assert cf == pytest.approx(norm, rel=0.05)


def test_cornish_fisher_var_exceeds_normal_for_left_skew():
    # Moderately left-skewed returns (the CF expansion is only valid for
    # moderate skew/kurtosis, so keep the asymmetry realistic).
    rng = np.random.default_rng(7)
    r = _series(list(stats.skewnorm.rvs(-4, scale=0.01, size=20_000, random_state=rng)))
    cf = value_at_risk(r, 0.99, "cornish_fisher")
    norm = value_at_risk(r, 0.99, "normal")
    assert cf > norm


def test_var_is_positive_for_losing_tail():
    assert value_at_risk(_normal_sample(), 0.95) > 0


# ------------------------------------------------------------ Expected Shortfall


def test_historical_es_hand_case():
    r = _series([-0.04, -0.02, 0.01, 0.03])
    # Only -0.04 lies at or below the 25% quantile cutoff (-0.025).
    assert expected_shortfall(r, level=0.75) == pytest.approx(0.04)


def test_normal_es_matches_formula():
    r = _series([0.01, -0.02, 0.005, 0.03, -0.015])
    z = stats.norm.ppf(0.05)
    expected = -(float(r.mean()) - float(r.std()) * stats.norm.pdf(z) / 0.05)
    assert expected_shortfall(r, 0.95, "normal") == pytest.approx(expected)


def test_cornish_fisher_es_converges_to_normal_on_gaussian_data():
    r = _normal_sample()
    cf = expected_shortfall(r, 0.95, "cornish_fisher")
    norm = expected_shortfall(r, 0.95, "normal")
    assert cf == pytest.approx(norm, rel=0.05)


@pytest.mark.parametrize("method", ["historical", "normal", "cornish_fisher"])
def test_es_at_least_var(method):
    r = _normal_sample(n=5000)
    assert expected_shortfall(r, 0.95, method) >= value_at_risk(r, 0.95, method) - 1e-12


# ------------------------------------------------------------ shared validation


@pytest.mark.parametrize("level", [0.3, 1.0, 1.5])
def test_tail_metrics_reject_bad_level(level):
    with pytest.raises(DataError, match="level"):
        value_at_risk(_series([0.01, -0.01, 0.02]), level=level)


def test_tail_metrics_reject_unknown_method():
    r = _series([0.01, -0.01, 0.02])
    with pytest.raises(DataError, match="Unknown"):
        value_at_risk(r, method="bootstrap")
    with pytest.raises(DataError, match="Unknown"):
        expected_shortfall(r, method="bootstrap")


# ------------------------------------------------------------ beta


def test_beta_recovers_linear_relation():
    rng = np.random.default_rng(2)
    bench = _series(list(rng.normal(0, 0.01, 300)))
    portfolio = 1.5 * bench + 0.0001
    assert beta(portfolio, bench) == pytest.approx(1.5)


def test_beta_aligns_on_common_dates():
    bench = _series([0.01, -0.01, 0.02, 0.00, 0.01])
    portfolio = (2.0 * bench).iloc[1:]  # shorter history
    assert beta(portfolio, bench) == pytest.approx(2.0)


def test_beta_rejects_flat_benchmark():
    flat = _series([0.01, 0.01, 0.01, 0.01])
    with pytest.raises(DataError, match="zero variance"):
        beta(_series([0.02, -0.01, 0.00, 0.01]), flat)


# ------------------------------------------------------------ drawdowns

# wealth: 1.10, 0.88, 0.924, 1.155, 1.0395 -> dd: 0, -0.2, -0.16, 0, -0.1
DD_RETURNS = _series([0.10, -0.20, 0.05, 0.25, -0.10])


def test_drawdown_series_hand_case():
    dd = drawdown_series(DD_RETURNS)
    np.testing.assert_allclose(dd.to_numpy(), [0.0, -0.2, -0.16, 0.0, -0.1])


def test_max_drawdown_hand_case():
    assert max_drawdown(DD_RETURNS) == pytest.approx(-0.2)


def test_drawdown_episodes_hand_case():
    episodes = drawdown_episodes(DD_RETURNS)
    idx = DD_RETURNS.index
    assert len(episodes) == 2

    first = episodes.iloc[0]
    assert first["start"] == idx[1]
    assert first["trough"] == idx[1]
    assert first["recovery"] == idx[3]
    assert first["depth"] == pytest.approx(-0.2)
    assert first["length"] == 2
    assert bool(first["recovered"])

    second = episodes.iloc[1]
    assert second["start"] == idx[4]
    assert pd.isna(second["recovery"])
    assert second["depth"] == pytest.approx(-0.1)
    assert not bool(second["recovered"])


def test_drawdown_episodes_empty_when_always_rising():
    episodes = drawdown_episodes(_series([0.01, 0.02, 0.01]))
    assert episodes.empty
    assert list(episodes.columns) == [
        "start", "trough", "recovery", "depth", "length", "recovered",
    ]


def test_ulcer_index_hand_case():
    expected = np.sqrt((0.0 + 0.2**2 + 0.16**2 + 0.0 + 0.1**2) / 5)
    assert ulcer_index(DD_RETURNS) == pytest.approx(expected)


def test_ulcer_penalizes_duration():
    crash_and_recover = _series([0.0, -0.20, 0.25, 0.0, 0.0, 0.0])
    long_grind = _series([0.0, -0.20, 0.0, 0.0, 0.0, 0.25])
    assert ulcer_index(long_grind) > ulcer_index(crash_and_recover)
