"""Evaluation: alignment correctness, IC extremes, quantile monotonicity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.exceptions import DataError
from quantlab.research import (
    evaluate_signal,
    forward_returns,
    ic_decay,
    ic_series,
    ic_summary,
    newey_west_tstat,
    quantile_returns,
    quantile_spread,
    signal_turnover,
)


@pytest.fixture
def rng():
    return np.random.default_rng(7)


def _prices(rng, n_t=300, n_sym=10):
    idx = pd.bdate_range("2020-01-01", periods=n_t)
    cols = [f"S{i}" for i in range(n_sym)]
    rets = rng.normal(0.0002, 0.01, (n_t, n_sym))
    return pd.DataFrame(100 * np.exp(rets.cumsum(axis=0)), index=idx, columns=cols)


def test_forward_returns_alignment(rng):
    prices = _prices(rng, n_t=10, n_sym=2)
    fwd = forward_returns(prices, horizon=1)
    expected = np.log(prices.iloc[4, 0] / prices.iloc[3, 0])
    assert fwd.iloc[3, 0] == pytest.approx(expected)
    assert fwd.iloc[-1].isna().all()  # last row has no next bar
    fwd3 = forward_returns(prices, horizon=3)
    assert fwd3.iloc[-3:].isna().all().all()
    assert fwd3.iloc[0, 1] == pytest.approx(np.log(prices.iloc[3, 1] / prices.iloc[0, 1]))


def test_ic_extremes_and_orientation(rng):
    prices = _prices(rng)
    fwd = forward_returns(prices, 1)
    perfect = ic_series(fwd, fwd)  # a signal equal to what it predicts
    assert perfect.dropna().min() == pytest.approx(1.0)
    inverse = ic_series(-fwd, fwd)
    assert inverse.dropna().max() == pytest.approx(-1.0)
    noise = pd.DataFrame(
        rng.normal(size=fwd.shape), index=fwd.index, columns=fwd.columns
    )
    assert abs(ic_series(noise, fwd).mean()) < 0.05


def test_ic_spearman_invariant_to_monotone_transform(rng):
    prices = _prices(rng)
    fwd = forward_returns(prices, 1)
    base = ic_series(fwd, fwd, method="spearman")
    warped = ic_series(fwd**3, fwd, method="spearman")  # cube preserves order
    pd.testing.assert_series_equal(
        base.dropna(), warped.dropna(), check_names=False
    )


def test_newey_west_reduces_to_plain_tstat_at_lag0(rng):
    x = pd.Series(rng.normal(0.5, 1.0, 400))
    plain = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_tstat(x, lags=0) == pytest.approx(plain, rel=1e-6)


def test_newey_west_shrinks_tstat_for_autocorrelated_series(rng):
    # Overlapping-window style series: heavy positive autocorrelation.
    eps = rng.normal(0.02, 1.0, 2000)
    x = pd.Series(eps).rolling(10).mean().dropna()
    t0 = newey_west_tstat(x, lags=0)
    t9 = newey_west_tstat(x, lags=9)
    assert abs(t9) < abs(t0)  # HAC correction must widen the error bars


def test_ic_summary_fields(rng):
    prices = _prices(rng)
    fwd = forward_returns(prices, 1)
    s = ic_summary(ic_series(fwd, fwd))
    assert s["ic_mean"] == pytest.approx(1.0)
    assert s["pct_positive"] == 1.0
    assert set(s.index) == {
        "ic_mean", "ic_std", "ic_ir", "t_stat_nw", "pct_positive", "n_periods",
    }


def test_quantile_returns_monotone_for_perfect_signal(rng):
    prices = _prices(rng, n_sym=20)
    fwd = forward_returns(prices, 1)
    q = quantile_returns(fwd, fwd, n_quantiles=5)
    means = q.mean()
    assert list(means.index) == [1, 2, 3, 4, 5]
    assert means.is_monotonic_increasing
    spread = quantile_spread(q)
    assert spread.mean() > 0
    assert (spread.dropna() >= 0).all()


def test_quantile_returns_handles_nans(rng):
    prices = _prices(rng, n_sym=12)
    fwd = forward_returns(prices, 1)
    signal = fwd.copy()
    signal.iloc[:, :3] = np.nan  # only 9 usable names
    q = quantile_returns(signal, fwd, n_quantiles=3)
    assert q.dropna(how="all").shape[0] > 0


def test_signal_turnover_extremes(rng):
    idx = pd.bdate_range("2022-01-03", periods=50)
    cols = [f"S{i}" for i in range(10)]
    frozen = pd.DataFrame(
        np.tile(np.arange(10.0), (50, 1)), index=idx, columns=cols
    )
    assert signal_turnover(frozen).iloc[1:].max() == 0.0
    noisy = pd.DataFrame(
        rng.normal(size=(50, 10)), index=idx, columns=cols
    )
    assert signal_turnover(noisy).iloc[1:].mean() > 0.15


def test_ic_decay_shape(rng):
    prices = _prices(rng)
    fwd = forward_returns(prices, 1)
    decay = ic_decay(fwd, prices, horizons=(1, 5, 10))
    assert list(decay.index) == [1, 5, 10]
    assert decay.loc[1] == pytest.approx(1.0)
    assert decay.loc[10] < decay.loc[1]  # a 1-day signal fades at 10 days


def test_evaluate_signal_bundle(rng):
    prices = _prices(rng)
    fwd = forward_returns(prices, 1)
    ev = evaluate_signal(fwd, prices, name="oracle", horizon=1, n_quantiles=4)
    assert ev.name == "oracle"
    assert ev.summary["ic_mean"] == pytest.approx(1.0)
    head = ev.headline()
    assert head.name == "oracle"
    assert {"ic_mean", "ic_ir", "t_stat_nw", "spread_mean", "rank_turnover"} == set(
        head.index
    )


def test_misaligned_inputs_raise():
    idx = pd.bdate_range("2022-01-03", periods=5)
    a = pd.DataFrame(1.0, index=idx, columns=["X", "Y"])
    with pytest.raises(DataError, match="overlap"):
        ic_series(a, a)  # fewer than 3 shared columns
    with pytest.raises(DataError):
        forward_returns(a, horizon=0)
