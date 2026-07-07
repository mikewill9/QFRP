"""Factors: hand-computed values, sign conventions, warmup NaN discipline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data import MarketDataset
from quantlab.exceptions import DataError
from quantlab.research import (
    AmihudIlliquidity,
    DollarVolume,
    EWMAVolatility,
    MarketBeta,
    Momentum,
    PCAResidualReversal,
    PriceToMovingAverage,
    RealizedVolatility,
    RollSpread,
    ShortTermReversal,
    TimeSeriesMomentum,
)

ALL_FACTORS = [
    Momentum(lookback=63, skip=5),
    TimeSeriesMomentum(lookback=63),
    ShortTermReversal(lookback=5),
    PriceToMovingAverage(window=21),
    RealizedVolatility(window=21),
    RealizedVolatility(window=21, estimator="parkinson"),
    RealizedVolatility(window=21, estimator="garman_klass"),
    EWMAVolatility(halflife=20),
    AmihudIlliquidity(window=21),
    DollarVolume(window=21),
    RollSpread(window=21),
    MarketBeta(window=63),
    PCAResidualReversal(window=63, n_components=2, stride=10),
]


def _trend_dataset(rates: dict[str, float], n: int = 300) -> MarketDataset:
    """Deterministic exponential-trend market: close_i(t) = 100 * e^(a_i t)."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    t = np.arange(n, dtype=float)
    close = pd.DataFrame(
        {sym: 100.0 * np.exp(a * t) for sym, a in rates.items()}, index=idx
    )
    return MarketDataset(
        open=close * 0.999,
        high=close * 1.001,
        low=close * 0.998,
        close=close,
        volume=close * 0 + 1e6,
    )


@pytest.mark.parametrize("factor", ALL_FACTORS, ids=lambda f: repr(f))
def test_shape_alignment_and_warmup(synth, factor):
    scores = factor.compute(synth)
    assert scores.index.equals(synth.index)
    assert list(scores.columns) == synth.symbols
    # Warmup rows must be NaN: nothing before min_history may be scored.
    warmup = scores.iloc[: factor.min_history - 1]
    assert warmup.isna().all().all(), f"{factor.name} leaked scores into warmup"
    # And after warmup there must be real values.
    assert scores.iloc[factor.min_history + 5 :].notna().any().any()


def test_momentum_hand_computed():
    ds = _trend_dataset({"FAST": 0.002, "SLOW": 0.0005})
    factor = Momentum(lookback=60, skip=10)
    scores = factor.compute(ds)
    # log close = log(100) + a*t, so score = a * (lookback - skip) exactly.
    expected_fast = 0.002 * 50
    assert scores["FAST"].iloc[-1] == pytest.approx(expected_fast, rel=1e-9)
    assert scores["SLOW"].iloc[-1] == pytest.approx(0.0005 * 50, rel=1e-9)
    assert (scores["FAST"].dropna() > scores["SLOW"].dropna()).all()


def test_momentum_skip_excludes_recent_window():
    """A huge move inside the skip window must not affect the score."""
    ds = _trend_dataset({"X": 0.001}, n=200)
    close = ds.close.copy()
    close.iloc[-3:, 0] *= 5.0  # violent move in the last 3 bars
    spiked = MarketDataset(
        open=close * 0.999, high=close * 1.001, low=close * 0.998,
        close=close, volume=ds.volume,
    )
    base = Momentum(lookback=60, skip=10).compute(ds)["X"].iloc[-1]
    spiked_score = Momentum(lookback=60, skip=10).compute(spiked)["X"].iloc[-1]
    assert spiked_score == pytest.approx(base, rel=1e-12)


def test_reversal_is_negative_momentum_window():
    ds = _trend_dataset({"UP": 0.003, "DOWN": -0.003})
    scores = ShortTermReversal(lookback=5).compute(ds)
    assert (scores["DOWN"].dropna() > 0).all()  # recent loser scores high
    assert (scores["UP"].dropna() < 0).all()
    assert scores["UP"].iloc[-1] == pytest.approx(-0.003 * 5, rel=1e-9)


def test_price_to_ma_sign():
    idx = pd.bdate_range("2020-01-01", periods=60)
    flat = np.full(60, 100.0)
    dipped = flat.copy()
    dipped[-1] = 90.0  # sudden dip below its MA
    close = pd.DataFrame({"FLAT": flat, "DIP": dipped}, index=idx)
    close += np.linspace(0, 0.5, 60)[:, None]  # tiny slope so std > 0
    ds = MarketDataset(
        open=close, high=close * 1.001, low=close * 0.999, close=close,
        volume=close * 0 + 1e5,
    )
    scores = PriceToMovingAverage(window=21).compute(ds)
    assert scores["DIP"].iloc[-1] > 2.0  # far below MA => strongly attractive


def test_realized_vol_ranks_low_vol_high(synth):
    scores = RealizedVolatility(window=42).compute(synth)
    vols = synth.returns().std()
    # invert=True: the calmest asset should outrank the wildest on average.
    calm, wild = vols.idxmin(), vols.idxmax()
    assert scores[calm].mean() > scores[wild].mean()
    raw = RealizedVolatility(window=42, invert=False).compute(synth)
    assert (raw.stack().dropna() > 0).all()


def test_vol_estimators_agree_in_scale(synth):
    cc = RealizedVolatility(window=42, invert=False).compute(synth)
    pk = RealizedVolatility(window=42, estimator="parkinson", invert=False).compute(synth)
    gk = RealizedVolatility(window=42, estimator="garman_klass", invert=False).compute(synth)
    m_cc, m_pk, m_gk = (f.mean().mean() for f in (cc, pk, gk))
    # Range-based estimators measure the same object; same order of magnitude.
    assert 0.2 < m_pk / m_cc < 5.0
    assert 0.2 < m_gk / m_cc < 5.0


def test_market_beta_of_market_clone(synth):
    """An asset that IS the equal-weight market must have beta ~ 1."""
    rets = synth.returns()
    mkt = rets.mean(axis=1)
    clone_logp = mkt.fillna(0.0).cumsum()
    close = synth.close.copy()
    close["CLONE"] = 100.0 * np.exp(clone_logp)
    frames = {
        "open": close * 0.999, "high": close * 1.001,
        "low": close * 0.998, "close": close, "volume": close * 0 + 1e6,
    }
    ds = MarketDataset(**frames)
    beta = MarketBeta(window=63, invert=False).compute(ds)
    # CLONE is one of n+1 assets so the new market is slightly diluted;
    # beta should still hover very near 1.
    assert beta["CLONE"].dropna().mean() == pytest.approx(1.0, abs=0.1)


def test_amihud_ranks_illiquid_high(small_synth):
    volume = small_synth.volume.copy()
    thin, thick = small_synth.symbols[0], small_synth.symbols[1]
    volume[thin] = volume[thin] * 0.001
    volume[thick] = volume[thick] * 1000.0
    from dataclasses import replace

    ds = replace(small_synth, volume=volume)
    scores = AmihudIlliquidity(window=21).compute(ds)
    assert scores[thin].dropna().mean() > scores[thick].dropna().mean()


def test_roll_spread_detects_bid_ask_bounce():
    """Alternating bounce around a flat mid implies a wide Roll spread."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=250)
    n = len(idx)
    bounce = 100.0 + 0.5 * np.where(np.arange(n) % 2 == 0, 1, -1)
    smooth = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    close = pd.DataFrame({"BOUNCE": bounce, "SMOOTH": smooth}, index=idx)
    ds = MarketDataset(
        open=close, high=close * 1.001 + 1, low=close * 0.999 - 1,
        close=close, volume=close * 0 + 1e5,
    )
    scores = RollSpread(window=42).compute(ds)
    # Bouncy name has a strongly negative score (wide spread), smooth near 0.
    assert scores["BOUNCE"].dropna().mean() < scores["SMOOTH"].dropna().mean()
    assert scores["BOUNCE"].dropna().mean() < -0.005


def test_pca_residual_reversal_deterministic(synth):
    f = PCAResidualReversal(window=63, n_components=2, stride=10)
    a = f.compute(synth)
    b = PCAResidualReversal(window=63, n_components=2, stride=10).compute(synth)
    pd.testing.assert_frame_equal(a, b)
    valid = a.iloc[f.min_history + 5 :]
    assert np.isfinite(valid.to_numpy()).all()
    # Cross-sectional dispersion exists (not a degenerate constant signal).
    assert (valid.std(axis=1) > 0).all()


def test_params_echo_and_validation():
    f = Momentum(lookback=126, skip=21)
    assert f.params == {"lookback": 126, "skip": 21, "min_history": 127}
    with pytest.raises(DataError):
        Momentum(lookback=10, skip=10)
    with pytest.raises(DataError):
        RealizedVolatility(estimator="yang_zhang")
    with pytest.raises(DataError):
        PCAResidualReversal(window=10)


def test_insufficient_history_raises(small_synth):
    tiny = small_synth.between("2021-01-01", "2021-02-01")
    with pytest.raises(DataError, match="needs >="):
        Momentum(lookback=252, skip=21).compute(tiny)
