"""Synthetic source: determinism and structural realism."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data import SyntheticSource, make_synthetic_dataset
from quantlab.exceptions import DataError
from quantlab.types import Frequency


def test_deterministic_same_seed():
    a = make_synthetic_dataset(n_assets=6, start="2021-01-01", end="2021-03-31", seed=5)
    b = make_synthetic_dataset(n_assets=6, start="2021-01-01", end="2021-03-31", seed=5)
    assert a.equals(b)


def test_different_seed_differs():
    a = make_synthetic_dataset(n_assets=6, start="2021-01-01", end="2021-03-31", seed=5)
    b = make_synthetic_dataset(n_assets=6, start="2021-01-01", end="2021-03-31", seed=6)
    assert not a.close.equals(b.close)


def test_window_consistency_with_fixed_inception():
    """Overlapping fetches from the same source agree exactly on overlap."""
    src = SyntheticSource(n_assets=4, n_sectors=2, seed=3, inception="2021-01-01")
    full = src.fetch(src.symbols, "2021-01-01", "2021-06-30")
    tail = src.fetch(src.symbols, "2021-04-01", "2021-06-30")
    overlap = full.between("2021-04-01", "2021-06-30")
    assert overlap.equals(tail)


def test_symbol_subset_consistency():
    src = SyntheticSource(n_assets=6, n_sectors=2, seed=9, inception="2021-01-01")
    full = src.fetch(src.symbols, "2021-01-01", "2021-02-26")
    sub = src.fetch(["SYN002", "SYN005"], "2021-01-01", "2021-02-26")
    pd.testing.assert_frame_equal(full.close[["SYN002", "SYN005"]], sub.close)


def test_ohlc_internally_consistent(synth):
    assert (synth.high >= synth.open - 1e-12).all().all()
    assert (synth.high >= synth.close - 1e-12).all().all()
    assert (synth.low <= synth.open + 1e-12).all().all()
    assert (synth.low <= synth.close + 1e-12).all().all()
    assert (synth.low > 0).all().all()
    assert (synth.volume >= 0).all().all()


def test_business_day_index(synth):
    assert synth.index.dayofweek.max() <= 4  # no weekends
    assert synth.index.is_monotonic_increasing


def test_metadata_and_sector_structure(synth):
    sectors = synth.metadata["sectors"]
    assert set(sectors) == set(synth.symbols)
    assert len(set(sectors.values())) == 3
    # Same-sector average correlation should exceed cross-sector correlation.
    rets = synth.returns()
    corr = rets.corr()
    same, cross = [], []
    syms = synth.symbols
    for i, a in enumerate(syms):
        for b in syms[i + 1 :]:
            (same if sectors[a] == sectors[b] else cross).append(corr.loc[a, b])
    assert np.mean(same) > np.mean(cross)


def test_market_factor_creates_common_correlation(synth):
    rets = synth.returns()
    corr = rets.corr().to_numpy()
    off_diag = corr[~np.eye(corr.shape[0], dtype=bool)]
    assert off_diag.mean() > 0.05  # assets clearly co-move


def test_weekly_resampled_fetch():
    ds = make_synthetic_dataset(
        n_assets=4, start="2021-01-01", end="2021-06-30", seed=2,
        frequency=Frequency.WEEKLY,
    )
    assert ds.frequency is Frequency.WEEKLY
    assert ds.n_periods < 30


def test_request_before_inception_rejected():
    src = SyntheticSource(n_assets=3, n_sectors=1, seed=1, inception="2021-06-01")
    with pytest.raises(DataError, match="precedes inception"):
        src.fetch(src.symbols, "2021-01-01", "2021-12-31")


def test_unknown_symbol_rejected():
    src = SyntheticSource(n_assets=3, n_sectors=1, seed=1, inception="2021-01-01")
    with pytest.raises(DataError, match="Unknown synthetic"):
        src.fetch(["SYN999"], "2021-01-04", "2021-02-01")
