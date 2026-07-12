"""Tests for point-in-time universe construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data import MarketDataset
from quantlab.data.universe import (
    US_LARGE_CAP_150,
    US_LARGE_CAP_SECTORS,
    apply_universe,
    rolling_universe,
)
from quantlab.exceptions import DataError


def _dataset(close: pd.DataFrame, volume: pd.DataFrame) -> MarketDataset:
    return MarketDataset(
        open=close * 0.999,
        high=close * 1.001,
        low=close * 0.998,
        close=close,
        volume=volume,
    )


def _flat_market(
    volumes: dict[str, float], n: int = 12, price: float = 10.0
) -> MarketDataset:
    """Constant prices, constant per-symbol volumes."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame({s: np.full(n, price) for s in volumes}, index=idx)
    vol = pd.DataFrame({s: np.full(n, v) for s, v in volumes.items()}, index=idx)
    return _dataset(close, vol)


def test_rolling_universe_hand_computed():
    # Dollar-volume order A > B > C > D at every bar.
    ds = _flat_market({"A": 1000.0, "B": 800.0, "C": 600.0, "D": 400.0}, n=12)
    mask = rolling_universe(ds, top_n=2, window=3, rebalance_every=5)

    assert mask.dtypes.eq(bool).all()
    assert mask.index.equals(ds.index)
    assert list(mask.columns) == ds.symbols
    # Before the first full window (bar index 2): all False.
    assert not mask.iloc[:2].any().any()
    # From bar 2 onward: exactly {A, B}.
    assert mask.iloc[2:][["A", "B"]].all().all()
    assert not mask.iloc[2:][["C", "D"]].any().any()


def test_rolling_universe_membership_flips_at_rebalance_only():
    n = 15
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(
        {s: np.full(n, 10.0) for s in ["A", "B", "C", "D"]}, index=idx
    )
    vol = pd.DataFrame(
        {
            "A": np.full(n, 1000.0),
            "B": np.full(n, 800.0),
            "C": np.full(n, 100.0),
            "D": np.full(n, 600.0),
        },
        index=idx,
    )
    vol.loc[idx[5]:, "D"] = 2000.0  # D overtakes B from bar 5 onward
    ds = _dataset(close, vol)
    mask = rolling_universe(ds, top_n=2, window=3, rebalance_every=5)

    # Rebalances at bars 2, 7, 12. Bar-2 ranks: A > B > D > C -> {A, B}.
    assert mask.iloc[2:7]["B"].all()
    assert not mask.iloc[2:7]["D"].any()
    # D's surge starts at bar 5 but membership holds until bar 7's rebalance.
    assert mask.iloc[7:]["D"].all()
    assert not mask.iloc[7:]["B"].any()
    assert mask.iloc[2:]["A"].all()


def test_rolling_universe_is_point_in_time():
    """Volume after a rebalance bar must not affect membership at that bar.

    Mirror of the momentum-skip test: perturb D's volume only on bars
    strictly after the bar-2 rebalance (and before the next rebalance at
    bar 7); the mask through bar 6 must be identical.
    """
    base = _flat_market({"A": 1000.0, "B": 800.0, "C": 600.0, "D": 400.0}, n=12)
    vol = base.volume.copy()
    vol.iloc[3:7, vol.columns.get_loc("D")] *= 1000.0
    perturbed = _dataset(base.close, vol)

    base_mask = rolling_universe(base, top_n=2, window=3, rebalance_every=5)
    perturbed_mask = rolling_universe(
        perturbed, top_n=2, window=3, rebalance_every=5
    )
    pd.testing.assert_frame_equal(base_mask.iloc[:7], perturbed_mask.iloc[:7])


def test_rolling_universe_price_floor_and_nan_handling():
    n = 8
    idx = pd.bdate_range("2020-01-01", periods=n)
    close = pd.DataFrame(
        {
            "CHEAP": np.full(n, 2.0),  # sub-$5 median despite huge volume
            "GOOD": np.full(n, 10.0),
            "NANV": np.full(n, 10.0),
            "OK2": np.full(n, 10.0),
        },
        index=idx,
    )
    vol = pd.DataFrame(
        {
            "CHEAP": np.full(n, 1e9),
            "GOOD": np.full(n, 500.0),
            "NANV": np.full(n, 450.0),
            "OK2": np.full(n, 400.0),
        },
        index=idx,
    )
    vol.iloc[0, vol.columns.get_loc("NANV")] = np.nan  # trailing mean NaN at bar 2
    ds = _dataset(close, vol)
    mask = rolling_universe(ds, top_n=3, window=3, rebalance_every=5)

    assert not mask["CHEAP"].any()  # price floor
    assert not mask.iloc[2:7]["NANV"].any()  # NaN trailing volume at bar 2
    assert mask.iloc[7:]["NANV"].all()  # full window again by bar 7
    assert mask.iloc[2:][["GOOD", "OK2"]].all().all()


def test_rolling_universe_top_n_exceeding_symbols_keeps_all():
    ds = _flat_market({"A": 1000.0, "B": 800.0, "C": 600.0, "D": 400.0}, n=10)
    mask = rolling_universe(ds, top_n=50, window=3, rebalance_every=5)
    assert mask.iloc[2:].all().all()
    assert not mask.iloc[:2].any().any()


def test_apply_universe_masks_and_passes_through():
    idx = pd.bdate_range("2020-01-01", periods=3)
    frame = pd.DataFrame(
        {"X": [1.0, 2.0, 3.0], "Y": [4.0, 5.0, 6.0], "Z": [7.0, 8.0, 9.0]},
        index=idx,
    )
    universe = pd.DataFrame(
        {"X": [True, False, True], "Y": [False, True, True]}, index=idx
    )
    out = apply_universe(frame, universe)

    assert out.loc[idx[0], "X"] == 1.0
    assert np.isnan(out.loc[idx[1], "X"])
    assert np.isnan(out.loc[idx[0], "Y"])
    assert out.loc[idx[2], "Y"] == 6.0
    assert out["Z"].isna().all()  # symbol the universe never admitted
    assert out.index.equals(frame.index)
    assert list(out.columns) == list(frame.columns)


def test_universe_constants_sane():
    assert len(US_LARGE_CAP_150) >= 100
    assert len(set(US_LARGE_CAP_150)) == len(US_LARGE_CAP_150)
    assert all(isinstance(t, str) and t for t in US_LARGE_CAP_150)
    assert set(US_LARGE_CAP_SECTORS) == set(US_LARGE_CAP_150)
    assert all(
        isinstance(s, str) and s for s in US_LARGE_CAP_SECTORS.values()
    )


def test_rolling_universe_validates_params():
    ds = _flat_market({"A": 1000.0, "B": 800.0}, n=10)
    with pytest.raises(DataError):
        rolling_universe(ds, top_n=0)
    with pytest.raises(DataError):
        rolling_universe(ds, window=0)
    with pytest.raises(DataError):
        rolling_universe(ds, rebalance_every=0)
