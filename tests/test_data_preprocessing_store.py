"""Preprocessing transforms, DatasetStore persistence, CSV ingestion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data import (
    CSVSource,
    DatasetStore,
    MarketDataset,
    align_intersection,
    compute_returns,
    filter_universe,
    forward_fill_prices,
    resample_dataset,
)
from quantlab.exceptions import DataError
from quantlab.types import Frequency

# ------------------------------------------------------------- preprocessing


def test_compute_returns_matches_hand_math():
    idx = pd.bdate_range("2022-01-03", periods=3)
    prices = pd.DataFrame({"X": [100.0, 110.0, 99.0]}, index=idx)
    simple = compute_returns(prices, "simple")["X"]
    log = compute_returns(prices, "log")["X"]
    assert simple.iloc[1] == pytest.approx(0.10)
    assert simple.iloc[2] == pytest.approx(99 / 110 - 1)
    assert log.iloc[2] == pytest.approx(np.log(99 / 110))


def test_forward_fill_respects_limit_and_zeroes_volume(small_synth):
    from dataclasses import replace

    close = small_synth.close.copy()
    close.iloc[10:13, 0] = np.nan  # 3-bar hole: fillable at limit=5
    close.iloc[40:50, 0] = np.nan  # 10-bar hole: partially filled only
    ds = replace(small_synth, close=close)
    filled = forward_fill_prices(ds, limit=5)
    assert filled.close.iloc[10:13, 0].notna().all()
    assert (filled.volume.iloc[10:13, 0] == 0.0).all()
    assert filled.close.iloc[46:50, 0].isna().all()  # beyond the fill limit


def test_resample_weekly_ohlcv_semantics():
    idx = pd.bdate_range("2022-01-03", periods=10)  # two full Mon-Fri weeks
    close = pd.DataFrame({"X": np.arange(1.0, 11.0)}, index=idx)
    ds = MarketDataset(
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=close * 0 + 10.0,
    )
    wk = resample_dataset(ds, Frequency.WEEKLY)
    assert wk.n_periods == 2
    assert wk.open.iloc[0, 0] == 0.5  # first open of week 1
    assert wk.close.iloc[0, 0] == 5.0  # last close of week 1
    assert wk.high.iloc[0, 0] == 6.0  # max high of week 1
    assert wk.low.iloc[0, 0] == 0.0  # min low of week 1
    assert wk.volume.iloc[0, 0] == 50.0  # summed volume
    with pytest.raises(DataError, match="upsample"):
        resample_dataset(wk, Frequency.DAILY)


def test_align_intersection(small_synth):
    a = small_synth.between("2021-01-01", "2021-04-30").select(small_synth.symbols[:4])
    b = small_synth.between("2021-03-01", "2021-06-30").select(small_synth.symbols[2:])
    aa, bb = align_intersection(a, b)
    assert aa.index.equals(bb.index)
    assert aa.symbols == bb.symbols == small_synth.symbols[2:4]
    assert aa.start >= pd.Timestamp("2021-03-01")
    assert aa.end <= pd.Timestamp("2021-04-30")


def test_filter_universe_liquidity(synth):
    med_dv = synth.dollar_volume().median(axis=0)
    cutoff = float(med_dv.median())
    filtered = filter_universe(synth, min_median_dollar_volume=cutoff)
    assert 0 < filtered.n_symbols < synth.n_symbols
    assert all(med_dv[s] >= cutoff for s in filtered.symbols)
    with pytest.raises(DataError, match="every symbol"):
        filter_universe(synth, min_median_dollar_volume=1e18)


# ---------------------------------------------------------------------- store


def test_dataset_store_round_trip(tmp_path, small_synth):
    store = DatasetStore(tmp_path)
    store.save(small_synth, "universes/test")
    assert store.exists("universes/test")
    assert store.list() == ["universes/test"]
    loaded = store.load("universes/test")
    assert loaded.equals(small_synth)
    assert loaded.frequency is small_synth.frequency
    assert loaded.metadata["source"] == "synthetic"


def test_dataset_store_missing(tmp_path):
    store = DatasetStore(tmp_path)
    from quantlab.exceptions import CacheError

    with pytest.raises(CacheError, match="No dataset"):
        store.load("nope")


# ------------------------------------------------------------------ csv source


def test_csv_directory_mode(tmp_path, small_synth):
    for sym in small_synth.symbols[:3]:
        frame = small_synth.symbol_frame(sym).reset_index(names="date")
        frame.to_csv(tmp_path / f"{sym}.csv", index=False)
    src = CSVSource(tmp_path)
    ds = src.fetch(small_synth.symbols[:3], small_synth.start, small_synth.end)
    ref = small_synth.select(small_synth.symbols[:3])
    pd.testing.assert_frame_equal(ds.close, ref.close, check_freq=False)
    with pytest.raises(DataError, match="Missing CSV"):
        src.fetch(["GHOST"], small_synth.start, small_synth.end)


def test_csv_long_mode_with_column_map(tmp_path, small_synth):
    long = small_synth.to_long().rename(
        columns={"date": "Date", "symbol": "Ticker", "close": "Adj Close"}
    )
    path = tmp_path / "prices.csv"
    long.to_csv(path, index=False)
    src = CSVSource(
        path,
        column_map={"date": "Date", "symbol": "Ticker", "close": "Adj Close"},
    )
    ds = src.fetch(small_synth.symbols, small_synth.start, small_synth.end)
    pd.testing.assert_frame_equal(
        ds.close[small_synth.symbols], small_synth.close, check_freq=False
    )
