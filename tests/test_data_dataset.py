"""MarketDataset invariants and round-trips."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.data import FIELDS, MarketDataset
from quantlab.exceptions import DataError
from quantlab.types import Frequency


def _tiny_frames(n_t: int = 4, symbols: tuple[str, ...] = ("AAA", "BBB")):
    idx = pd.bdate_range("2022-01-03", periods=n_t)
    base = pd.DataFrame(
        np.arange(1, n_t * len(symbols) + 1, dtype=float).reshape(n_t, len(symbols)),
        index=idx,
        columns=list(symbols),
    )
    return {
        "open": base * 0.99,
        "high": base * 1.02,
        "low": base * 0.98,
        "close": base,
        "volume": base * 100,
    }


def test_construction_and_properties():
    ds = MarketDataset(**_tiny_frames())
    assert ds.symbols == ["AAA", "BBB"]
    assert ds.n_periods == 4
    assert ds.start < ds.end
    assert set(ds.fields) == set(FIELDS)


def test_misaligned_index_rejected():
    frames = _tiny_frames()
    frames["open"] = frames["open"].iloc[:-1]
    with pytest.raises(DataError, match="index differs"):
        MarketDataset(**frames)


def test_duplicate_index_rejected():
    frames = _tiny_frames()
    idx = frames["close"].index.tolist()
    idx[1] = idx[0]
    for name in FIELDS:
        frames[name].index = pd.DatetimeIndex(idx)
    with pytest.raises(DataError, match="duplicate"):
        MarketDataset(**frames)


def test_tz_aware_index_rejected():
    frames = _tiny_frames()
    for name in FIELDS:
        frames[name].index = frames[name].index.tz_localize("UTC")
    with pytest.raises(DataError, match="tz-naive"):
        MarketDataset(**frames)


def test_returns_log_and_simple():
    ds = MarketDataset(**_tiny_frames())
    p = ds.close["AAA"]
    simple = ds.returns(method="simple")["AAA"]
    log = ds.returns(method="log")["AAA"]
    assert np.isnan(simple.iloc[0])
    assert simple.iloc[1] == pytest.approx(p.iloc[1] / p.iloc[0] - 1)
    assert log.iloc[1] == pytest.approx(np.log(p.iloc[1] / p.iloc[0]))
    with pytest.raises(DataError):
        ds.returns(method="arithmetic")


def test_select_and_between(synth):
    sub = synth.select(["SYN001", "SYN000"])
    assert sub.symbols == ["SYN001", "SYN000"]
    window = synth.between("2020-06-01", "2020-06-30")
    assert window.start >= pd.Timestamp("2020-06-01")
    assert window.end <= pd.Timestamp("2020-06-30")
    with pytest.raises(DataError):
        synth.select(["NOPE"])


def test_long_round_trip(small_synth):
    long = small_synth.to_long()
    assert set(long.columns) == {"date", "symbol", *FIELDS}
    rebuilt = MarketDataset.from_long(long, frequency=small_synth.frequency)
    aligned = rebuilt.select(small_synth.symbols)
    assert aligned.equals(small_synth)


def test_from_long_column_mapping():
    idx = pd.bdate_range("2022-01-03", periods=3)
    rows = []
    for i, d in enumerate(idx):
        for s, base in (("X", 10.0), ("Y", 20.0)):
            rows.append({"dt": d, "ticker": s, "px": base + i})
    long = pd.DataFrame(rows)
    ds = MarketDataset.from_long(
        long, date_col="dt", symbol_col="ticker", field_cols={"close": "px"}
    )
    assert ds.close.loc[idx[2], "Y"] == 22.0
    assert ds.open.isna().all().all()  # unmapped fields become NaN


def test_symbol_frame_round_trip(small_synth):
    frames = {s: small_synth.symbol_frame(s) for s in small_synth.symbols}
    rebuilt = MarketDataset.from_symbol_frames(frames)
    assert rebuilt.select(small_synth.symbols).equals(small_synth)


def test_drop_incomplete_symbols(small_synth):
    close = small_synth.close.copy()
    sym = small_synth.symbols[0]
    close.iloc[: len(close) // 2, close.columns.get_loc(sym)] = np.nan
    frames = {name: small_synth.field(name).copy() for name in FIELDS}
    frames["close"] = close
    ds = MarketDataset(**frames, frequency=Frequency.DAILY)
    kept = ds.drop_incomplete_symbols(max_missing_frac=0.10)
    assert sym not in kept.symbols
    assert len(kept.symbols) == len(small_synth.symbols) - 1
