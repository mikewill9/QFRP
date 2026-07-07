"""CachedSource: incremental fetching, persistence, corruption resilience."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import pytest

from quantlab.data import CachedSource, MarketDataset
from quantlab.data.sources.base import DataSource, validate_request
from quantlab.types import Frequency


class RecordingSource(DataSource):
    """Deterministic in-memory source that records every fetch call."""

    name = "recording"

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], pd.Timestamp, pd.Timestamp]] = []

    def fetch(
        self,
        symbols: Sequence[str],
        start,
        end,
        frequency: Frequency = Frequency.DAILY,
    ) -> MarketDataset:
        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        self.calls.append((tuple(symbols_list), start_ts, end_ts))
        idx = pd.bdate_range(start_ts, end_ts)
        frames = {}
        for sym in symbols_list:
            base = float(sum(ord(c) for c in sym))
            px = pd.Series(
                base + (idx - pd.Timestamp("2020-01-01")).days.astype(float) * 0.1,
                index=idx,
            )
            frames[sym] = pd.DataFrame(
                {
                    "open": px * 0.999,
                    "high": px * 1.001,
                    "low": px * 0.998,
                    "close": px,
                    "volume": 1000.0,
                }
            )
        return MarketDataset.from_symbol_frames(frames, frequency=frequency)


@pytest.fixture
def cached(tmp_path):
    src = RecordingSource()
    return src, CachedSource(src, cache_dir=tmp_path)


def test_first_fetch_hits_source_second_is_pure_cache(cached):
    src, cache = cached
    a = cache.fetch(["AAA", "BBB"], "2021-01-04", "2021-03-31")
    assert len(src.calls) == 1
    b = cache.fetch(["AAA", "BBB"], "2021-01-04", "2021-03-31")
    assert len(src.calls) == 1  # no new upstream call
    assert a.equals(b)
    assert b.metadata["cached"] is True


def test_tail_extension_fetches_only_the_gap(cached):
    src, cache = cached
    cache.fetch(["AAA"], "2021-01-04", "2021-03-31")
    cache.fetch(["AAA"], "2021-01-04", "2021-06-30")
    assert len(src.calls) == 2
    _, gap_start, gap_end = src.calls[1]
    assert gap_start == pd.Timestamp("2021-04-01")
    assert gap_end == pd.Timestamp("2021-06-30")


def test_head_extension_fetches_only_the_gap(cached):
    src, cache = cached
    cache.fetch(["AAA"], "2021-03-01", "2021-06-30")
    cache.fetch(["AAA"], "2021-01-04", "2021-06-30")
    assert len(src.calls) == 2
    _, gap_start, gap_end = src.calls[1]
    assert gap_start == pd.Timestamp("2021-01-04")
    assert gap_end == pd.Timestamp("2021-02-28")


def test_interior_subwindow_is_pure_cache(cached):
    src, cache = cached
    full = cache.fetch(["AAA"], "2021-01-04", "2021-06-30")
    sub = cache.fetch(["AAA"], "2021-02-01", "2021-03-15")
    assert len(src.calls) == 1
    assert sub.equals(full.between("2021-02-01", "2021-03-15"))


def test_covered_range_prevents_weekend_thrash(cached):
    """A tail ending on a weekend must not refetch on repeat calls."""
    src, cache = cached
    cache.fetch(["AAA"], "2021-01-04", "2021-01-15")  # Friday
    cache.fetch(["AAA"], "2021-01-04", "2021-01-17")  # Sunday: gap has no bars
    n = len(src.calls)
    cache.fetch(["AAA"], "2021-01-04", "2021-01-17")
    assert len(src.calls) == n  # remembered as covered despite zero new bars


def test_mixed_symbols_group_gaps(cached):
    src, cache = cached
    cache.fetch(["AAA"], "2021-01-04", "2021-03-31")
    src.calls.clear()
    cache.fetch(["AAA", "BBB"], "2021-01-04", "2021-03-31")
    # AAA fully cached; BBB needs the whole window in exactly one call.
    assert src.calls == [(("BBB",), pd.Timestamp("2021-01-04"), pd.Timestamp("2021-03-31"))]


def test_persistence_across_instances(tmp_path):
    src1 = RecordingSource()
    CachedSource(src1, cache_dir=tmp_path).fetch(["AAA"], "2021-01-04", "2021-02-26")
    src2 = RecordingSource()
    out = CachedSource(src2, cache_dir=tmp_path).fetch(["AAA"], "2021-01-04", "2021-02-26")
    assert src2.calls == []  # served entirely from disk
    assert out.n_periods > 0


def test_corrupt_cache_is_refetched_not_fatal(cached, tmp_path):
    src, cache = cached
    cache.fetch(["AAA"], "2021-01-04", "2021-02-26")
    parquet = next(tmp_path.rglob("AAA.parquet"))
    parquet.write_bytes(b"not a parquet file")
    out = cache.fetch(["AAA"], "2021-01-04", "2021-02-26")
    assert len(src.calls) == 2  # refetched after detecting corruption
    assert out.n_periods > 0


def test_clear(cached, tmp_path):
    src, cache = cached
    cache.fetch(["AAA", "BBB"], "2021-01-04", "2021-02-26")
    assert cache.clear(["AAA"]) == 1
    assert cache.cached_range("AAA") is None
    assert cache.cached_range("BBB") is not None
    assert cache.clear() == 1  # remaining BBB
