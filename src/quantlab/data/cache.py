"""Parquet cache with incremental updates.

:class:`CachedSource` wraps any :class:`~quantlab.data.sources.base.DataSource`
and persists per-symbol OHLCV history under::

    {cache_dir}/{source.name}/{frequency}/{symbol}.parquet   # bars
    {cache_dir}/{source.name}/{frequency}/{symbol}.meta.json # covered range

On ``fetch`` it computes, per symbol, the missing head and/or tail relative
to the **covered range** (the union of previously requested windows, tracked
in the sidecar), fetches only those gaps from the wrapped source (batched by
identical gap), merges, and rewrites atomically. Tracking covered ranges
rather than bar ranges means a request whose tail contains no new bars (a
weekend, a holiday) is remembered and not refetched on the next call.

Corrupt cache files are treated as missing (logged and refetched) rather than
fatal, so a damaged cache degrades to slower fetches instead of broken
research runs.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from quantlab.data.dataset import MarketDataset
from quantlab.data.sources.base import DataSource, validate_request
from quantlab.data.store import read_symbol_frame, write_symbol_frame
from quantlab.exceptions import CacheError, DataError
from quantlab.log import get_logger
from quantlab.types import Frequency

__all__ = ["CachedSource"]

logger = get_logger(__name__)

_ONE_DAY = pd.Timedelta(days=1)


def _safe_symbol(symbol: str) -> str:
    """Filesystem-safe file stem for a symbol."""
    return "".join(c if (c.isalnum() or c in "._-^=") else "_" for c in symbol)


class CachedSource(DataSource):
    """Transparent caching wrapper around another data source.

    Parameters
    ----------
    source:
        The wrapped source; its ``name`` keys the cache directory.
    cache_dir:
        Cache root. Defaults to ``Settings.cache_dir``.
    """

    def __init__(self, source: DataSource, cache_dir: Path | str | None = None) -> None:
        if cache_dir is None:
            from quantlab.config import get_settings

            cache_dir = get_settings().cache_dir
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.name = f"cached[{source.name}]"

    # ----------------------------------------------------------- filesystem
    def _dir(self, frequency: Frequency) -> Path:
        return self.cache_dir / self.source.name / frequency.value

    def _bar_path(self, symbol: str, frequency: Frequency) -> Path:
        return self._dir(frequency) / f"{_safe_symbol(symbol)}.parquet"

    def _meta_path(self, symbol: str, frequency: Frequency) -> Path:
        return self._dir(frequency) / f"{_safe_symbol(symbol)}.meta.json"

    def _read_cached(
        self, symbol: str, frequency: Frequency
    ) -> tuple[pd.DataFrame | None, tuple[pd.Timestamp, pd.Timestamp] | None]:
        """Return (bars, covered_range) or (None, None) if absent/corrupt."""
        bar_path = self._bar_path(symbol, frequency)
        if not bar_path.is_file():
            return None, None
        try:
            frame = read_symbol_frame(bar_path)
        except CacheError as exc:
            logger.warning("Corrupt cache for %s (%s); refetching", symbol, exc)
            return None, None
        covered = None
        meta_path = self._meta_path(symbol, frequency)
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                covered = (
                    pd.Timestamp(meta["covered_start"]),
                    pd.Timestamp(meta["covered_end"]),
                )
            except (KeyError, ValueError, json.JSONDecodeError):
                covered = None
        if covered is None and len(frame):
            covered = (frame.index[0], frame.index[-1])
        return frame, covered

    def _write_cached(
        self,
        symbol: str,
        frequency: Frequency,
        frame: pd.DataFrame,
        covered: tuple[pd.Timestamp, pd.Timestamp],
    ) -> None:
        write_symbol_frame(self._bar_path(symbol, frequency), frame)
        meta = {
            "covered_start": str(covered[0].date()),
            "covered_end": str(covered[1].date()),
        }
        meta_path = self._meta_path(symbol, frequency)
        fd, tmp_name = tempfile.mkstemp(dir=meta_path.parent, suffix=".json.tmp")
        os.close(fd)
        tmp = Path(tmp_name)
        try:
            tmp.write_text(json.dumps(meta))
            os.replace(tmp, meta_path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise CacheError(f"Failed writing cache meta for {symbol}: {exc}") from exc

    # ------------------------------------------------------------------ api
    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        frequency: Frequency = Frequency.DAILY,
    ) -> MarketDataset:
        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        self._dir(frequency).mkdir(parents=True, exist_ok=True)

        bars: dict[str, pd.DataFrame | None] = {}
        covered: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {}
        gaps: dict[tuple[pd.Timestamp, pd.Timestamp], list[str]] = {}

        for sym in symbols_list:
            frame, cov = self._read_cached(sym, frequency)
            bars[sym], covered[sym] = frame, cov
            for gap in self._missing_ranges(cov, start_ts, end_ts):
                gaps.setdefault(gap, []).append(sym)

        for (gap_start, gap_end), gap_symbols in sorted(gaps.items()):
            logger.info(
                "Cache miss [%s..%s] for %d symbol(s) via %s",
                gap_start.date(),
                gap_end.date(),
                len(gap_symbols),
                self.source.name,
            )
            fetched = self.source.fetch(gap_symbols, gap_start, gap_end, frequency)
            for sym in gap_symbols:
                if sym in fetched.symbols:
                    new = fetched.symbol_frame(sym).dropna(how="all")
                    merged = self._merge(bars[sym], new)
                else:
                    merged = bars[sym]
                bars[sym] = merged
                covered[sym] = self._extend_covered(covered[sym], gap_start, gap_end)

        result_frames: dict[str, pd.DataFrame] = {}
        for sym in symbols_list:
            frame, cov = bars[sym], covered[sym]
            if frame is None or frame.empty:
                logger.warning("No data available for %s in requested range", sym)
                continue
            self._write_cached(sym, frequency, frame, cov or (start_ts, end_ts))
            window = frame.loc[start_ts:end_ts]
            if not window.empty:
                result_frames[sym] = window
        if not result_frames:
            raise DataError(
                f"No bars in [{start_ts.date()}, {end_ts.date()}] for any of {symbols_list[:5]}"
            )
        ds = MarketDataset.from_symbol_frames(result_frames, frequency=frequency)
        return ds.with_metadata(source=self.source.name, cached=True)

    # --------------------------------------------------------------- ranges
    @staticmethod
    def _missing_ranges(
        covered: tuple[pd.Timestamp, pd.Timestamp] | None,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        if covered is None:
            return [(start, end)]
        c0, c1 = covered
        out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        if start < c0:
            out.append((start, c0 - _ONE_DAY))
        if end > c1:
            out.append((c1 + _ONE_DAY, end))
        return out

    @staticmethod
    def _extend_covered(
        covered: tuple[pd.Timestamp, pd.Timestamp] | None,
        gap_start: pd.Timestamp,
        gap_end: pd.Timestamp,
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        if covered is None:
            return gap_start, gap_end
        return min(covered[0], gap_start), max(covered[1], gap_end)

    @staticmethod
    def _merge(old: pd.DataFrame | None, new: pd.DataFrame) -> pd.DataFrame:
        if old is None or old.empty:
            return new
        if new.empty:
            return old
        combined = pd.concat([old, new])
        combined = combined[~combined.index.duplicated(keep="last")]
        return combined.sort_index()

    # ------------------------------------------------------------ utilities
    def cached_range(
        self, symbol: str, frequency: Frequency = Frequency.DAILY
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Covered (requested-and-stored) date range for ``symbol``, if any."""
        _, cov = self._read_cached(symbol, frequency)
        return cov

    def clear(
        self,
        symbols: Sequence[str] | None = None,
        frequency: Frequency = Frequency.DAILY,
    ) -> int:
        """Delete cached files; all symbols when ``symbols`` is None.

        Returns the number of parquet files removed.
        """
        d = self._dir(frequency)
        if not d.exists():
            return 0
        if symbols is None:
            targets = list(d.glob("*.parquet"))
        else:
            targets = [self._bar_path(s, frequency) for s in symbols]
        removed = 0
        for bar_path in targets:
            if bar_path.is_file():
                bar_path.unlink()
                removed += 1
            meta = bar_path.parent / (bar_path.stem + ".meta.json")
            meta.unlink(missing_ok=True)
        return removed
