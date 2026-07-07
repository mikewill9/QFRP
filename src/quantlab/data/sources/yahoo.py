"""Yahoo Finance data source.

Requires the optional ``yfinance`` dependency (``pip install quantlab[data]``).
Prices are requested with ``auto_adjust=True`` so the OHLC series are
split/dividend adjusted, consistent with the framework-wide convention.

This source performs a network call on every ``fetch``. Always wrap it in
:class:`~quantlab.data.cache.CachedSource` for research work::

    source = CachedSource(YahooFinanceSource())
    ds = source.fetch(["AAPL", "MSFT"], "2015-01-01", "2024-12-31")
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quantlab.data.dataset import FIELDS, MarketDataset
from quantlab.data.sources.base import DataSource, validate_request
from quantlab.exceptions import DataError
from quantlab.log import get_logger
from quantlab.types import Frequency

__all__ = ["YahooFinanceSource"]

logger = get_logger(__name__)

_INTERVAL = {
    Frequency.DAILY: "1d",
    Frequency.WEEKLY: "1wk",
    Frequency.MONTHLY: "1mo",
}
_YF_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
}


class YahooFinanceSource(DataSource):
    """Fetch adjusted OHLCV bars from Yahoo Finance via ``yfinance``.

    Parameters
    ----------
    strict:
        If True (default), symbols that return no data raise
        :class:`~quantlab.exceptions.DataError`; otherwise they are dropped
        with a warning.
    """

    name = "yahoo"

    def __init__(self, strict: bool = True) -> None:
        self.strict = bool(strict)

    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        frequency: Frequency = Frequency.DAILY,
    ) -> MarketDataset:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise DataError(
                "YahooFinanceSource requires 'yfinance'; install quantlab[data]"
            ) from exc

        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        raw = yf.download(
            tickers=symbols_list,
            start=start_ts,
            end=end_ts + pd.Timedelta(days=1),  # yfinance end is exclusive
            interval=_INTERVAL[frequency],
            auto_adjust=True,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if raw is None or raw.empty:
            raise DataError(f"Yahoo returned no data for {symbols_list[:5]}")

        frames: dict[str, pd.DataFrame] = {}
        for sym in symbols_list:
            block = self._symbol_block(raw, sym, n_requested=len(symbols_list))
            if block is None:
                if self.strict:
                    raise DataError(f"Yahoo returned no data for symbol '{sym}'")
                logger.warning("Dropping symbol with no Yahoo data: %s", sym)
                continue
            frames[sym] = block
        if not frames:
            raise DataError("Yahoo returned no usable data for any symbol")

        ds = MarketDataset.from_symbol_frames(frames, frequency=frequency)
        return ds.between(start_ts, end_ts).with_metadata(
            source=self.name, adjusted=True
        )

    @staticmethod
    def _symbol_block(
        raw: pd.DataFrame, symbol: str, n_requested: int
    ) -> pd.DataFrame | None:
        """Extract one symbol's OHLCV from yfinance's (multi)frame output."""
        if n_requested == 1 and not isinstance(raw.columns, pd.MultiIndex):
            block = raw
        else:
            if symbol not in raw.columns.get_level_values(0):
                return None
            block = raw[symbol]
        out = pd.DataFrame(index=block.index)
        for name in FIELDS:
            col = _YF_COLUMNS[name]
            out[name] = pd.to_numeric(block[col], errors="coerce") if col in block else float("nan")
        if out["close"].dropna().empty:
            return None
        idx = pd.DatetimeIndex(out.index)
        if idx.tz is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        out.index = idx.normalize()
        out = out[~out.index.duplicated(keep="last")].sort_index()
        return out
