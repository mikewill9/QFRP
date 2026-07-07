"""DataSource interface.

A source turns ``(symbols, start, end, frequency)`` into a
:class:`~quantlab.data.dataset.MarketDataset`. Sources must:

- return **adjusted** prices (splits/dividends applied);
- return a tz-naive index restricted to ``[start, end]`` inclusive;
- omit symbols with no data in-range only when ``strict=False``, otherwise
  raise :class:`~quantlab.exceptions.DataError`;
- be deterministic for a fixed request wherever the upstream permits
  (the synthetic source is deterministic by construction).

Sources should NOT cache; wrap any source in
:class:`~quantlab.data.cache.CachedSource` for persistence and incremental
updates.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.types import Frequency

__all__ = ["DataSource", "normalize_date", "validate_request"]


def normalize_date(value: pd.Timestamp | str) -> pd.Timestamp:
    """Parse to a tz-naive, midnight-normalized Timestamp."""
    ts = pd.Timestamp(value)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def validate_request(
    symbols: Sequence[str],
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
) -> tuple[list[str], pd.Timestamp, pd.Timestamp]:
    """Normalize and sanity-check a fetch request.

    Returns de-duplicated symbols (order preserved) and normalized dates.
    """
    if not symbols:
        raise DataError("fetch() requires at least one symbol")
    seen: dict[str, None] = {}
    for s in symbols:
        if not isinstance(s, str) or not s:
            raise DataError(f"Invalid symbol: {s!r}")
        seen.setdefault(s, None)
    start_ts, end_ts = normalize_date(start), normalize_date(end)
    if start_ts > end_ts:
        raise DataError(f"start {start_ts.date()} is after end {end_ts.date()}")
    return list(seen), start_ts, end_ts


class DataSource(ABC):
    """Abstract market-data source. See module docstring for the contract."""

    #: Stable identifier used to key cache directories; subclasses override.
    name: str = "base"

    @abstractmethod
    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        frequency: Frequency = Frequency.DAILY,
    ) -> MarketDataset:
        """Fetch an aligned OHLCV dataset for ``symbols`` over ``[start, end]``."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
