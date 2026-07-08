"""MarketDataset: aligned multi-asset OHLCV container.

The dataset holds one wide frame per field (open/high/low/close/volume), all
sharing an identical, sorted, unique, tz-naive ``DatetimeIndex`` and identical
string symbol columns. Alignment is enforced at construction so downstream
code never re-checks it. Missing observations are represented as NaN; a
symbol that did not trade on a date simply carries NaN across all fields.

Prices are assumed to be **fully adjusted** (splits and dividends). Sources
are responsible for delivering adjusted series; the framework does not apply
corporate actions itself (documented in ``docs/architecture.md``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from quantlab.exceptions import DataError
from quantlab.types import Frequency, WideFrame

__all__ = ["FIELDS", "MarketDataset"]

#: Canonical field order for OHLCV data throughout the framework.
FIELDS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


def _check_index(index: pd.Index) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise DataError(f"Dataset index must be a DatetimeIndex, got {type(index).__name__}")
    if index.tz is not None:
        raise DataError("Dataset index must be tz-naive; strip timezones in the source")
    if not index.is_monotonic_increasing:
        raise DataError("Dataset index must be sorted ascending")
    if index.has_duplicates:
        dupes = index[index.duplicated()].unique()[:5].tolist()
        raise DataError(f"Dataset index has duplicate timestamps, e.g. {dupes}")


def _check_columns(columns: pd.Index) -> None:
    if columns.has_duplicates:
        dupes = columns[columns.duplicated()].unique()[:5].tolist()
        raise DataError(f"Dataset has duplicate symbol columns, e.g. {dupes}")
    if not all(isinstance(c, str) for c in columns):
        raise DataError("Dataset symbol columns must all be strings")


@dataclass(frozen=True, slots=True)
class MarketDataset:
    """Aligned multi-asset OHLCV panel.

    Parameters
    ----------
    open, high, low, close, volume:
        Wide frames (dates x symbols) with identical index and columns.
    frequency:
        Sampling frequency of the panel.
    metadata:
        Free-form provenance and universe metadata. Conventional keys used by
        the framework: ``"source"`` (str), ``"sectors"`` (dict symbol -> str),
        ``"asset_class"`` (:class:`~quantlab.types.AssetClass` value).
    """

    open: WideFrame
    high: WideFrame
    low: WideFrame
    close: WideFrame
    volume: WideFrame
    frequency: Frequency = Frequency.DAILY
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ init
    def __post_init__(self) -> None:
        ref = self.close
        _check_index(ref.index)
        _check_columns(ref.columns)
        for name in FIELDS:
            frame = getattr(self, name)
            if not isinstance(frame, pd.DataFrame):
                raise DataError(f"Field '{name}' must be a DataFrame")
            if not frame.index.equals(ref.index):
                raise DataError(f"Field '{name}' index differs from 'close'")
            if not frame.columns.equals(ref.columns):
                raise DataError(f"Field '{name}' columns differ from 'close'")
            if len(frame.columns) and not all(
                np.issubdtype(dt, np.number) for dt in frame.dtypes
            ):
                raise DataError(f"Field '{name}' contains non-numeric columns")

    # ------------------------------------------------------------ properties
    @property
    def index(self) -> pd.DatetimeIndex:
        """Shared date index."""
        return self.close.index

    @property
    def symbols(self) -> list[str]:
        """Symbols in column order."""
        return list(self.close.columns)

    @property
    def n_symbols(self) -> int:
        return self.close.shape[1]

    @property
    def n_periods(self) -> int:
        return self.close.shape[0]

    @property
    def start(self) -> pd.Timestamp:
        return self.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self.index[-1]

    @property
    def fields(self) -> dict[str, WideFrame]:
        """Mapping of field name -> wide frame (read-through view)."""
        return {name: getattr(self, name) for name in FIELDS}

    def field(self, name: str) -> WideFrame:
        """Return a single field frame by name (``'open'``..``'volume'``)."""
        if name not in FIELDS:
            raise DataError(f"Unknown field '{name}'; expected one of {FIELDS}")
        return getattr(self, name)

    # ------------------------------------------------------------ derivation
    def returns(
        self,
        field_name: str = "close",
        method: str = "log",
        periods: int = 1,
    ) -> WideFrame:
        """Compute per-symbol returns over ``periods`` bars.

        Parameters
        ----------
        field_name:
            Price field to difference (usually ``"close"``).
        method:
            ``"log"`` for ``ln(p_t / p_{t-k})`` or ``"simple"`` for
            ``p_t / p_{t-k} - 1``.
        periods:
            Horizon in bars (``k`` above); must be positive.
        """
        if periods < 1:
            raise DataError("periods must be >= 1")
        prices = self.field(field_name)
        shifted = prices.shift(periods)
        if method == "log":
            return np.log(prices / shifted)
        if method == "simple":
            return prices / shifted - 1.0
        raise DataError(f"Unknown return method '{method}' (use 'log' or 'simple')")

    def dollar_volume(self) -> WideFrame:
        """Close price times share volume, per symbol per bar."""
        return self.close * self.volume

    # ------------------------------------------------------------- selection
    def select(self, symbols: Sequence[str]) -> MarketDataset:
        """Return a dataset restricted to ``symbols`` (order preserved)."""
        missing = [s for s in symbols if s not in self.close.columns]
        if missing:
            raise DataError(f"Symbols not in dataset: {missing[:10]}")
        cols = list(symbols)
        return replace(self, **{name: getattr(self, name)[cols] for name in FIELDS})

    def between(
        self,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
    ) -> MarketDataset:
        """Return the dataset sliced to ``[start, end]`` (inclusive)."""
        return replace(self, **{name: getattr(self, name).loc[start:end] for name in FIELDS})

    def drop_incomplete_symbols(self, max_missing_frac: float = 0.0) -> MarketDataset:
        """Drop symbols whose close series exceeds ``max_missing_frac`` NaN."""
        frac = self.close.isna().mean(axis=0)
        keep = frac.index[frac <= max_missing_frac].tolist()
        return self.select(keep)

    # ------------------------------------------------------------ conversion
    def symbol_frame(self, symbol: str) -> pd.DataFrame:
        """Return one symbol's history as a frame with OHLCV columns."""
        if symbol not in self.close.columns:
            raise DataError(f"Symbol '{symbol}' not in dataset")
        return pd.DataFrame(
            {name: getattr(self, name)[symbol] for name in FIELDS},
            index=self.index,
        )

    @classmethod
    def from_symbol_frames(
        cls,
        frames: Mapping[str, pd.DataFrame],
        frequency: Frequency = Frequency.DAILY,
        metadata: dict[str, Any] | None = None,
    ) -> MarketDataset:
        """Build a dataset from per-symbol OHLCV frames.

        Frames are outer-joined on the union of their indices; symbols absent
        on a date carry NaN. Each frame must contain all OHLCV columns.
        """
        if not frames:
            raise DataError("from_symbol_frames received an empty mapping")
        for sym, df in frames.items():
            missing = [c for c in FIELDS if c not in df.columns]
            if missing:
                raise DataError(f"Symbol '{sym}' frame missing columns {missing}")
        union = pd.Index([])
        for df in frames.values():
            union = union.union(df.index)
        union = pd.DatetimeIndex(union).sort_values()
        wide = {
            name: pd.DataFrame(
                {sym: df[name].reindex(union) for sym, df in frames.items()},
                index=union,
            ).astype(float)
            for name in FIELDS
        }
        return cls(frequency=frequency, metadata=dict(metadata or {}), **wide)

    @classmethod
    def from_long(
        cls,
        long: pd.DataFrame,
        *,
        date_col: str = "date",
        symbol_col: str = "symbol",
        field_cols: Mapping[str, str] | None = None,
        frequency: Frequency = Frequency.DAILY,
        metadata: dict[str, Any] | None = None,
    ) -> MarketDataset:
        """Build a dataset from a long frame with one row per (date, symbol).

        Parameters
        ----------
        field_cols:
            Optional mapping from canonical field name to the column name in
            ``long`` (e.g. ``{"close": "adj_close"}``). Unmapped fields use
            their canonical names; entirely absent fields become NaN.
        """
        field_cols = dict(field_cols or {})
        for col in (date_col, symbol_col):
            if col not in long.columns:
                raise DataError(f"Long frame missing required column '{col}'")
        work = long.copy()
        work[date_col] = pd.to_datetime(work[date_col])
        if work.duplicated([date_col, symbol_col]).any():
            raise DataError("Long frame has duplicate (date, symbol) rows")

        wide: dict[str, WideFrame] = {}
        template: WideFrame | None = None
        for name in FIELDS:
            src = field_cols.get(name, name)
            if src in work.columns:
                frame = work.pivot(index=date_col, columns=symbol_col, values=src)
                frame = frame.sort_index()
                frame.columns = [str(c) for c in frame.columns]
                frame = frame[sorted(frame.columns)].astype(float)
                wide[name] = frame
                template = frame
        if template is None:
            raise DataError("Long frame contains none of the OHLCV fields")
        for name in FIELDS:
            if name not in wide:
                wide[name] = pd.DataFrame(
                    np.nan, index=template.index, columns=template.columns
                )
        wide = {
            name: frame.reindex(index=template.index, columns=template.columns)
            for name, frame in wide.items()
        }
        return cls(frequency=frequency, metadata=dict(metadata or {}), **wide)

    def to_long(self, dropna: bool = True) -> pd.DataFrame:
        """Flatten to a long frame with columns [date, symbol, open..volume].

        Parameters
        ----------
        dropna:
            Drop rows where every field is NaN (default). Set False to keep a
            fully dense (date x symbol) cartesian product.
        """
        pieces = []
        for name in FIELDS:
            s = getattr(self, name).stack()
            s.name = name
            pieces.append(s)
        long = pd.concat(pieces, axis=1)
        long.index.set_names(["date", "symbol"], inplace=True)
        if dropna:
            long = long.dropna(how="all")
        return long.reset_index()

    # -------------------------------------------------------------- plumbing
    def with_metadata(self, **updates: Any) -> MarketDataset:
        """Return a copy with metadata keys merged in."""
        merged = {**self.metadata, **updates}
        return replace(self, metadata=merged)

    def copy(self) -> MarketDataset:
        """Deep copy (frames copied; metadata shallow-copied)."""
        return replace(
            self,
            metadata=dict(self.metadata),
            **{name: getattr(self, name).copy() for name in FIELDS},
        )

    def equals(self, other: MarketDataset) -> bool:
        """Field-wise frame equality (NaN == NaN), ignoring metadata."""
        return all(getattr(self, n).equals(getattr(other, n)) for n in FIELDS)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        if self.n_periods == 0 or self.n_symbols == 0:
            span = "empty"
        else:
            span = f"{self.start.date()} .. {self.end.date()}"
        return (
            f"MarketDataset({self.n_symbols} symbols x {self.n_periods} bars, "
            f"{self.frequency.value}, {span})"
        )
