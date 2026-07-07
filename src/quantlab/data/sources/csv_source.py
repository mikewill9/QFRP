"""CSV-backed data source.

Two layouts are supported, selected by whether ``path`` is a directory or a
file:

Directory layout (one file per symbol)
    ``{path}/{symbol}.csv`` (pattern configurable), each file containing a
    date column plus OHLCV columns.

Long layout (single file)
    One CSV with ``date`` and ``symbol`` columns plus OHLCV columns; rows are
    (date, symbol) observations.

Column names are remapped via ``column_map`` (canonical name -> file column),
so vendor exports like ``Adj Close`` slot in without preprocessing. Missing
optional fields (commonly ``volume``) are filled with NaN.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd

from quantlab.data.dataset import FIELDS, MarketDataset
from quantlab.data.sources.base import DataSource, validate_request
from quantlab.exceptions import DataError
from quantlab.types import Frequency

__all__ = ["CSVSource"]


class CSVSource(DataSource):
    """Read OHLCV data from local CSV files.

    Parameters
    ----------
    path:
        Directory of per-symbol files, or a single long-format file.
    filename_pattern:
        Used in directory mode; must contain ``{symbol}``.
    column_map:
        Canonical -> file column names, e.g. ``{"close": "Adj Close",
        "date": "Date"}``. Canonical names: ``date``, ``symbol`` (long mode
        only), and the OHLCV fields.
    strict:
        If True (default), missing symbols/files raise; otherwise they are
        skipped with the remainder returned.
    frequency:
        Declared frequency of the files (no resampling is performed here).
    """

    name = "csv"

    def __init__(
        self,
        path: Path | str,
        *,
        filename_pattern: str = "{symbol}.csv",
        column_map: Mapping[str, str] | None = None,
        strict: bool = True,
        frequency: Frequency = Frequency.DAILY,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise DataError(f"CSV path does not exist: {self.path}")
        if "{symbol}" not in filename_pattern:
            raise DataError("filename_pattern must contain '{symbol}'")
        self.filename_pattern = filename_pattern
        self.column_map = dict(column_map or {})
        self.strict = bool(strict)
        self.frequency = frequency

    # ------------------------------------------------------------------ api
    def fetch(
        self,
        symbols: Sequence[str],
        start: pd.Timestamp | str,
        end: pd.Timestamp | str,
        frequency: Frequency = Frequency.DAILY,
    ) -> MarketDataset:
        symbols_list, start_ts, end_ts = validate_request(symbols, start, end)
        if frequency is not self.frequency:
            raise DataError(
                f"CSVSource declared {self.frequency.value} data; "
                f"requested {frequency.value}. Resample downstream instead."
            )
        if self.path.is_dir():
            ds = self._fetch_per_symbol(symbols_list)
        else:
            ds = self._fetch_long(symbols_list)
        ds = ds.between(start_ts, end_ts)
        if ds.n_periods == 0:
            raise DataError(
                f"No CSV rows in [{start_ts.date()}, {end_ts.date()}] for {symbols_list[:5]}"
            )
        return ds.with_metadata(source=self.name, path=str(self.path))

    # ------------------------------------------------------------- internals
    def _col(self, canonical: str) -> str:
        return self.column_map.get(canonical, canonical)

    def _read_csv(self, path: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except (OSError, pd.errors.ParserError) as exc:
            raise DataError(f"Failed to read CSV {path}: {exc}") from exc

    def _extract_fields(self, raw: pd.DataFrame, context: str) -> pd.DataFrame:
        """Return frame with a DatetimeIndex and canonical OHLCV columns."""
        date_col = self._col("date")
        if date_col not in raw.columns:
            raise DataError(f"{context}: missing date column '{date_col}'")
        out = pd.DataFrame(index=pd.to_datetime(raw[date_col]))
        found = 0
        for name in FIELDS:
            src = self._col(name)
            if src in raw.columns:
                out[name] = pd.to_numeric(raw[src], errors="coerce").to_numpy()
                found += 1
            else:
                out[name] = float("nan")
        if found == 0:
            raise DataError(f"{context}: no OHLCV columns found (check column_map)")
        if "close" not in [f for f in FIELDS if self._col(f) in raw.columns]:
            raise DataError(f"{context}: a close column is required")
        out = out.sort_index()
        out = out[~out.index.duplicated(keep="last")]
        return out

    def _fetch_per_symbol(self, symbols: list[str]) -> MarketDataset:
        frames: dict[str, pd.DataFrame] = {}
        missing: list[str] = []
        for sym in symbols:
            fp = self.path / self.filename_pattern.format(symbol=sym)
            if not fp.is_file():
                missing.append(sym)
                continue
            frames[sym] = self._extract_fields(self._read_csv(fp), context=str(fp))
        if missing and self.strict:
            raise DataError(f"Missing CSV files for symbols: {missing[:10]}")
        if not frames:
            raise DataError("No CSV files found for any requested symbol")
        return MarketDataset.from_symbol_frames(frames, frequency=self.frequency)

    def _fetch_long(self, symbols: list[str]) -> MarketDataset:
        raw = self._read_csv(self.path)
        sym_col = self._col("symbol")
        if sym_col not in raw.columns:
            raise DataError(f"{self.path}: long CSV missing symbol column '{sym_col}'")
        raw = raw[raw[sym_col].astype(str).isin(symbols)]
        present = set(raw[sym_col].astype(str))
        missing = [s for s in symbols if s not in present]
        if missing and self.strict:
            raise DataError(f"Symbols absent from {self.path}: {missing[:10]}")
        if raw.empty:
            raise DataError(f"No rows for requested symbols in {self.path}")
        frames = {
            str(sym): self._extract_fields(grp, context=f"{self.path}[{sym}]")
            for sym, grp in raw.groupby(sym_col)
        }
        return MarketDataset.from_symbol_frames(frames, frequency=self.frequency)
