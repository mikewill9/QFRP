"""Local parquet storage.

Two layers:

- Low-level atomic helpers (:func:`write_symbol_frame`,
  :func:`read_symbol_frame`) used by the cache. Writes go to a temp file in
  the same directory followed by ``os.replace``, so a crash never leaves a
  half-written parquet behind.
- :class:`DatasetStore`, a named store for whole
  :class:`~quantlab.data.dataset.MarketDataset` objects (five parquet files
  plus a JSON manifest), used to persist research inputs and experiment
  artifacts.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.dataset import FIELDS, MarketDataset
from quantlab.exceptions import CacheError, DataError
from quantlab.types import Frequency

__all__ = [
    "DatasetStore",
    "read_symbol_frame",
    "write_symbol_frame",
]

_MANIFEST = "manifest.json"


def write_symbol_frame(path: Path, frame: pd.DataFrame) -> None:
    """Atomically write one symbol's OHLCV frame to parquet.

    The frame must be indexed by date with the canonical OHLCV columns.
    """
    missing = [c for c in FIELDS if c not in frame.columns]
    if missing:
        raise DataError(f"Symbol frame missing columns {missing}")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame[list(FIELDS)].copy()
    out.index = pd.DatetimeIndex(out.index, name="date")
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".parquet.tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        out.to_parquet(tmp)
        os.replace(tmp, path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise CacheError(f"Failed writing parquet {path}: {exc}") from exc


def read_symbol_frame(path: Path) -> pd.DataFrame:
    """Read one symbol's OHLCV frame; raises CacheError on corruption."""
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise CacheError(f"Failed reading parquet {path}: {exc}") from exc
    missing = [c for c in FIELDS if c not in frame.columns]
    if missing:
        raise CacheError(f"Cached frame {path} missing columns {missing}")
    frame.index = pd.DatetimeIndex(frame.index)
    return frame.sort_index()


class DatasetStore:
    """Named persistence for whole datasets under a root directory.

    Layout: ``{root}/{name}/{field}.parquet`` plus ``manifest.json`` carrying
    frequency and metadata. Names may contain ``/`` to create hierarchies
    (e.g. ``"universes/us_liquid"``).
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def _dir(self, name: str) -> Path:
        d = (self.root / name).resolve()
        if self.root.resolve() not in d.parents and d != self.root.resolve():
            raise DataError(f"Dataset name escapes store root: {name!r}")
        return d

    def save(self, dataset: MarketDataset, name: str) -> Path:
        """Persist ``dataset`` under ``name``; returns the directory written."""
        d = self._dir(name)
        d.mkdir(parents=True, exist_ok=True)
        for field_name in FIELDS:
            fd, tmp_name = tempfile.mkstemp(dir=d, suffix=".parquet.tmp")
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                dataset.field(field_name).to_parquet(tmp)
                os.replace(tmp, d / f"{field_name}.parquet")
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                raise CacheError(f"Failed writing {field_name} for '{name}': {exc}") from exc
        manifest: dict[str, Any] = {
            "frequency": dataset.frequency.value,
            "metadata": _jsonable(dataset.metadata),
            "symbols": dataset.symbols,
            "start": str(dataset.start.date()) if dataset.n_periods else None,
            "end": str(dataset.end.date()) if dataset.n_periods else None,
        }
        (d / _MANIFEST).write_text(json.dumps(manifest, indent=2))
        return d

    def load(self, name: str) -> MarketDataset:
        """Load a dataset previously saved under ``name``."""
        d = self._dir(name)
        manifest_path = d / _MANIFEST
        if not manifest_path.is_file():
            raise CacheError(f"No dataset named '{name}' under {self.root}")
        manifest = json.loads(manifest_path.read_text())
        frames = {}
        for field_name in FIELDS:
            fp = d / f"{field_name}.parquet"
            if not fp.is_file():
                raise CacheError(f"Dataset '{name}' missing {field_name}.parquet")
            frame = pd.read_parquet(fp)
            frame.index = pd.DatetimeIndex(frame.index)
            frames[field_name] = frame
        return MarketDataset(
            frequency=Frequency(manifest["frequency"]),
            metadata=dict(manifest.get("metadata", {})),
            **frames,
        )

    def exists(self, name: str) -> bool:
        """True if a dataset named ``name`` is present."""
        return (self._dir(name) / _MANIFEST).is_file()

    def list(self) -> list[str]:
        """All dataset names in the store (relative paths)."""
        if not self.root.exists():
            return []
        names = [
            str(p.parent.relative_to(self.root))
            for p in self.root.rglob(_MANIFEST)
        ]
        return sorted(names)


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion of metadata to JSON-serializable values."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)
