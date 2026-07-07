"""Core shared types and data-shape conventions.

Data-shape conventions used across the entire framework
-------------------------------------------------------
Wide frame
    ``pd.DataFrame`` with a tz-naive, sorted, unique ``DatetimeIndex`` (rows =
    observation dates) and string symbol columns. Prices, returns, factor
    scores, signals, and portfolio weights are all wide frames. This is the
    native shape for vectorized cross-sectional research.

Long frame
    ``pd.DataFrame`` with columns ``["date", "symbol", ...]``. Used only at
    IO boundaries (CSV ingestion, ML panel datasets), never inside compute
    kernels.

Returns
    ``method="log"`` means ``ln(p_t / p_{t-k})``; ``method="simple"`` means
    ``p_t / p_{t-k} - 1``. Log returns are the default for research because
    they aggregate additively over time.

Timestamps
    All indices are tz-naive. Sources that produce tz-aware data must strip
    the timezone (converting to UTC first) before constructing datasets.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TypeAlias

import pandas as pd

__all__ = [
    "AssetClass",
    "Frequency",
    "LongFrame",
    "ReturnMethod",
    "WideFrame",
]

# Documentation aliases. They carry intent in signatures; both are plain
# DataFrames at runtime.
WideFrame: TypeAlias = pd.DataFrame
LongFrame: TypeAlias = pd.DataFrame

ReturnMethod: TypeAlias = str  # "log" | "simple"


class Frequency(StrEnum):
    """Sampling frequency of a dataset."""

    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"

    @property
    def pandas_freq(self) -> str:
        """Pandas offset alias for this frequency (business calendar)."""
        return {
            Frequency.DAILY: "B",
            Frequency.WEEKLY: "W-FRI",
            Frequency.MONTHLY: "BME",
        }[self]

    @property
    def periods_per_year(self) -> int:
        """Annualization factor for this frequency."""
        return {
            Frequency.DAILY: 252,
            Frequency.WEEKLY: 52,
            Frequency.MONTHLY: 12,
        }[self]


class AssetClass(StrEnum):
    """Coarse asset-class tag carried in dataset/universe metadata."""

    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    FX = "fx"
    CRYPTO = "crypto"
    RATE = "rate"
