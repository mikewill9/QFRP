"""Dataset preprocessing.

Stateless, composable transformations applied between ingestion and research:
return computation, controlled forward-filling, frequency resampling,
multi-dataset alignment, and liquidity/coverage-based universe filtering.
All functions return new objects; nothing mutates inputs.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from quantlab.data.dataset import FIELDS, MarketDataset
from quantlab.exceptions import DataError
from quantlab.types import Frequency, WideFrame

__all__ = [
    "align_intersection",
    "compute_returns",
    "filter_universe",
    "forward_fill_prices",
    "resample_dataset",
    "rolling_dollar_volume",
]


def compute_returns(
    prices: WideFrame,
    method: str = "log",
    periods: int = 1,
) -> WideFrame:
    """Per-column returns over ``periods`` bars.

    Parameters
    ----------
    prices:
        Wide price frame.
    method:
        ``"log"`` -> ``ln(p_t / p_{t-k})``; ``"simple"`` -> ``p_t/p_{t-k} - 1``.
    periods:
        Horizon ``k`` in bars.
    """
    if periods < 1:
        raise DataError("periods must be >= 1")
    shifted = prices.shift(periods)
    if method == "log":
        return np.log(prices / shifted)
    if method == "simple":
        return prices / shifted - 1.0
    raise DataError(f"Unknown return method '{method}' (use 'log' or 'simple')")


def forward_fill_prices(dataset: MarketDataset, limit: int = 5) -> MarketDataset:
    """Forward-fill short holes in price fields; volume holes become zero.

    Filling is capped at ``limit`` consecutive bars so genuinely dead
    listings stay visibly missing rather than silently flat. Volume is set to
    0 on filled bars only (no trade happened), never forward-filled.
    """
    if limit < 0:
        raise DataError("limit must be >= 0")
    close_was_nan = dataset.close.isna()
    filled = {
        name: dataset.field(name).ffill(limit=limit)
        for name in ("open", "high", "low", "close")
    }
    now_filled = close_was_nan & filled["close"].notna()
    volume = dataset.volume.copy()
    volume[now_filled] = 0.0
    return replace(dataset, volume=volume, **filled)


def resample_dataset(dataset: MarketDataset, frequency: Frequency) -> MarketDataset:
    """Resample OHLCV to a lower frequency with correct per-field aggregation.

    open=first, high=max, low=min, close=last, volume=sum within each target
    period. Bars whose period contains no source observations are dropped.
    Only downsampling is supported (daily -> weekly/monthly, weekly ->
    monthly).
    """
    order = {Frequency.DAILY: 0, Frequency.WEEKLY: 1, Frequency.MONTHLY: 2}
    if order[frequency] <= order[dataset.frequency]:
        if frequency is dataset.frequency:
            return dataset.copy()
        raise DataError(
            f"Cannot upsample {dataset.frequency.value} data to {frequency.value}"
        )
    rule = frequency.pandas_freq
    agg = {
        "open": lambda df: df.resample(rule).first(),
        "high": lambda df: df.resample(rule).max(),
        "low": lambda df: df.resample(rule).min(),
        "close": lambda df: df.resample(rule).last(),
        "volume": lambda df: df.resample(rule).sum(min_count=1),
    }
    resampled = {name: fn(dataset.field(name)) for name, fn in agg.items()}
    keep = resampled["close"].notna().any(axis=1)
    resampled = {name: frame.loc[keep] for name, frame in resampled.items()}
    return replace(dataset, frequency=frequency, **resampled)


def align_intersection(*datasets: MarketDataset) -> tuple[MarketDataset, ...]:
    """Restrict every dataset to the shared dates and shared symbols.

    Useful when combining sources (e.g. prices from one vendor, volume from
    another) before research. Frequencies must match.
    """
    if not datasets:
        raise DataError("align_intersection requires at least one dataset")
    freqs = {ds.frequency for ds in datasets}
    if len(freqs) > 1:
        raise DataError(f"Cannot align mixed frequencies: {sorted(f.value for f in freqs)}")
    common_index = datasets[0].index
    common_symbols = pd.Index(datasets[0].symbols)
    for ds in datasets[1:]:
        common_index = common_index.intersection(ds.index)
        common_symbols = common_symbols.intersection(pd.Index(ds.symbols))
    if len(common_index) == 0 or len(common_symbols) == 0:
        raise DataError("Datasets share no common dates/symbols")
    symbols = list(common_symbols)
    out = []
    for ds in datasets:
        sliced = {
            name: ds.field(name).loc[common_index, symbols] for name in FIELDS
        }
        out.append(replace(ds, **sliced))
    return tuple(out)


def rolling_dollar_volume(dataset: MarketDataset, window: int = 21) -> WideFrame:
    """Trailing-mean dollar volume (close x volume) per symbol.

    ``min_periods`` is half the window so early history is usable without
    being dominated by a single print.
    """
    if window < 1:
        raise DataError("window must be >= 1")
    dv = dataset.dollar_volume()
    return dv.rolling(window, min_periods=max(1, window // 2)).mean()


def filter_universe(
    dataset: MarketDataset,
    *,
    max_missing_frac: float = 0.10,
    min_median_dollar_volume: float | None = None,
    min_price: float | None = None,
) -> MarketDataset:
    """Drop symbols failing coverage, liquidity, or price floors.

    Parameters
    ----------
    max_missing_frac:
        Maximum fraction of missing close observations.
    min_median_dollar_volume:
        Minimum median daily dollar volume over the full window (skipped when
        volume is entirely NaN for a symbol).
    min_price:
        Minimum median close price (filters sub-penny noise names).

    Raises
    ------
    DataError
        If no symbol survives the filters.
    """
    keep = pd.Series(True, index=pd.Index(dataset.symbols))
    keep &= dataset.close.isna().mean(axis=0) <= max_missing_frac
    if min_price is not None:
        keep &= dataset.close.median(axis=0) >= min_price
    if min_median_dollar_volume is not None:
        med_dv = dataset.dollar_volume().median(axis=0)
        keep &= med_dv.isna() | (med_dv >= min_median_dollar_volume)
    survivors = keep.index[keep].tolist()
    if not survivors:
        raise DataError("filter_universe removed every symbol; loosen thresholds")
    return dataset.select(survivors)
