"""Drawdown analysis.

A drawdown is the loss from the portfolio's running peak. Where
``analytics.performance.max_drawdown`` gives the single headline number,
this module exposes the full anatomy: the per-bar drawdown series, a
table of every episode (when it started, bottomed, and recovered), and
the ulcer index, which penalizes drawdowns by both depth and duration.

All functions take simple per-period returns and compound them
internally; drawdown values are negative fractions (0 = at the peak).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.analytics.performance import max_drawdown
from quantlab.exceptions import DataError

__all__ = ["drawdown_episodes", "drawdown_series", "max_drawdown", "ulcer_index"]

_EPISODE_COLUMNS = ["start", "trough", "recovery", "depth", "length", "recovered"]


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Per-bar drawdown from the running peak (<= 0)."""
    r = returns.dropna()
    if r.empty:
        raise DataError("Return series is empty after dropping NaN")
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    return (wealth / peak - 1.0).rename("drawdown")


def drawdown_episodes(returns: pd.Series) -> pd.DataFrame:
    """One row per drawdown episode, chronological.

    Columns: ``start`` (first bar below the prior peak), ``trough`` (bar
    of maximum loss), ``recovery`` (first bar back at the peak; ``NaT``
    while still underwater), ``depth`` (trough drawdown, negative),
    ``length`` (bars spent below the peak), ``recovered`` (bool).
    """
    dd = drawdown_series(returns)
    under = dd.to_numpy() < 0
    rows = []
    i, n = 0, len(dd)
    while i < n:
        if not under[i]:
            i += 1
            continue
        j = i
        while j < n and under[j]:
            j += 1
        episode = dd.iloc[i:j]
        recovered = j < n
        rows.append(
            {
                "start": dd.index[i],
                "trough": episode.idxmin(),
                "recovery": dd.index[j] if recovered else pd.NaT,
                "depth": float(episode.min()),
                "length": j - i,
                "recovered": recovered,
            }
        )
        i = j
    if not rows:
        return pd.DataFrame(columns=_EPISODE_COLUMNS)
    return pd.DataFrame(rows, columns=_EPISODE_COLUMNS)


def ulcer_index(returns: pd.Series) -> float:
    """Root-mean-square drawdown (a fraction, not percent).

    Unlike max drawdown it grows with how *long* the portfolio stays
    underwater, not just how deep the worst point was, so it separates a
    brief crash-and-recover from a grinding multi-year slump of the same
    depth.
    """
    dd = drawdown_series(returns)
    return float(np.sqrt((dd**2).mean()))
