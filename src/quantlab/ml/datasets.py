"""Supervised panel construction.

The research layer produces *wide* factor frames (dates x symbols). ML
models want a *long* design matrix: one row per (date, symbol) with one
column per factor and a target. :class:`PanelDataset.from_factors` does
that reshaping with the framework's no-lookahead alignment baked in:

- The feature row at date ``t`` holds factor values computed at ``t``
  (factors may only use data through the close of ``t`` — that is the
  Factor contract).
- The target at date ``t`` is the **forward** return over ``(t, t+h]``
  via :func:`quantlab.research.evaluation.forward_returns`, so the
  label window starts strictly after the features' information window
  ends. An optional extra ``feature_lag`` shifts features back further
  for sensitivity checks.

Rows with any missing feature or a missing target are dropped (the last
``horizon`` dates have no target by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.exceptions import DataError
from quantlab.research.evaluation import forward_returns
from quantlab.types import WideFrame

__all__ = ["PanelDataset"]


@dataclass(frozen=True, slots=True)
class PanelDataset:
    """Long supervised panel with group labels for date-aware CV.

    Attributes
    ----------
    X:
        Feature matrix, MultiIndex ``(date, symbol)`` rows sorted by
        date, one column per factor.
    y:
        Target aligned to ``X``: forward return over ``(t, t+horizon]``.
    groups:
        The date of each row (used by the purged/walk-forward splitters;
        all rows of one date must stay on one side of a split).
    horizon:
        Label horizon in bars; splitters need it to purge overlaps.
    """

    X: pd.DataFrame
    y: pd.Series
    groups: pd.Series
    horizon: int

    def __post_init__(self) -> None:
        if len(self.X) != len(self.y) or len(self.X) != len(self.groups):
            raise DataError("X, y, and groups must have equal length")
        if len(self.X) == 0:
            raise DataError("Panel is empty")

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Unique row dates, sorted."""
        return pd.DatetimeIndex(self.groups.unique()).sort_values()

    def __len__(self) -> int:
        return len(self.X)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"PanelDataset({len(self)} rows, {len(self.feature_names)} features, "
            f"{len(self.dates)} dates, horizon={self.horizon})"
        )

    @classmethod
    def from_factors(
        cls,
        factors: dict[str, WideFrame],
        prices: WideFrame,
        horizon: int = 21,
        *,
        return_method: str = "log",
        feature_lag: int = 0,
        target_transform: str = "raw",
    ) -> PanelDataset:
        """Build the panel from wide factor frames and a price frame.

        Factors are aligned on the intersection of their dates and
        symbols with ``prices``. ``feature_lag`` shifts every factor
        back by that many extra bars (0 keeps the standard signal-at-t /
        return-over-``(t, t+h]`` alignment, which is already causal).

        ``target_transform`` reshapes the label cross-sectionally per
        date, before stacking:

        - ``"raw"``: the forward return unchanged. Default only for
          backward compatibility.
        - ``"demeaned"``: subtract each date's cross-sectional mean, so
          the label is *relative* return within the universe. This is
          the recommended setting for cross-sectional research — the
          model should never fit the market level.
        - ``"rank"``: per-date centered percentile ranks in
          ``(-0.5, 0.5)`` (the ``research.signals.Rank(center=True)``
          construction).
        """
        if not factors:
            raise DataError("factors dict is empty")
        if horizon < 1:
            raise DataError(f"horizon must be >= 1, got {horizon}")
        if feature_lag < 0:
            raise DataError(f"feature_lag must be >= 0, got {feature_lag}")
        if target_transform not in ("raw", "demeaned", "rank"):
            raise DataError(
                f"Unknown target_transform '{target_transform}' "
                "(use 'raw', 'demeaned', or 'rank')"
            )

        index = prices.index
        columns = prices.columns
        for frame in factors.values():
            index = index.intersection(frame.index)
            columns = columns.intersection(frame.columns)
        if len(index) <= horizon or len(columns) == 0:
            raise DataError(
                "Factors and prices share too little overlap "
                f"({len(index)} dates x {len(columns)} symbols, horizon {horizon})"
            )
        index = index.sort_values()

        target = forward_returns(prices, horizon, method=return_method)
        if target_transform == "demeaned":
            target = target.sub(target.mean(axis=1), axis=0)
        elif target_transform == "rank":
            ranked = target.rank(axis=1)
            n = target.notna().sum(axis=1)
            target = ranked.sub((n + 1) / 2.0, axis=0).div(n, axis=0)
        long_parts = {"__target__": target.reindex(index=index, columns=columns)}
        for name, frame in factors.items():
            aligned = frame.reindex(index=index, columns=columns)
            if feature_lag:
                aligned = aligned.shift(feature_lag)
            long_parts[str(name)] = aligned

        # Wide -> long: stack each frame into a (date, symbol) MultiIndex.
        stacked = pd.DataFrame(
            {name: frame.stack(future_stack=True) for name, frame in long_parts.items()}
        )
        stacked.index.names = ["date", "symbol"]
        stacked = stacked.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        if stacked.empty:
            raise DataError("No complete rows remain after alignment and NaN drop")
        stacked = stacked.sort_index(level=["date", "symbol"])

        y = stacked.pop("__target__").rename("forward_return")
        groups = stacked.index.get_level_values("date").to_series(index=stacked.index)
        groups = groups.rename("date")
        return cls(X=stacked, y=y, groups=groups, horizon=horizon)
