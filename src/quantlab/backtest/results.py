"""Backtest results.

:class:`BacktestResult` is a frozen container of everything the engine
recorded, with convenience accessors and a :meth:`summary` that delegates
to :mod:`quantlab.analytics.performance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quantlab.analytics.performance import performance_summary
from quantlab.types import WideFrame

__all__ = ["BacktestResult"]


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Immutable output of one engine run.

    Attributes
    ----------
    equity_curve:
        Marked-to-market equity at each bar close.
    returns:
        Simple per-bar returns of the equity curve (first bar is the return
        from initial capital to the first close).
    weights:
        Post-close position weights per bar (cash is ``1 - net``).
    positions:
        Position quantities per bar.
    trades:
        One row per fill (timestamp, symbol, quantity, price, costs).
    costs_paid:
        Total commission + slippage dollars charged per bar.
    metadata:
        Engine configuration echo (fill timing, strategy name, counts).
    """

    equity_curve: pd.Series
    returns: pd.Series
    weights: WideFrame
    positions: WideFrame
    trades: pd.DataFrame
    costs_paid: pd.Series
    initial_capital: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1])

    @property
    def total_costs(self) -> float:
        """All commission and slippage dollars paid over the run."""
        return float(self.costs_paid.sum())

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    def summary(self, rf: float = 0.0) -> pd.Series:
        """Headline performance table (see ``analytics.performance``)."""
        return performance_summary(self.returns, weights=self.weights, rf=rf)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        span = f"{self.equity_curve.index[0].date()}..{self.equity_curve.index[-1].date()}"
        growth = self.final_equity / self.initial_capital - 1.0
        return (
            f"BacktestResult({span}, {len(self.equity_curve)} bars, "
            f"{self.n_trades} fills, total return {growth:+.2%})"
        )
