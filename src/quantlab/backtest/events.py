"""Event types for the backtest engine.

The engine advances bar by bar; each bar emits a :class:`MarketEvent`,
strategies respond with orders (wrapped as :class:`OrderEvent`), and the
execution simulator turns orders into :class:`FillEvent` records. Fills are
the single source of truth for portfolio accounting: cash and positions
change only through ``Portfolio.apply_fill``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.backtest.orders import Order

__all__ = ["FillEvent", "MarketEvent", "OrderEvent"]


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """A new bar became available at ``timestamp``."""

    timestamp: pd.Timestamp
    bar_index: int


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """A strategy submitted ``order`` in response to the bar at ``timestamp``."""

    timestamp: pd.Timestamp
    order: Order


@dataclass(frozen=True, slots=True)
class FillEvent:
    """An (possibly partial) execution of an order.

    Attributes
    ----------
    quantity:
        Signed filled quantity; positive = bought.
    price:
        Executed price *including* slippage.
    commission:
        Dollar commission charged on this fill.
    slippage_cost:
        Dollar cost of slippage versus the unperturbed reference price
        (always >= 0); informational, already embedded in ``price``.
    """

    timestamp: pd.Timestamp
    symbol: str
    quantity: float
    price: float
    commission: float
    slippage_cost: float
    order_id: int

    @property
    def notional(self) -> float:
        """Signed executed notional (quantity x price)."""
        return self.quantity * self.price
