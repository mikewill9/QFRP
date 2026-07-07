"""Orders.

Only market orders are modeled: this is a daily-bar research framework, and
pretending to know limit-order queue position from OHLCV bars manufactures
false precision. Fill realism is instead handled by the execution layer
(slippage, market impact, participation caps).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from quantlab.exceptions import BacktestError

__all__ = ["Order", "OrderType"]

_order_ids = itertools.count(1)


class OrderType(StrEnum):
    """Supported order types."""

    MARKET = "market"


@dataclass(frozen=True, slots=True)
class Order:
    """An instruction to trade ``quantity`` of ``symbol``.

    Positive quantity buys, negative sells; quantities are floats so target
    weight strategies can express exact rebalances (fractional shares).
    """

    symbol: str
    quantity: float
    timestamp: pd.Timestamp
    order_type: OrderType = OrderType.MARKET
    order_id: int = field(default_factory=lambda: next(_order_ids))

    def __post_init__(self) -> None:
        if not self.symbol:
            raise BacktestError("Order symbol must be non-empty")
        if not pd.notna(self.quantity) or self.quantity == 0.0:
            raise BacktestError(
                f"Order quantity must be finite and non-zero, got {self.quantity!r}"
            )

    @property
    def side(self) -> str:
        """``"buy"`` or ``"sell"``."""
        return "buy" if self.quantity > 0 else "sell"

    def __repr__(self) -> str:
        return (
            f"Order(#{self.order_id} {self.side} {abs(self.quantity):g} "
            f"{self.symbol} @ market, {self.timestamp.date()})"
        )
