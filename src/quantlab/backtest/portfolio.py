"""Portfolio accounting.

Cash and positions change *only* through :meth:`Portfolio.apply_fill`, which
enforces the accounting identity at every step:

    equity = cash + sum(position_qty * last_mark_price)

Positions may be fractional and negative (shorts). Short proceeds are
credited to cash and the liability is marked through the position term, the
standard frictionless margin treatment for research backtests (no borrow
fees or margin interest; add them via cost models if needed).
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from quantlab.backtest.events import FillEvent
from quantlab.exceptions import BacktestError

__all__ = ["Ledger", "Portfolio"]


class Ledger:
    """Append-only record of every fill, exportable as a DataFrame."""

    def __init__(self) -> None:
        self._fills: list[FillEvent] = []

    def record(self, fill: FillEvent) -> None:
        self._fills.append(fill)

    @property
    def fills(self) -> tuple[FillEvent, ...]:
        return tuple(self._fills)

    def __len__(self) -> int:
        return len(self._fills)

    def to_frame(self) -> pd.DataFrame:
        """One row per fill: timestamp, symbol, quantity, price, costs."""
        if not self._fills:
            return pd.DataFrame(
                columns=[
                    "timestamp", "symbol", "quantity", "price",
                    "notional", "commission", "slippage_cost", "order_id",
                ]
            )
        return pd.DataFrame(
            {
                "timestamp": [f.timestamp for f in self._fills],
                "symbol": [f.symbol for f in self._fills],
                "quantity": [f.quantity for f in self._fills],
                "price": [f.price for f in self._fills],
                "notional": [f.notional for f in self._fills],
                "commission": [f.commission for f in self._fills],
                "slippage_cost": [f.slippage_cost for f in self._fills],
                "order_id": [f.order_id for f in self._fills],
            }
        )


class Portfolio:
    """Positions, cash, and mark-to-market equity."""

    def __init__(self, initial_capital: float) -> None:
        if initial_capital <= 0:
            raise BacktestError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)
        self.cash: float = float(initial_capital)
        self._positions: dict[str, float] = {}
        self._marks: dict[str, float] = {}
        self.ledger = Ledger()

    # ------------------------------------------------------------- mutation
    def apply_fill(self, fill: FillEvent) -> None:
        """Settle a fill: move cash, adjust the position, record it."""
        self.cash -= fill.quantity * fill.price + fill.commission
        new_qty = self._positions.get(fill.symbol, 0.0) + fill.quantity
        if abs(new_qty) < 1e-12:
            self._positions.pop(fill.symbol, None)
        else:
            self._positions[fill.symbol] = new_qty
        # A fill is also the freshest mark we have for the symbol.
        self._marks[fill.symbol] = fill.price
        self.ledger.record(fill)

    def mark_to_market(self, prices: pd.Series) -> float:
        """Update marks from ``prices`` (NaN entries keep the old mark).

        Returns the resulting equity.
        """
        for symbol in self._positions:
            px = prices.get(symbol)
            if px is not None and pd.notna(px):
                self._marks[symbol] = float(px)
        return self.equity

    # ------------------------------------------------------------ inspection
    @property
    def positions(self) -> Mapping[str, float]:
        """Read-only view of current position quantities."""
        return dict(self._positions)

    def position(self, symbol: str) -> float:
        return self._positions.get(symbol, 0.0)

    @property
    def equity(self) -> float:
        """cash + marked value of all open positions."""
        value = sum(
            qty * self._marks[sym] for sym, qty in self._positions.items()
            if sym in self._marks
        )
        unmarked = [s for s in self._positions if s not in self._marks]
        if unmarked:
            raise BacktestError(f"Positions without any mark price: {unmarked}")
        return self.cash + value

    def weights(self) -> pd.Series:
        """Position weights as a fraction of current equity."""
        eq = self.equity
        if eq <= 0:
            raise BacktestError(f"Equity is non-positive ({eq:.2f}); cannot form weights")
        return pd.Series(
            {s: q * self._marks[s] / eq for s, q in self._positions.items()},
            dtype=float,
        )

    def gross_exposure(self) -> float:
        """Sum of absolute position values over equity."""
        w = self.weights()
        return float(w.abs().sum()) if len(w) else 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Portfolio(equity={self.equity:,.2f}, cash={self.cash:,.2f}, "
            f"{len(self._positions)} open positions, {len(self.ledger)} fills)"
        )
