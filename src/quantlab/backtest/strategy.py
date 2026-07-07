"""Strategies.

A strategy receives a :class:`BarContext` at each bar *after* the portfolio
has been marked to the bar's close and returns orders (or ``None``). With
the default ``NEXT_OPEN`` execution timing those orders fill at the next
bar's open, so the information flow is honest by construction: decide on
today's close, trade tomorrow.

:class:`TargetWeightStrategy` is the workhorse that connects the research
layer to the engine, and :func:`signal_to_weights` converts a score frame
into a long/short target-weight frame on a rebalance schedule.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantlab.backtest.orders import Order
from quantlab.backtest.portfolio import Portfolio
from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import BacktestError
from quantlab.types import WideFrame

__all__ = ["BarContext", "Strategy", "TargetWeightStrategy", "signal_to_weights"]


@dataclass(frozen=True, slots=True)
class BarContext:
    """Everything a strategy may look at when the bar at ``timestamp`` closes.

    ``dataset`` is the full dataset for efficiency; well-behaved strategies
    must only read rows up to ``bar_index`` (use :meth:`history`). The
    bundled strategies honor this contract.
    """

    timestamp: pd.Timestamp
    bar_index: int
    open_prices: pd.Series
    close_prices: pd.Series
    portfolio: Portfolio
    dataset: MarketDataset

    def history(self, field: str = "close", window: int | None = None) -> WideFrame:
        """Rows of ``field`` up to and including the current bar."""
        frame = self.dataset.field(field).iloc[: self.bar_index + 1]
        return frame if window is None else frame.iloc[-window:]


class Strategy(ABC):
    """Base class for strategies driven by the engine's event loop."""

    #: Identifier recorded in results metadata.
    name: str = "strategy"

    def on_start(self, dataset: MarketDataset) -> None:  # noqa: B027 - optional hook
        """Hook called once before the first bar (precompute state here)."""

    @abstractmethod
    def on_bar(self, context: BarContext) -> list[Order] | None:
        """React to a completed bar with orders for the next fill point."""


class TargetWeightStrategy(Strategy):
    """Trade toward target weights supplied as a frame or a callable.

    Parameters
    ----------
    weights:
        Wide frame of target weights. Rows that are entirely NaN are
        no-rebalance bars, so the frame's non-NaN rows *are* the schedule.
        Within a rebalance row, NaN means "no position" (target 0).
    weight_fn:
        Alternative to ``weights``: called with the :class:`BarContext`
        every ``rebalance_every`` bars; returns a weight Series or ``None``
        to skip. Exactly one of ``weights`` / ``weight_fn`` must be given.
    rebalance_every:
        Cadence in bars for ``weight_fn`` (ignored for ``weights``).
    tolerance:
        Per-symbol no-trade band: skip trades whose notional is below
        ``tolerance * equity``. Cuts cost churn from tiny weight drift.
    """

    name = "target_weight"

    def __init__(
        self,
        weights: WideFrame | None = None,
        weight_fn: Callable[[BarContext], pd.Series | None] | None = None,
        rebalance_every: int = 21,
        tolerance: float = 0.0005,
    ) -> None:
        if (weights is None) == (weight_fn is None):
            raise BacktestError("Provide exactly one of weights / weight_fn")
        if rebalance_every < 1:
            raise BacktestError("rebalance_every must be >= 1")
        if tolerance < 0:
            raise BacktestError("tolerance must be >= 0")
        self.weights = weights
        self.weight_fn = weight_fn
        self.rebalance_every = rebalance_every
        self.tolerance = tolerance

    def on_start(self, dataset: MarketDataset) -> None:  # noqa: B027 - optional hook
        if self.weights is not None:
            unknown = [c for c in self.weights.columns if c not in dataset.symbols]
            if unknown:
                raise BacktestError(
                    f"Target weights reference symbols outside the dataset: {unknown[:5]}"
                )

    def _target_for(self, context: BarContext) -> pd.Series | None:
        if self.weights is not None:
            if context.timestamp not in self.weights.index:
                return None
            row = self.weights.loc[context.timestamp]
            if row.isna().all():
                return None
            return row.fillna(0.0)
        if context.bar_index % self.rebalance_every != 0:
            return None
        target = self.weight_fn(context)  # type: ignore[misc]
        return None if target is None else target.fillna(0.0)

    def on_bar(self, context: BarContext) -> list[Order] | None:
        target = self._target_for(context)
        if target is None:
            return None
        equity = context.portfolio.equity
        prices = context.close_prices
        orders: list[Order] = []
        symbols = set(target.index) | set(context.portfolio.positions)
        for symbol in sorted(symbols):
            w = float(target.get(symbol, 0.0))
            price = prices.get(symbol)
            if price is None or not np.isfinite(price) or price <= 0:
                if w != 0.0:
                    # Cannot size a position without a price; keep flat.
                    continue
                continue
            target_qty = w * equity / float(price)
            delta = target_qty - context.portfolio.position(symbol)
            if abs(delta) * float(price) <= self.tolerance * equity:
                continue
            if delta != 0.0:
                orders.append(
                    Order(symbol=symbol, quantity=delta, timestamp=context.timestamp)
                )
        return orders or None


def signal_to_weights(
    signal: WideFrame,
    *,
    long_frac: float = 0.2,
    short_frac: float = 0.2,
    gross: float = 1.0,
    rebalance_every: int = 21,
) -> WideFrame:
    """Convert a score frame into equal-weight long/short target weights.

    Every ``rebalance_every`` bars (starting from the first bar with at
    least 3 valid scores), the top ``long_frac`` of names by score receive
    equal positive weights and the bottom ``short_frac`` equal negative
    weights, scaled so gross exposure is ``gross`` (dollar neutral when both
    fractions are positive). All other rows are NaN, which
    :class:`TargetWeightStrategy` interprets as "hold".

    Setting ``short_frac=0`` produces a long-only portfolio with gross
    (= net) exposure ``gross``.
    """
    if not 0 <= long_frac <= 1 or not 0 <= short_frac <= 1:
        raise BacktestError("long_frac/short_frac must be in [0, 1]")
    if long_frac == 0 and short_frac == 0:
        raise BacktestError("At least one of long_frac/short_frac must be > 0")
    if gross <= 0:
        raise BacktestError("gross must be positive")
    if rebalance_every < 1:
        raise BacktestError("rebalance_every must be >= 1")

    values = np.full(signal.shape, np.nan)
    sides = int(long_frac > 0) + int(short_frac > 0)
    arr = signal.to_numpy(dtype=float)
    first_valid: int | None = None
    for i in range(signal.shape[0]):
        row = arr[i]
        valid = np.isfinite(row)
        if valid.sum() < 3:
            continue
        if first_valid is None:
            first_valid = i
        if (i - first_valid) % rebalance_every != 0:
            continue
        idx_valid = np.flatnonzero(valid)
        order = idx_valid[np.argsort(row[idx_valid])]
        n = len(order)
        n_long = max(1, int(round(long_frac * n))) if long_frac > 0 else 0
        n_short = max(1, int(round(short_frac * n))) if short_frac > 0 else 0
        w_row = np.zeros(signal.shape[1])
        per_side = gross / sides
        if n_long:
            w_row[order[-n_long:]] = per_side / n_long
        if n_short:
            w_row[order[:n_short]] = -per_side / n_short
        values[i] = w_row
    return pd.DataFrame(values, index=signal.index, columns=signal.columns)
