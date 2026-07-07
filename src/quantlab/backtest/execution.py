"""Execution simulation.

Turns orders into fills against bar data. Realism levers:

- **Fill timing**: ``NEXT_OPEN`` (default) fills orders generated from the
  close of ``t`` at the open of ``t+1``, which is the honest daily-bar
  assumption. ``SAME_CLOSE`` fills at the very close the signal was computed
  on; it is explicitly optimistic and exists for quick research comparisons.
- **Slippage**: adverse price adjustment from a :class:`SlippageModel`.
- **Commission**: explicit fees from a :class:`CommissionModel`.
- **Participation cap**: at most ``max_participation`` of the bar's volume
  fills; the remainder of the order is cancelled (daily bars have no
  order book to rest in), so oversized rebalances degrade gracefully into
  partial fills rather than fantasy liquidity.
"""

from __future__ import annotations

import math
from enum import StrEnum

import pandas as pd

from quantlab.backtest.costs import (
    CommissionModel,
    SlippageModel,
    VolumeImpactSlippage,
    ZeroCommission,
)
from quantlab.backtest.events import FillEvent
from quantlab.backtest.orders import Order
from quantlab.exceptions import BacktestError
from quantlab.log import get_logger

__all__ = ["ExecutionSimulator", "FillTiming"]

logger = get_logger(__name__)


class FillTiming(StrEnum):
    """When orders generated at bar ``t`` execute."""

    NEXT_OPEN = "next_open"
    SAME_CLOSE = "same_close"


class ExecutionSimulator:
    """Fill simulator applied by the engine at each execution point.

    Parameters
    ----------
    fill_timing:
        See :class:`FillTiming`.
    slippage:
        Slippage model; defaults to :class:`VolumeImpactSlippage`.
    commission:
        Commission model; defaults to :class:`ZeroCommission`.
    max_participation:
        Cap on ``|filled quantity| / bar volume`` per symbol per bar;
        ``None`` disables the cap (and it cannot bind when volume is NaN).
    """

    def __init__(
        self,
        fill_timing: FillTiming = FillTiming.NEXT_OPEN,
        slippage: SlippageModel | None = None,
        commission: CommissionModel | None = None,
        max_participation: float | None = 0.1,
    ) -> None:
        if max_participation is not None and not 0 < max_participation <= 1:
            raise BacktestError("max_participation must be in (0, 1] or None")
        self.fill_timing = FillTiming(fill_timing)
        self.slippage = slippage if slippage is not None else VolumeImpactSlippage()
        self.commission = commission if commission is not None else ZeroCommission()
        self.max_participation = max_participation

    def execute(
        self,
        orders: list[Order],
        timestamp: pd.Timestamp,
        prices: pd.Series,
        volumes: pd.Series | None = None,
        sigmas: pd.Series | None = None,
    ) -> tuple[list[FillEvent], list[Order]]:
        """Fill ``orders`` against one bar.

        Parameters
        ----------
        prices:
            Reference execution prices per symbol (opens for ``NEXT_OPEN``,
            closes for ``SAME_CLOSE``); NaN price skips the order.
        volumes / sigmas:
            Per-symbol bar volume and trailing daily vol estimates used by
            participation caps and impact models. Both optional.

        Returns
        -------
        (fills, skipped):
            Executed fills and the orders that could not fill at all
            (missing price, or cap left zero fillable quantity).
        """
        fills: list[FillEvent] = []
        skipped: list[Order] = []
        for order in orders:
            price = prices.get(order.symbol)
            if price is None or not math.isfinite(price) or price <= 0:
                logger.warning(
                    "skipping order %s: no tradable price at %s",
                    order, timestamp.date(),
                )
                skipped.append(order)
                continue

            volume = None
            if volumes is not None:
                raw_v = volumes.get(order.symbol)
                volume = float(raw_v) if raw_v is not None and math.isfinite(raw_v) else None

            qty = order.quantity
            if self.max_participation is not None and volume is not None and volume > 0:
                cap = self.max_participation * volume
                if abs(qty) > cap:
                    qty = math.copysign(cap, qty)
            if qty == 0.0:
                skipped.append(order)
                continue

            sigma = None
            if sigmas is not None:
                raw_s = sigmas.get(order.symbol)
                sigma = float(raw_s) if raw_s is not None and math.isfinite(raw_s) else None

            frac = self.slippage.slippage_fraction(qty, float(price), volume, sigma)
            if frac < 0:
                raise BacktestError(
                    f"Slippage model returned negative fraction {frac} for {order}"
                )
            direction = 1.0 if qty > 0 else -1.0
            exec_price = float(price) * (1.0 + direction * frac)
            fills.append(
                FillEvent(
                    timestamp=timestamp,
                    symbol=order.symbol,
                    quantity=qty,
                    price=exec_price,
                    commission=self.commission.commission(qty, exec_price),
                    slippage_cost=abs(qty) * float(price) * frac,
                    order_id=order.order_id,
                )
            )
        return fills, skipped
