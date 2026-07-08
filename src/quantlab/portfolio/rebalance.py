"""Rebalancing policy.

Re-optimizing every period produces a stream of tiny trades whose costs
exceed their benefit. :class:`RebalancePolicy` sits between an optimizer's
target and the trades actually sent: per-asset **no-trade bands** drop
adjustments too small to matter, and a **turnover cap** limits how much of
the book may change hands in one rebalance.

Turnover here is one-way, matching ``analytics.performance.turnover``:
``0.5 * sum |w_new - w_current|`` (a full liquidation-and-rebuild of a
fully invested book is 1.0).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quantlab.exceptions import OptimizationError

__all__ = ["RebalancePolicy"]


@dataclass(frozen=True, slots=True)
class RebalancePolicy:
    """Map ``(current, target)`` weights to the weights actually traded to.

    Parameters
    ----------
    band:
        Per-asset no-trade band in weight units: assets whose target is
        within ``band`` of the current weight are left untouched.
    max_turnover:
        One-way turnover cap per rebalance. ``None`` means uncapped.
    partial:
        What to do when the desired trades exceed ``max_turnover``:
        ``True`` scales every trade down pro-rata so the cap binds
        exactly (move partway toward the target); ``False`` skips the
        rebalance entirely (all-or-nothing).
    """

    band: float = 0.0
    max_turnover: float | None = None
    partial: bool = True

    def __post_init__(self) -> None:
        if self.band < 0:
            raise OptimizationError(f"band must be >= 0, got {self.band}")
        if self.max_turnover is not None and self.max_turnover < 0:
            raise OptimizationError(
                f"max_turnover must be >= 0, got {self.max_turnover}"
            )

    def apply(self, current: pd.Series, target: pd.Series) -> pd.Series:
        """Weights to trade to, after bands and the turnover cap.

        ``current`` and ``target`` are aligned on the union of their
        indices (missing entries are 0: absent from the book / flat
        target).
        """
        index = current.index.union(target.index)
        cur = current.reindex(index).fillna(0.0).astype(float)
        tgt = target.reindex(index).fillna(0.0).astype(float)

        trades = tgt - cur
        if self.band > 0:
            trades[trades.abs() <= self.band] = 0.0

        turnover = 0.5 * float(trades.abs().sum())
        if self.max_turnover is not None and turnover > self.max_turnover:
            if not self.partial:
                return cur
            trades = trades * (self.max_turnover / turnover)
        return cur + trades

    def turnover(self, current: pd.Series, target: pd.Series) -> float:
        """One-way turnover of moving from ``current`` to ``apply`` output."""
        new = self.apply(current, target)
        index = current.index.union(new.index)
        cur = current.reindex(index).fillna(0.0)
        return 0.5 * float((new.reindex(index).fillna(0.0) - cur).abs().sum())
