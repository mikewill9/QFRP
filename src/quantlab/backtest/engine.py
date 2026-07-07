"""Event-driven backtest engine.

Per-bar loop (``NEXT_OPEN`` timing, the default):

1. Fill orders queued at the previous close against this bar's **open**.
2. Mark the portfolio to this bar's **close**; record equity, weights,
   positions, and costs.
3. Emit the bar to the strategy; queue any returned orders for the next
   bar's open.

Orders generated on the final bar never fill (there is no next open) and
are reported in ``metadata["expired_orders"]``. With ``SAME_CLOSE`` timing,
step 3's orders are filled immediately at this bar's close and the mark is
refreshed; that timing is optimistic by construction and intended only for
quick comparisons.

The volatility estimate a fill "knows" at bar ``t`` is precomputed with a
one-bar shift, so it uses data through ``t-1`` only; participation caps and
impact use the fill bar's own volume, the standard daily-bar treatment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quantlab.backtest.execution import ExecutionSimulator, FillTiming
from quantlab.backtest.orders import Order
from quantlab.backtest.portfolio import Portfolio
from quantlab.backtest.results import BacktestResult
from quantlab.backtest.strategy import BarContext, Strategy
from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import BacktestError
from quantlab.log import get_logger

__all__ = ["BacktestEngine"]

logger = get_logger(__name__)

_SIGMA_WINDOW = 63


class BacktestEngine:
    """Run one strategy over one dataset with one execution model."""

    def __init__(
        self,
        dataset: MarketDataset,
        strategy: Strategy,
        execution: ExecutionSimulator | None = None,
        initial_capital: float = 1_000_000.0,
    ) -> None:
        if dataset.n_periods < 2:
            raise BacktestError("Dataset must contain at least 2 bars")
        self.dataset = dataset
        self.strategy = strategy
        self.execution = execution if execution is not None else ExecutionSimulator()
        self.initial_capital = float(initial_capital)

    # ------------------------------------------------------------------ run
    def run(self) -> BacktestResult:
        ds = self.dataset
        index = ds.index
        n_t, n_sym = ds.n_periods, ds.n_symbols
        opens, closes, volumes = ds.open, ds.close, ds.volume

        # Impact inputs known strictly before each bar (shifted by one).
        rets = ds.returns(method="log")
        sigma = rets.rolling(_SIGMA_WINDOW, min_periods=20).std().shift(1)

        portfolio = Portfolio(self.initial_capital)
        self.strategy.on_start(ds)

        equity = np.empty(n_t)
        costs = np.zeros(n_t)
        weights_arr = np.zeros((n_t, n_sym))
        positions_arr = np.zeros((n_t, n_sym))
        col_of = {s: j for j, s in enumerate(ds.symbols)}

        pending: list[Order] = []
        next_open = self.execution.fill_timing is FillTiming.NEXT_OPEN
        n_orders = n_fills = n_skipped = 0

        for i in range(n_t):
            ts = index[i]

            # 1. Fill orders queued at the previous close at today's open.
            if next_open and pending:
                fills, skipped = self.execution.execute(
                    pending, ts,
                    prices=opens.iloc[i],
                    volumes=volumes.iloc[i],
                    sigmas=sigma.iloc[i],
                )
                for f in fills:
                    portfolio.apply_fill(f)
                    costs[i] += f.commission + f.slippage_cost
                n_fills += len(fills)
                n_skipped += len(skipped)
                pending = []

            # 2. Mark to the close and snapshot state.
            portfolio.mark_to_market(closes.iloc[i])
            ctx = BarContext(
                timestamp=ts,
                bar_index=i,
                open_prices=opens.iloc[i],
                close_prices=closes.iloc[i],
                portfolio=portfolio,
                dataset=ds,
            )

            # 3. Strategy reacts to the completed bar.
            orders = self.strategy.on_bar(ctx) or []
            n_orders += len(orders)

            if orders and not next_open:
                fills, skipped = self.execution.execute(
                    orders, ts,
                    prices=closes.iloc[i],
                    volumes=volumes.iloc[i],
                    sigmas=sigma.iloc[i],
                )
                for f in fills:
                    portfolio.apply_fill(f)
                    costs[i] += f.commission + f.slippage_cost
                n_fills += len(fills)
                n_skipped += len(skipped)
                portfolio.mark_to_market(closes.iloc[i])
            elif orders:
                pending = list(orders)

            equity[i] = portfolio.equity
            for sym, qty in portfolio.positions.items():
                positions_arr[i, col_of[sym]] = qty
            eq = equity[i]
            if eq <= 0:
                raise BacktestError(
                    f"Equity went non-positive ({eq:,.2f}) on {ts.date()}; "
                    "the book is insolvent under these costs/weights"
                )
            w = portfolio.weights()
            for sym, wv in w.items():
                weights_arr[i, col_of[sym]] = wv

        equity_curve = pd.Series(equity, index=index, name="equity")
        prev = equity_curve.shift(1)
        prev.iloc[0] = self.initial_capital
        returns = (equity_curve / prev - 1.0).rename("returns")

        metadata: dict[str, Any] = {
            "strategy": self.strategy.name,
            "fill_timing": self.execution.fill_timing.value,
            "max_participation": self.execution.max_participation,
            "slippage_model": type(self.execution.slippage).__name__,
            "commission_model": type(self.execution.commission).__name__,
            "n_orders": n_orders,
            "n_fills": n_fills,
            "n_skipped": n_skipped,
            "expired_orders": len(pending),
        }
        logger.info(
            "backtest complete: %d bars, %d fills, final equity %.2f",
            n_t, n_fills, equity[-1],
        )
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            weights=pd.DataFrame(weights_arr, index=index, columns=ds.symbols),
            positions=pd.DataFrame(positions_arr, index=index, columns=ds.symbols),
            trades=portfolio.ledger.to_frame(),
            costs_paid=pd.Series(costs, index=index, name="costs"),
            initial_capital=self.initial_capital,
            metadata=metadata,
        )
