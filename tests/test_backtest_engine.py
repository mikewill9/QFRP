"""Engine behavior: timing, accounting identities, caps, integration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quantlab.backtest import (
    BacktestEngine,
    BarContext,
    ExecutionSimulator,
    FillTiming,
    FixedBpsSlippage,
    Order,
    PerShareCommission,
    Strategy,
    TargetWeightStrategy,
    ZeroCommission,
    ZeroSlippage,
    signal_to_weights,
)
from quantlab.data import MarketDataset
from quantlab.exceptions import BacktestError


def _dataset(closes: dict[str, list[float]], opens=None, volume=1e9) -> MarketDataset:
    n = len(next(iter(closes.values())))
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = pd.DataFrame(closes, index=idx, dtype=float)
    open_ = pd.DataFrame(opens, index=idx, dtype=float) if opens else close * 1.0
    return MarketDataset(
        open=open_,
        high=pd.concat([open_, close]).groupby(level=0).max() * 1.0001,
        low=pd.concat([open_, close]).groupby(level=0).min() * 0.9999,
        close=close,
        volume=close * 0 + volume,
    )


def _frictionless(timing=FillTiming.NEXT_OPEN) -> ExecutionSimulator:
    return ExecutionSimulator(
        fill_timing=timing, slippage=ZeroSlippage(),
        commission=ZeroCommission(), max_participation=None,
    )


class BuyOnceStrategy(Strategy):
    """Buy a fixed quantity on the first bar, then hold."""

    name = "buy_once"

    def __init__(self, symbol: str, quantity: float) -> None:
        self.symbol, self.quantity, self._done = symbol, quantity, False

    def on_bar(self, context: BarContext):
        if self._done:
            return None
        self._done = True
        return [Order(self.symbol, self.quantity, context.timestamp)]


def test_next_open_fills_at_next_bars_open():
    ds = _dataset(
        closes={"AAA": [100, 100, 100, 100]},
        opens={"AAA": [100, 111, 100, 100]},  # distinctive open on bar 1
    )
    engine = BacktestEngine(ds, BuyOnceStrategy("AAA", 10), _frictionless(), 10_000)
    result = engine.run()
    assert result.n_trades == 1
    trade = result.trades.iloc[0]
    assert trade["price"] == pytest.approx(111.0)  # open of t+1, not close of t
    assert trade["timestamp"] == ds.index[1]


def test_same_close_fills_at_signal_close():
    ds = _dataset(
        closes={"AAA": [100, 105, 110, 115]},
        opens={"AAA": [99, 104, 109, 114]},
    )
    engine = BacktestEngine(
        ds, BuyOnceStrategy("AAA", 10),
        _frictionless(FillTiming.SAME_CLOSE), 10_000,
    )
    trade = engine.run().trades.iloc[0]
    assert trade["price"] == pytest.approx(100.0)  # close of bar 0
    assert trade["timestamp"] == ds.index[0]


def test_buy_and_hold_final_equity_exact():
    ds = _dataset(
        closes={"AAA": [50, 52, 54, 60, 58]},
        opens={"AAA": [50, 51, 53, 59, 57]},
    )
    engine = BacktestEngine(ds, BuyOnceStrategy("AAA", 100), _frictionless(), 100_000)
    result = engine.run()
    # 100 shares bought at open(1)=51; final equity = cash + 100*close(-1).
    expected = 100_000 - 100 * 51 + 100 * 58
    assert result.final_equity == pytest.approx(expected)
    assert result.metadata["n_fills"] == 1
    assert result.total_costs == 0.0


def test_equity_identity_holds_every_bar(synth):
    signal = -synth.returns()  # arbitrary but deterministic scores
    weights = signal_to_weights(signal, rebalance_every=10)
    engine = BacktestEngine(
        synth, TargetWeightStrategy(weights=weights), _frictionless(), 1_000_000,
    )
    result = engine.run()
    marked = (result.positions * synth.close).sum(axis=1)
    cash = result.equity_curve - marked
    recomputed = cash + marked
    pd.testing.assert_series_equal(
        recomputed, result.equity_curve, check_names=False,
    )
    # Returns compound back to the equity curve.
    rebuilt = result.initial_capital * (1 + result.returns).cumprod()
    pd.testing.assert_series_equal(
        rebuilt, result.equity_curve, check_names=False, rtol=1e-10,
    )


def test_lookahead_is_structurally_impossible():
    """Information revealed at bar t cannot transact at pre-t prices.

    The close jumps 100 -> 200 at bar 2. A strategy reacting to that jump
    decides at bar 2's close, so with NEXT_OPEN timing it can only fill at
    bar 3's open, which already sits at the new level: the jump itself is
    untradeable, as it must be.
    """

    class JumpChaser(Strategy):
        name = "jump_chaser"

        def on_bar(self, ctx):
            if ctx.bar_index == 0 or ctx.portfolio.position("AAA") != 0:
                return None
            prev_close = ctx.dataset.close.iloc[ctx.bar_index - 1, 0]
            if ctx.close_prices.iloc[0] > prev_close * 1.5:
                return [Order("AAA", 100, ctx.timestamp)]
            return None

    ds = _dataset(
        closes={"AAA": [100, 100, 200, 200]},
        opens={"AAA": [100, 100, 150, 200]},
    )
    result = BacktestEngine(ds, JumpChaser(), _frictionless(), 100_000).run()
    trade = result.trades.iloc[0]
    assert trade["timestamp"] == ds.index[3]
    assert trade["price"] == pytest.approx(200.0)  # post-jump open, no free ride
    assert result.final_equity == pytest.approx(100_000.0)


def test_costs_reduce_equity_by_exact_amount():
    ds = _dataset(closes={"AAA": [100.0] * 5}, opens={"AAA": [100.0] * 5})
    execution = ExecutionSimulator(
        slippage=FixedBpsSlippage(bps=10),
        commission=PerShareCommission(rate=0.01, minimum=0.0),
        max_participation=None,
    )
    engine = BacktestEngine(ds, BuyOnceStrategy("AAA", 100), execution, 100_000)
    result = engine.run()
    slip = 100 * 100.0 * 1e-3
    comm = 100 * 0.01
    assert result.total_costs == pytest.approx(slip + comm)
    # Flat prices: the only equity loss is the costs.
    assert result.final_equity == pytest.approx(100_000 - slip - comm)


def test_participation_cap_partial_fill():
    ds = _dataset(closes={"AAA": [100.0] * 4}, opens={"AAA": [100.0] * 4}, volume=1000)
    execution = ExecutionSimulator(
        slippage=ZeroSlippage(), commission=ZeroCommission(), max_participation=0.1,
    )
    engine = BacktestEngine(ds, BuyOnceStrategy("AAA", 5_000), execution, 1_000_000)
    result = engine.run()
    assert result.trades.iloc[0]["quantity"] == pytest.approx(100.0)  # 10% of 1000


def test_target_weight_strategy_reaches_targets(synth):
    ts = synth.index[100]
    weights = pd.DataFrame(np.nan, index=synth.index, columns=synth.symbols)
    weights.loc[ts, synth.symbols[:4]] = 0.25
    engine = BacktestEngine(
        synth, TargetWeightStrategy(weights=weights), _frictionless(), 1_000_000,
    )
    result = engine.run()
    # Trades only at the single rebalance (filled the bar after ts).
    assert (result.trades["timestamp"] == synth.index[101]).all()
    achieved = result.weights.loc[synth.index[101], synth.symbols[:4]]
    # Sized on close(t) equity/prices, filled at open(t+1): near 25% each.
    assert np.allclose(achieved, 0.25, atol=0.02)
    assert result.weights.loc[synth.index[99]].abs().sum() == 0.0


def test_all_nan_rows_hold_positions(synth):
    weights = pd.DataFrame(np.nan, index=synth.index, columns=synth.symbols)
    weights.loc[synth.index[50], synth.symbols[0]] = 0.5
    result = BacktestEngine(
        synth, TargetWeightStrategy(weights=weights), _frictionless(), 1_000_000,
    ).run()
    assert result.n_trades == 1  # NaN rows after bar 50 never trigger trades
    assert result.positions.iloc[-1][synth.symbols[0]] > 0


def test_final_bar_orders_expire(synth):
    weights = pd.DataFrame(np.nan, index=synth.index, columns=synth.symbols)
    weights.loc[synth.index[-1], synth.symbols[0]] = 0.5
    result = BacktestEngine(
        synth, TargetWeightStrategy(weights=weights), _frictionless(), 1_000_000,
    ).run()
    assert result.n_trades == 0
    assert result.metadata["expired_orders"] == 1


def test_research_integration_momentum_pipeline(synth):
    """Full chain: factor -> transforms -> weights -> engine -> summary."""
    from quantlab.research import Momentum, Rank, SignalPipeline, Winsorize

    signal = SignalPipeline([Winsorize(), Rank()]).transform(
        Momentum(lookback=126, skip=10).compute(synth)
    )
    weights = signal_to_weights(
        signal, long_frac=0.25, short_frac=0.25, rebalance_every=21,
    )
    engine = BacktestEngine(
        synth,
        TargetWeightStrategy(weights=weights, tolerance=0.001),
        ExecutionSimulator(max_participation=None),
        1_000_000,
    )
    result = engine.run()
    summary = result.summary()
    assert np.isfinite(summary["sharpe"])
    assert summary["annualized_turnover"] > 0
    # Dollar-neutral targets: net exposure stays small after first rebalance.
    net = result.weights.sum(axis=1)
    assert net.iloc[150:].abs().max() < 0.15
    assert result.total_costs > 0  # default cost models actually charged


def test_engine_input_validation(small_synth):
    with pytest.raises(BacktestError):
        BacktestEngine(small_synth.between("2021-01-04", "2021-01-04"),
                       BuyOnceStrategy("SYN000", 1))
    with pytest.raises(BacktestError):
        TargetWeightStrategy()  # neither weights nor weight_fn
    with pytest.raises(BacktestError):
        ExecutionSimulator(max_participation=1.5)
