"""Example 03: momentum long/short backtest.

Full chain: synthetic market -> momentum factor -> signal pipeline ->
long/short target weights -> event-driven backtest with realistic costs ->
performance summary, including a frictionless run for cost attribution.
Runs fully offline.

    python examples/03_momentum_backtest.py
"""

from __future__ import annotations

import pandas as pd

from quantlab.backtest import (
    BacktestEngine,
    ExecutionSimulator,
    PerShareCommission,
    TargetWeightStrategy,
    VolumeImpactSlippage,
    ZeroCommission,
    ZeroSlippage,
    signal_to_weights,
)
from quantlab.data import make_synthetic_dataset
from quantlab.research import Momentum, Rank, SignalPipeline, Winsorize


def build_weights(ds) -> pd.DataFrame:
    """Momentum scores -> normalized signal -> monthly L/S weights."""
    raw = Momentum(lookback=252, skip=21).compute(ds)
    signal = SignalPipeline([Winsorize(0.01, 0.99), Rank()]).transform(raw)
    return signal_to_weights(
        signal, long_frac=0.2, short_frac=0.2, gross=1.0, rebalance_every=21,
    )


def run(ds, weights, execution) -> pd.Series:
    strategy = TargetWeightStrategy(weights=weights, tolerance=0.001)
    result = BacktestEngine(ds, strategy, execution, initial_capital=1_000_000).run()
    return result


def main() -> None:
    ds = make_synthetic_dataset(
        n_assets=40, start="2018-01-02", end="2023-12-29", n_sectors=5, seed=7
    )
    weights = build_weights(ds)
    n_rebalances = int(weights.notna().any(axis=1).sum())
    print(f"dataset: {ds}")
    print(f"monthly rebalances scheduled: {n_rebalances}")

    realistic = ExecutionSimulator(
        slippage=VolumeImpactSlippage(spread_bps=5, impact_coeff=0.1),
        commission=PerShareCommission(rate=0.005, minimum=1.0),
        max_participation=0.05,
    )
    frictionless = ExecutionSimulator(
        slippage=ZeroSlippage(), commission=ZeroCommission(), max_participation=None,
    )

    result = run(ds, weights, realistic)
    ideal = run(ds, weights, frictionless)

    print(f"\n{result}")
    print(f"fills: {result.metadata['n_fills']}, "
          f"skipped: {result.metadata['n_skipped']}, "
          f"total costs: ${result.total_costs:,.0f}")

    table = pd.concat(
        [result.summary().rename("with_costs"), ideal.summary().rename("frictionless")],
        axis=1,
    )
    print("\nperformance:")
    print(table.round(4).to_string())

    cost_drag = ideal.summary()["cagr"] - result.summary()["cagr"]
    print(f"\nannualized cost drag: {cost_drag:.2%}")

    gross = result.weights.abs().sum(axis=1)
    net = result.weights.sum(axis=1)
    print(f"avg gross exposure: {gross.iloc[252:].mean():.2f}, "
          f"avg |net|: {net.iloc[252:].abs().mean():.3f}")


if __name__ == "__main__":
    main()
