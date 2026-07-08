"""Example 04: optimizer-driven portfolio with risk management.

Full chain: synthetic market -> rolling Ledoit-Wolf covariance ->
risk-based optimizer (inverse-vol / ERC / HRP) with volatility targeting
-> no-trade-band rebalancing -> event-driven backtest with realistic
costs -> consolidated risk report with sector exposures and limits.
Runs fully offline.

    python examples/04_optimized_portfolio.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.backtest import (
    BacktestEngine,
    BarContext,
    ExecutionSimulator,
    PerShareCommission,
    TargetWeightStrategy,
    VolumeImpactSlippage,
)
from quantlab.data import make_synthetic_dataset
from quantlab.portfolio import (
    EqualRiskContribution,
    HierarchicalRiskParity,
    InverseVolatility,
    Optimizer,
    RebalancePolicy,
    VolatilityTargeting,
    ledoit_wolf_cc,
)
from quantlab.risk import GrossLeverageLimit, PositionLimit, RiskReport

LOOKBACK = 252  # trailing window for covariance estimation
TARGET_VOL = 0.10  # 10% annualized ex-ante volatility


def make_weight_fn(optimizer: Optimizer, policy: RebalancePolicy):
    """Close the optimizer and rebalance policy over the bar context.

    At each rebalance: estimate a shrunk covariance from trailing log
    returns (data through today's close only), let the optimizer set
    targets, then let the policy drop trades too small to pay for.
    """

    def weight_fn(context: BarContext) -> pd.Series | None:
        prices = context.history("close", window=LOOKBACK + 1)
        if len(prices) < LOOKBACK + 1:
            return None  # not enough history yet
        returns = np.log(prices).diff().dropna(how="all")
        cov, _ = ledoit_wolf_cc(returns)
        target = VolatilityTargeting(optimizer, target=TARGET_VOL).weights(None, cov)
        current = context.portfolio.weights().reindex(target.index).fillna(0.0)
        return policy.apply(current, target)

    return weight_fn


def run(ds, optimizer: Optimizer):
    strategy = TargetWeightStrategy(
        weight_fn=make_weight_fn(optimizer, RebalancePolicy(band=0.002)),
        rebalance_every=21,
    )
    execution = ExecutionSimulator(
        slippage=VolumeImpactSlippage(spread_bps=5, impact_coeff=0.1),
        commission=PerShareCommission(rate=0.005, minimum=1.0),
        max_participation=0.05,
    )
    return BacktestEngine(ds, strategy, execution, initial_capital=1_000_000).run()


def main() -> None:
    ds = make_synthetic_dataset(
        n_assets=40, start="2018-01-02", end="2023-12-29", n_sectors=5, seed=7
    )
    print(f"dataset: {ds}")

    optimizers: dict[str, Optimizer] = {
        "inverse_vol": InverseVolatility(),
        "erc": EqualRiskContribution(),
        "hrp": HierarchicalRiskParity(),
    }
    results = {name: run(ds, opt) for name, opt in optimizers.items()}

    table = pd.concat(
        {name: result.summary() for name, result in results.items()}, axis=1
    )
    print("\nperformance (vol-targeted to 10%, realistic costs):")
    print(table.round(4).to_string())

    # Deep risk inspection of the ERC book.
    report = RiskReport.from_backtest(
        results["erc"],
        sectors=ds.metadata["sectors"],
        limits=[GrossLeverageLimit(max_gross=2.0), PositionLimit(max_weight=0.10)],
    )
    print(f"\n{report}")


if __name__ == "__main__":
    main()
