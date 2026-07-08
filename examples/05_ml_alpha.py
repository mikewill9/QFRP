"""Example 05: ML alpha pipeline, end to end.

Full chain: synthetic market -> factor computation -> supervised panel
(features at t, forward return over (t, t+h]) -> walk-forward training
(every prediction comes from a model fitted strictly on the past) ->
out-of-sample signal -> long/short backtest with realistic costs ->
performance summary. Runs fully offline.

    python examples/05_ml_alpha.py
"""

from __future__ import annotations

from quantlab.backtest import (
    BacktestEngine,
    ExecutionSimulator,
    PerShareCommission,
    TargetWeightStrategy,
    VolumeImpactSlippage,
    signal_to_weights,
)
from quantlab.data import make_synthetic_dataset
from quantlab.ml import (
    LinearModel,
    PanelDataset,
    WalkForward,
    WalkForwardTrainer,
    prediction_rank_ic,
)
from quantlab.research import Momentum, RealizedVolatility, ShortTermReversal, ZScore

HORIZON = 21  # predict the return over the next month of bars


def main() -> None:
    ds = make_synthetic_dataset(
        n_assets=30, start="2018-01-02", end="2023-12-29", n_sectors=5, seed=7
    )
    print(f"dataset: {ds}")

    # Features: classic factors, z-scored cross-sectionally so the linear
    # model sees comparable scales.
    zscore = ZScore()
    factors = {
        "momentum": Momentum(lookback=252, skip=21),
        "reversal": ShortTermReversal(lookback=5),
        "low_vol": RealizedVolatility(window=63),
    }
    features = {name: zscore.transform(f.compute(ds)) for name, f in factors.items()}

    panel = PanelDataset.from_factors(features, ds.close, horizon=HORIZON)
    print(f"panel: {panel}")

    # Walk-forward: train on a year, skip the label horizon, predict a
    # quarter, roll forward. Every prediction is out of sample.
    trainer = WalkForwardTrainer(
        model_factory=lambda: LinearModel(alpha=1.0),
        splitter=WalkForward(train_periods=252, test_periods=63),
    )
    signal = trainer.run(panel)
    print(f"trained: {signal}")

    ic = prediction_rank_ic(signal.predictions, ds.close, horizon=HORIZON)
    momentum_ic = prediction_rank_ic(
        features["momentum"].loc[signal.predictions.index], ds.close, horizon=HORIZON
    )
    print(
        f"\nOOS rank IC:  ml={ic.mean():+.4f}  "
        f"raw momentum={momentum_ic.mean():+.4f}  (same dates)"
    )

    weights = signal_to_weights(
        signal.predictions, long_frac=0.2, short_frac=0.2, gross=1.0,
        rebalance_every=HORIZON,
    )
    execution = ExecutionSimulator(
        slippage=VolumeImpactSlippage(spread_bps=5, impact_coeff=0.1),
        commission=PerShareCommission(rate=0.005, minimum=1.0),
        max_participation=0.05,
    )
    strategy = TargetWeightStrategy(weights=weights, tolerance=0.001)
    result = BacktestEngine(ds, strategy, execution, initial_capital=1_000_000).run()

    print(f"\n{result}")
    print("\nperformance (long/short on the OOS ML signal, realistic costs):")
    print(result.summary().round(4).to_string())

    print("\nper-fold training windows:")
    print(
        signal.folds[["train_start", "train_end", "test_start", "test_end"]]
        .map(lambda ts: ts.date())
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
