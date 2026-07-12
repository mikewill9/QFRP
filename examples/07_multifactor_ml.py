"""Example 07: multi-factor ML combination (Stage 3).

Full chain: synthetic market + synthetic quarterly statements -> six
price AND fundamental features assembled with build_features -> supervised
panel with a cross-sectionally demeaned target (the model predicts
RELATIVE return within the universe, never the market level) -> purged-CV
model comparison (selection only) -> ONE walk-forward run with the chosen
model -> long/short backtest with realistic costs. Runs fully offline.

Two disciplines to notice, both deliberate:

1. **Selection vs results.** compare_models scores interior purged folds —
   fine for choosing a model, never quotable as performance. Only the
   walk-forward output (every prediction fitted strictly on the past) may
   be called out-of-sample and backtested. One walk-forward per finalized
   configuration; re-running after peeking is training on the test set.
2. **Honest expectations.** Synthetic fundamentals are generated
   independently of synthetic prices, so fundamental features contribute
   roughly nothing here, and the ML blend should land near the best single
   price factor. The value of this example is the *workflow*; a dramatic
   outperformance on synthetic data means a leak, not a discovery.

    python examples/07_multifactor_ml.py
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
from quantlab.data import SyntheticFundamentalSource, make_synthetic_dataset
from quantlab.ml import (
    FeatureSpec,
    GradientBoostingModel,
    LinearModel,
    PanelDataset,
    PurgedKFold,
    TreeModel,
    WalkForward,
    WalkForwardTrainer,
    build_features,
    compare_models,
    prediction_rank_ic,
)
from quantlab.research import (
    EarningsYield,
    Momentum,
    RealizedVolatility,
    ReturnOnEquity,
    RevenueGrowth,
    ShortTermReversal,
    SignalPipeline,
    Winsorize,
    ZScore,
)

N_ASSETS = 30
SEED = 7
HORIZON = 21  # predict the relative return over the next month of bars


def main() -> None:
    # 1. Prices: six years of the deterministic synthetic market.
    #    Fundamentals: same universe naming, fetched from TWO YEARS BEFORE
    #    the backtest start. Trailing-twelve-month and year-over-year
    #    constructions need 4-8 quarters of history before their first
    #    usable value — starting the fundamental fetch at the backtest start
    #    would silently cost the first two years of signal.
    ds = make_synthetic_dataset(
        n_assets=N_ASSETS, start="2018-01-02", end="2023-12-29", n_sectors=5, seed=SEED
    )
    source = SyntheticFundamentalSource(
        n_assets=N_ASSETS, seed=SEED, inception="2016-01-02"
    )
    fundamentals = source.fetch(source.symbols, "2016-01-02", "2023-12-29")
    print(f"prices: {ds}")
    print(f"fundamentals: {fundamentals}")

    # 2. Features: three price factors and three fundamental factors, every
    #    one winsorized then z-scored per date so no feature smuggles in a
    #    market-level component or an incomparable scale.
    norm = SignalPipeline([Winsorize(), ZScore()])
    specs = [
        FeatureSpec("momentum", Momentum(lookback=252, skip=21), norm),
        FeatureSpec("st_reversal", ShortTermReversal(lookback=5), norm),
        FeatureSpec("realized_vol", RealizedVolatility(window=63), norm),
        FeatureSpec("earnings_yield", EarningsYield(fundamentals), norm),
        FeatureSpec("roe", ReturnOnEquity(fundamentals), norm),
        FeatureSpec("revenue_growth", RevenueGrowth(fundamentals), norm),
    ]
    features = build_features(ds, specs)
    print(f"features: {list(features)}")

    # 3. Supervised panel with a DEMEANED target: the label is each name's
    #    forward return minus the date's cross-sectional mean, i.e. relative
    #    attractiveness — the model has no business fitting the market.
    panel = PanelDataset.from_factors(
        features, ds.close, horizon=HORIZON, target_transform="demeaned"
    )
    print(f"panel: {panel}")

    # 4. Model comparison on identical purged folds. Quarterly fundamentals
    #    make adjacent months informationally correlated beyond the label
    #    window, so an embargo of a month on top of the horizon purge.
    factories = {
        "ridge": lambda: LinearModel(alpha=1.0),
        "random_forest": lambda: TreeModel(n_estimators=200, max_depth=3),
        "gradient_boosting": lambda: GradientBoostingModel(),
    }
    table = compare_models(
        panel, factories, PurgedKFold(n_splits=5, horizon=HORIZON, embargo=21)
    )
    print("\nmodel comparison (SELECTION ONLY -- interior purged folds,")
    print("never quotable as out-of-sample results):")
    print(table.round(4).to_string())

    # 5. ONE walk-forward run with the winner. Every prediction comes from
    #    a model fitted strictly on the past; this is the only output that
    #    may be called out-of-sample.
    best = str(table.index[0])
    print(f"\nwalk-forward with the selected model: {best}")
    trainer = WalkForwardTrainer(
        model_factory=factories[best],
        splitter=WalkForward(train_periods=252, test_periods=63),
    )
    signal = trainer.run(panel)
    print(f"trained: {signal}")

    ml_ic = prediction_rank_ic(signal.predictions, ds.close, horizon=HORIZON)
    print(f"\nOOS rank IC on the walk-forward dates (horizon {HORIZON}):")
    print(f"  {'ml_blend':<16s} {ml_ic.mean():+.4f}")
    for name in features:
        single = prediction_rank_ic(
            features[name].loc[signal.predictions.index], ds.close, horizon=HORIZON
        )
        print(f"  {name:<16s} {single.mean():+.4f}")

    # 6. Long/short portfolio on the OOS signal, monthly rebalance,
    #    realistic costs.
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
    print("\nperformance (long/short on the OOS ML blend, realistic costs):")
    print(result.summary().round(4).to_string())

    # 7. Honest closing note.
    print(
        "\nOn synthetic data the ML blend should roughly match the best single"
        "\nprice factor, and the fundamental features should contribute about"
        "\nnothing (statements are generated independently of prices). The"
        "\nvalue of this example is the workflow -- feature assembly, demeaned"
        "\ntargets, purged selection, one walk-forward — and any dramatic"
        "\noutperformance on synthetic data means a leak, not a discovery."
    )


if __name__ == "__main__":
    main()
