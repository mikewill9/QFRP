"""Example 08: universe expansion to 100+ names (Stage 4).

Cross-sectional signals are statistical bets, and their information ratio
scales roughly with the square root of independent breadth: a 5-quantile
portfolio on 30 names holds 6 per bucket (single-stock noise dominates),
while 100 names hold 20 per bucket. This example runs the same momentum
signal on a 100-name point-in-time universe and on a 30-name subset, and
the thing to look at is the ERROR BARS — the per-date IC std shrinking
roughly with the square root of breadth — not the IC level itself, which
on a 30-name subset is dominated by the luck of the draw. Runs fully
offline on the deterministic synthetic market.

Real-data path (documentation only, not executed here): with the ``data``
extra installed, the same workflow runs on
``CachedSource(YahooFinanceSource())`` fetched over ``US_LARGE_CAP_150``,
plus ``CachedFundamentalSource(YahooFundamentalSource())`` for the
fundamental factors. The first fetch is slow; subsequent runs are served
from the cache. Results on that list carry survivorship bias — it holds
today's survivors only, so historical backtests on it overstate returns,
full stop (see the quantlab.data.universe docstring and
docs/research_roadmap.md Appendix A).

    python examples/08_large_universe.py
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
from quantlab.data import apply_universe, make_synthetic_dataset, rolling_universe
from quantlab.research import (
    Momentum,
    Neutralize,
    SignalPipeline,
    Winsorize,
    ZScore,
    forward_returns,
    ic_series,
    ic_summary,
)

SEED = 7
HORIZON = 21


def main() -> None:
    # 1. 150 synthetic names across 10 sectors, six years of daily bars.
    ds = make_synthetic_dataset(
        n_assets=150, start="2018-01-02", end="2023-12-29", n_sectors=10, seed=SEED
    )
    print(f"prices: {ds}")

    # 2. Point-in-time membership: every month, the top 100 names by the
    #    trailing quarter's mean dollar volume THROUGH that bar only —
    #    membership at t never sees data after t.
    universe = rolling_universe(ds, top_n=100, window=63, rebalance_every=21)
    print("\nmembers per bar:")
    print(universe.sum(axis=1).describe().round(2).to_string())

    # 3. Momentum -> winsorize -> z-score -> sector-neutralize, then NaN
    #    everything outside the universe. Masking comes last so the signal
    #    only ever bets on names that were investable at the time.
    sectors = ds.metadata["sectors"]
    pipeline = SignalPipeline([Winsorize(), ZScore(), Neutralize(sectors)])
    raw = Momentum(lookback=252, skip=21).compute(ds)
    signal = apply_universe(pipeline.transform(raw), universe)

    # 4. Same signal, two breadths. The IC level is similar by construction
    #    (it is the same factor on the same market); what breadth buys is
    #    tighter error bars — watch ic_ir and t_stat_nw, not ic_mean.
    sub = ds.select(ds.symbols[:30])
    sub_signal = SignalPipeline(
        [Winsorize(), ZScore(), Neutralize(sectors)]
    ).transform(Momentum(lookback=252, skip=21).compute(sub))

    fwd_100 = forward_returns(ds.close, horizon=HORIZON)
    fwd_30 = forward_returns(sub.close, horizon=HORIZON)
    summary_100 = ic_summary(ic_series(signal, fwd_100), lags=HORIZON - 1)
    summary_30 = ic_summary(ic_series(sub_signal, fwd_30), lags=HORIZON - 1)

    print(f"\nmomentum IC at horizon {HORIZON}, 100-name universe vs 30 names:")
    print(f"  {'':<12s}{'100-name':>12s}{'30-name':>12s}")
    for stat in ("ic_mean", "ic_std", "ic_ir", "t_stat_nw"):
        print(f"  {stat:<12s}{summary_100[stat]:>12.4f}{summary_30[stat]:>12.4f}")
    print(
        "  Look at ic_std: per-date IC noise shrinks roughly with the square"
        "\n  root of breadth (3.3x the names -> about half the std). That is"
        "\n  what breadth buys -- a per-date IC measured with tighter error"
        "\n  bars. ic_mean differs between the two columns because a 30-name"
        "\n  subset is a noisy draw whose luck dominates its measured IC --"
        "\n  which is precisely the small-universe problem breadth fixes."
    )

    # 5. Long/short backtest on the universed signal, monthly rebalance,
    #    realistic costs and participation limits.
    weights = signal_to_weights(
        signal, long_frac=0.2, short_frac=0.2, gross=1.0, rebalance_every=HORIZON
    )
    execution = ExecutionSimulator(
        slippage=VolumeImpactSlippage(spread_bps=5, impact_coeff=0.1),
        commission=PerShareCommission(rate=0.005, minimum=1.0),
        max_participation=0.05,
    )
    strategy = TargetWeightStrategy(weights=weights, tolerance=0.001)
    result = BacktestEngine(ds, strategy, execution, initial_capital=1_000_000).run()
    print(f"\n{result}")
    print("\nperformance (long/short momentum on the 100-name universe):")
    print(result.summary().round(4).to_string())

    print(
        "\nOn real data this exact workflow additionally requires the"
        "\nsurvivorship disclaimer from quantlab.data.universe: a seed list"
        "\nof today's large caps contains only survivors, so historical"
        "\nresults on it overstate returns."
    )


if __name__ == "__main__":
    main()
