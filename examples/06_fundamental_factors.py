"""Example 06: fundamental factors with point-in-time discipline.

Full chain: synthetic market + synthetic quarterly statements -> validated
point-in-time fundamental panel -> earnings-yield factor -> signal pipeline
-> long/short target weights -> event-driven backtest with realistic costs
-> tearsheet. Runs fully offline.

Two things to notice, both deliberate:

1. **Publication timing.** Each statement carries the date it became
   usable (period end + a 45-day assumed lag). The factor value on a date
   uses only statements published by that date — the March quarter does
   not exist for the strategy until mid-May.
2. **Honest expectations.** Synthetic fundamentals are generated
   independently of synthetic prices, so the fundamentals themselves carry
   no signal — an earnings LEVEL factor has an IC statistically at zero.
   The earnings YIELD traded here divides by price, which makes it mildly
   anti-momentum (cheap = has fallen), and because the synthetic market
   has permanent per-asset drift, momentum persists and the yield factor
   inherits a small *negative* IC. That is an artifact of the price model
   (and a real phenomenon: value and momentum are negatively correlated),
   not alpha or lookahead. This example demonstrates plumbing and
   no-lookahead discipline; if you ever see a fundamental factor with a
   spectacular Sharpe on synthetic data, hunt for the bug.

    python examples/06_fundamental_factors.py
"""

from __future__ import annotations

from pathlib import Path

from quantlab.backtest import (
    BacktestEngine,
    ExecutionSimulator,
    PerShareCommission,
    TargetWeightStrategy,
    VolumeImpactSlippage,
    signal_to_weights,
)
from quantlab.data import (
    FundamentalDataValidator,
    SyntheticFundamentalSource,
    make_synthetic_dataset,
)
from quantlab.research import (
    EarningsYield,
    Rank,
    SignalPipeline,
    Winsorize,
    evaluate_signal,
    signal_turnover,
)
from quantlab.viz import apply_style, tearsheet

N_ASSETS = 30
SEED = 7


def main() -> None:
    # 1. Prices: six years of the deterministic synthetic market.
    ds = make_synthetic_dataset(
        n_assets=N_ASSETS, start="2018-01-02", end="2023-12-29", n_sectors=5, seed=SEED
    )
    print(f"prices: {ds}")

    # 2. Fundamentals: same universe naming, fetched from TWO YEARS BEFORE
    #    the backtest start. Trailing-twelve-month and year-over-year
    #    constructions need 4-8 quarters of history before their first
    #    usable value — starting the fundamental fetch at the backtest start
    #    would silently cost the first two years of signal.
    source = SyntheticFundamentalSource(
        n_assets=N_ASSETS, seed=SEED, inception="2016-01-02"
    )
    fundamentals = source.fetch(source.symbols, "2016-01-02", "2023-12-29")
    print(f"fundamentals: {fundamentals}")
    lag = fundamentals.metadata["publication_lag_days"]
    print(f"publication basis: {fundamentals.metadata['publication_basis']} "
          f"({lag} days after period end)")

    # 3. Validate the statements before trusting them.
    report = FundamentalDataValidator().validate(fundamentals)
    print(f"validation: {report}")
    report.raise_for_errors()

    # 4. Factor: TTM earnings yield, point-in-time aligned. The score frame
    #    has the same shape as any price factor and slots into the same
    #    pipeline machinery.
    raw = EarningsYield(fundamentals).compute(ds)
    signal = SignalPipeline([Winsorize(0.01, 0.99), Rank()]).transform(raw)

    # 5. Evaluate against forward returns. The fundamentals carry no signal
    #    by construction; the mildly negative IC is the yield's price
    #    denominator making it anti-momentum in a drift-persistent market
    #    (see module docstring).
    ev = evaluate_signal(signal, ds.close, name="earnings_yield", horizon=21)
    print(f"\nIC (21-day horizon): mean {ev.summary['ic_mean']:+.4f}, "
          f"t(NW) {ev.summary['t_stat_nw']:+.2f}  <- mildly negative: "
          "anti-momentum artifact, not signal")
    print(f"signal rank turnover: {signal_turnover(signal).mean():.4f} "
          "(quarterly statements -> far below any daily price factor)")

    # 6. Long/short portfolio, monthly rebalance, realistic costs.
    weights = signal_to_weights(
        signal, long_frac=0.2, short_frac=0.2, gross=1.0, rebalance_every=21
    )
    execution = ExecutionSimulator(
        slippage=VolumeImpactSlippage(spread_bps=5, impact_coeff=0.1),
        commission=PerShareCommission(rate=0.005, minimum=1.0),
        max_participation=0.05,
    )
    strategy = TargetWeightStrategy(weights=weights, tolerance=0.001)
    result = BacktestEngine(ds, strategy, execution, initial_capital=1_000_000).run()
    print(f"\n{result}")
    print("\nperformance (long/short earnings yield, realistic costs):")
    print(result.summary().round(4).to_string())

    # 7. Tearsheet.
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    apply_style()
    fig = tearsheet(result)
    out_path = out_dir / "06_fundamental_tearsheet.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\ntearsheet saved to {out_path}")


if __name__ == "__main__":
    main()
