"""Quick research script: a simple 12-1 month momentum strategy.

What this script does, step by step:

    1. Download (and cache) real daily prices for 15 large-cap stocks
       from Yahoo Finance.
    2. Compute a "12-1 month momentum" factor for each stock: the return
       over the past 12 months, skipping the most recent month.
    3. Turn that factor into a long/short portfolio and run it through
       the event-driven backtester with realistic trading costs.
    4. Print a performance summary table to the terminal.
    5. Save a one-page "tearsheet" chart to outputs/momentum_tearsheet.png.

This requires the optional 'data' extra (installs yfinance) and an
internet connection on the first run:

    pip install -e ".[data]"
    python scripts/quick_research.py

After the first run, prices are cached on disk, so re-running the
script is fast and works offline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so this script can run without a display
# (e.g. over SSH or in CI) and never pops up a window.
matplotlib.use("Agg")

from quantlab.backtest import (
    BacktestEngine,
    ExecutionSimulator,
    PerShareCommission,
    TargetWeightStrategy,
    VolumeImpactSlippage,
    signal_to_weights,
)
from quantlab.data import CachedSource, DataValidator, YahooFinanceSource
from quantlab.research import Momentum, Rank, SignalPipeline, Winsorize
from quantlab.viz import apply_style, tearsheet

# ---------------------------------------------------------------------------
# 1. Settings for this research run.
# ---------------------------------------------------------------------------

# 15 large, liquid US stocks across different sectors, so the long/short
# portfolio isn't just one industry bet.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "JPM", "JNJ", "V",
    "PG", "HD", "MA", "UNH", "BAC", "XOM",
]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"

# Momentum is measured in trading days: 252 bars is about 12 months, and
# skipping the most recent 21 bars (~1 month) avoids the well-documented
# short-term reversal effect that would otherwise contaminate the signal.
MOMENTUM_LOOKBACK = 126
MOMENTUM_SKIP = 21

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
TEARSHEET_PATH = OUTPUT_DIR / "momentum_tearsheet.png"


def main() -> None:
    # -----------------------------------------------------------------
    # 2. Download and cache the data.
    #
    # YahooFinanceSource does the actual network download. Wrapping it in
    # CachedSource means the first run saves prices to disk (under
    # ~/.quantlab/cache by default), and every run after that reuses the
    # cached data instead of re-downloading it.
    # -----------------------------------------------------------------
    print(f"Fetching {len(TICKERS)} tickers from {START_DATE} to {END_DATE}...")
    source = CachedSource(YahooFinanceSource())
    dataset = source.fetch(TICKERS, START_DATE, END_DATE)
    print(f"Loaded: {dataset}")

    # Sanity-check the data before trusting it (missing bars, bad prices,
    # etc.). raise_for_errors() stops the script if something is seriously
    # wrong, rather than silently backtesting on bad data.
    report = DataValidator().validate(dataset)
    report.raise_for_errors()

    # -----------------------------------------------------------------
    # 3. Compute the momentum factor.
    #
    # Momentum(lookback=252, skip=21) scores each stock, on each day, by
    # its return over the past ~12 months, excluding the most recent
    # ~1 month. Higher score = stronger recent trend = more attractive.
    # -----------------------------------------------------------------
    print("Computing 12-1 month momentum factor...")
    raw_momentum = Momentum(lookback=MOMENTUM_LOOKBACK, skip=MOMENTUM_SKIP).compute(dataset)

    # Raw factor scores aren't ready to trade on directly. A SignalPipeline
    # cleans them up in two steps:
    #   - Winsorize: cap extreme outlier scores so one crazy value can't
    #     dominate the ranking.
    #   - Rank: convert scores to a 0-1 cross-sectional rank each day, so
    #     the signal only cares about relative order, not raw magnitude.
    signal = SignalPipeline([Winsorize(0.01, 0.99), Rank()]).transform(raw_momentum)

    # -----------------------------------------------------------------
    # 4. Turn the signal into a tradeable long/short portfolio.
    #
    # Each month, go long the top 20% of stocks by signal and short the
    # bottom 20%, with equal weight on each side (dollar-neutral overall).
    # -----------------------------------------------------------------
    weights = signal_to_weights(
        signal, long_frac=0.2, short_frac=0.2, gross=1.0, rebalance_every=21,
    )

    # -----------------------------------------------------------------
    # 5. Run the backtest with realistic trading costs.
    #
    # VolumeImpactSlippage models the extra cost of trading a large order
    # relative to how much the stock normally trades (market impact).
    # PerShareCommission charges a small fee per share, with a minimum
    # ticket cost, similar to a real broker.
    # -----------------------------------------------------------------
    print("Running backtest...")
    execution = ExecutionSimulator(
        slippage=VolumeImpactSlippage(spread_bps=5, impact_coeff=0.1),
        commission=PerShareCommission(rate=0.005, minimum=1.0),
        max_participation=0.05,  # never be more than 5% of a bar's volume
    )
    strategy = TargetWeightStrategy(weights=weights, tolerance=0.001)
    result = BacktestEngine(dataset, strategy, execution, initial_capital=1_000_000).run()

    # -----------------------------------------------------------------
    # 6. Print a performance summary.
    # -----------------------------------------------------------------
    print(f"\n{result}")
    print("\nPerformance summary:")
    print(result.summary().round(4).to_string())

    # -----------------------------------------------------------------
    # 7. Save a one-page tearsheet chart (equity curve, drawdown, rolling
    # stats, monthly returns, position weights) as a PNG.
    # -----------------------------------------------------------------
    OUTPUT_DIR.mkdir(exist_ok=True)
    apply_style()
    fig = tearsheet(result)
    fig.savefig(TEARSHEET_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved tearsheet to {TEARSHEET_PATH}")


if __name__ == "__main__":
    main()
