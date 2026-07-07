"""Example 02: factor research workflow.

Computes several factors on a synthetic market, normalizes them through a
signal pipeline (winsorize -> zscore -> sector neutralize), evaluates each
against forward returns, and logs everything to the experiment tracker.
Runs fully offline.

    python examples/02_factor_research.py
"""

from __future__ import annotations

import tempfile

import pandas as pd

from quantlab.data import SyntheticSource
from quantlab.research import (
    ExperimentTracker,
    Momentum,
    Neutralize,
    PriceToMovingAverage,
    RealizedVolatility,
    ShortTermReversal,
    SignalPipeline,
    Winsorize,
    ZScore,
    evaluate_signal,
)


def main() -> None:
    # 1. Data: 40 assets, 5 sectors, six years.
    source = SyntheticSource(n_assets=40, n_sectors=5, seed=7, inception="2018-01-02")
    ds = source.fetch(source.symbols, "2018-01-02", "2023-12-29")
    sectors = source.sector_map()
    print(f"dataset: {ds}")

    # 2. Candidate factors.
    factors = [
        Momentum(lookback=252, skip=21),
        ShortTermReversal(lookback=5),
        PriceToMovingAverage(window=21),
        RealizedVolatility(window=63),
    ]

    # 3. One shared normalization pipeline.
    pipeline = SignalPipeline([
        Winsorize(0.01, 0.99),
        ZScore(),
        Neutralize(sectors),
    ])

    # 4. Evaluate each factor at a 5-day horizon and track the runs.
    horizon = 5
    with tempfile.TemporaryDirectory() as tmp:
        tracker = ExperimentTracker(root=tmp)
        headlines = []
        for factor in factors:
            raw = factor.compute(ds)
            signal = pipeline.transform(raw)
            ev = evaluate_signal(
                signal,
                ds.close,
                name=factor.name,
                horizon=horizon,
                n_quantiles=5,
                params=factor.params,
            )
            run = tracker.start_run(
                "factor_screen",
                {"factor": factor.name, "horizon": horizon, **factor.params},
            )
            run.log_metrics({
                "ic_mean": ev.summary["ic_mean"],
                "ic_ir": ev.summary["ic_ir"],
                "t_stat_nw": ev.summary["t_stat_nw"],
                "spread_mean": ev.spread.mean(),
                "rank_turnover": ev.turnover.mean(),
            })
            run.log_frame("quantile_returns", ev.quantile_rets)
            run.finish()
            headlines.append(ev.headline())
            print(f"  evaluated {factor.name:<16} "
                  f"IC {ev.summary['ic_mean']:+.4f}  "
                  f"t(NW) {ev.summary['t_stat_nw']:+.2f}  "
                  f"turnover {ev.turnover.mean():.3f}")

        # 5. Compare runs by IC, straight from the tracker on disk.
        board = tracker.compare("ic_mean", "factor_screen")
        cols = ["config.factor", "metric.ic_mean", "metric.t_stat_nw",
                "metric.rank_turnover"]
        print("\nleaderboard (by ic_mean):")
        print(board[cols].to_string(index=False))

    # 6. Side-by-side headline table.
    table = pd.DataFrame(headlines)
    print("\nheadline comparison:")
    print(table.round(4).to_string())


if __name__ == "__main__":
    main()
