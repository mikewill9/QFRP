# quantlab

An institutional-style quantitative research framework: market data
infrastructure, factor research, an event-driven backtesting engine,
portfolio construction, risk monitoring, and a machine-learning alpha
pipeline, built as one coherent codebase with a shared set of data
conventions.

quantlab is designed to be the foundation a systematic research effort can
grow on for years: strict layering, small stable interfaces, deterministic
offline operation, and a test suite that pins down the quantitative
behavior of every component.

## Design principles

- **Correctness first.** No-lookahead is a structural property of the
  backtester (signals on close, fills on next open), not a convention users
  must remember. Every quantitative kernel is covered by tests against
  hand-computed cases.
- **Deterministic and offline by default.** A factor-model synthetic market
  generator (`quantlab.data.SyntheticSource`) drives all tests, examples,
  and notebooks. Live data (Yahoo Finance) is an optional extra, always used
  through the cache.
- **One data convention everywhere.** Wide frames (dates x symbols, tz-naive
  sorted `DatetimeIndex`) for prices, returns, signals, and weights. Long
  frames appear only at IO and ML boundaries.
- **Strict layering.** Lower layers never import higher ones:
  `data -> research -> backtest -> portfolio/risk -> ml -> analytics/viz`.
- **Soft optional dependencies.** `yfinance` and `lightgbm` are extras; the
  core never imports them at module load.

## Repository layout

```
src/quantlab/
├── config.py, log.py, types.py, exceptions.py   # cross-cutting core
├── data/                    # ingestion, caching, storage, validation
│   ├── dataset.py           # MarketDataset (aligned OHLCV panel)
│   ├── sources/             # DataSource ABC, synthetic, CSV, Yahoo
│   ├── cache.py             # CachedSource (incremental parquet cache)
│   ├── store.py             # atomic parquet IO, named DatasetStore
│   ├── validation.py        # composable data-quality checks
│   └── preprocessing.py     # returns, fills, resampling, universe filters
├── research/                # alpha research
│   ├── factors/             # Factor ABC + momentum, mean reversion,
│   │                        #   volatility, liquidity, statistical (PCA)
│   ├── signals.py           # cross-sectional transforms and pipelines
│   ├── evaluation.py        # IC analysis, quantile portfolios, decay
│   └── experiments.py       # local experiment registry
├── backtest/                # event-driven engine
│   ├── events.py, orders.py # event types, order lifecycle
│   ├── execution.py         # fill simulation, participation limits
│   ├── costs.py             # commission + slippage/impact models
│   ├── portfolio.py         # position and cash accounting
│   ├── strategy.py          # Strategy ABC, TargetWeightStrategy
│   ├── engine.py            # BacktestEngine event loop
│   └── results.py           # BacktestResult container
├── portfolio/               # portfolio construction
│   ├── covariance.py        # sample, EWMA, Ledoit-Wolf shrinkage
│   ├── optimizers.py        # vol targeting, risk parity (ERC),
│   │                        #   mean-variance, max Sharpe, HRP
│   ├── constraints.py       # bounds, leverage, long-only
│   └── rebalance.py         # no-trade bands, turnover caps
├── risk/                    # monitoring and limits
│   ├── metrics.py           # vol, VaR/ES (hist, normal, Cornish-Fisher)
│   ├── drawdown.py          # drawdown series, episodes, stats
│   ├── exposures.py         # beta/factor/sector exposure tracking
│   ├── limits.py            # declarative limit framework
│   └── report.py            # consolidated risk report
├── ml/                      # ML alpha pipeline
│   ├── datasets.py          # wide factors -> supervised panel
│   ├── validation.py        # purged k-fold w/ embargo, walk-forward
│   ├── models.py            # linear, tree, ensemble wrappers
│   ├── training.py          # walk-forward trainer -> OOS signal
│   ├── evaluation.py        # metrics, rank IC, importances
│   └── persistence.py       # model registry with manifests
├── analytics/               # performance statistics
│   ├── performance.py       # Sharpe, Sortino, Calmar, CAGR, ...
│   └── attribution.py       # asset and factor attribution
└── viz/                     # matplotlib plotting + tearsheets
```

Supporting directories: `tests/` (pytest suite), `examples/` (end-to-end
scripts, all offline), `notebooks/` (research walkthrough), `docs/`
(architecture and per-subsystem documentation).

## Installation

```bash
pip install -e ".[dev]"          # core + test/lint tooling
pip install -e ".[dev,data,ml]"  # + yfinance and lightgbm extras
```

Requires Python 3.11+.

## Quickstart

```python
from quantlab.data import (
    DataValidator, make_synthetic_dataset, filter_universe,
)

# Deterministic 20-asset market with sector structure; runs offline.
ds = make_synthetic_dataset(n_assets=20, start="2020-01-01", end="2023-12-29")

report = DataValidator().validate(ds)
report.raise_for_errors()
print(report)

universe = filter_universe(ds, min_median_dollar_volume=1e6)
returns = universe.returns(method="log")
```

Fetching real data through the cache (requires `pip install -e ".[data]"`):

```python
from quantlab.data import CachedSource, YahooFinanceSource

source = CachedSource(YahooFinanceSource())
prices = source.fetch(["AAPL", "MSFT", "NVDA"], "2018-01-01", "2024-12-31")
```

Repeated fetches are served from the local parquet cache; extending the date
range downloads only the missing gap.

## Data conventions

- Prices are **fully adjusted** for splits and dividends. Sources deliver
  adjusted series; the framework applies no corporate actions itself.
- Indices are tz-naive, sorted, and unique; the `MarketDataset` constructor
  enforces this so downstream code never re-checks.
- `method="log"` returns are the research default; `"simple"` returns are
  used for portfolio accounting.
- Missing observations are NaN; forward-filling is explicit, capped, and
  zero-volume-tagged (`forward_fill_prices`).

## Development

```bash
python -m pytest            # full suite, offline, deterministic
ruff check src tests        # lint
mypy                        # type check
```

## Status

The subsystems are being built in dependency order; `docs/architecture.md`
pins the interfaces each layer exposes. Data infrastructure is complete and
tested; research, backtesting, portfolio construction, risk, ML, analytics,
and visualization follow in that order.

## Disclaimer

quantlab is a research framework. Nothing in this repository is investment
advice, and simulated results do not represent live trading.
