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
│   ├── fundamentals.py      # point-in-time fundamental statements:
│   │                        #   FundamentalDataset, publication-date
│   │                        #   alignment, synthetic/Yahoo sources, cache
│   ├── cache.py             # CachedSource (incremental parquet cache)
│   ├── store.py             # atomic parquet IO, named DatasetStore
│   ├── validation.py        # composable data-quality checks
│   │                        #   (prices and fundamentals)
│   └── preprocessing.py     # returns, fills, resampling, universe filters
├── research/                # alpha research
│   ├── factors/             # Factor ABC + momentum, mean reversion,
│   │                        #   volatility, liquidity, statistical (PCA),
│   │                        #   fundamental (value, quality, growth)
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
│   ├── features.py          # factor specs -> normalized feature frames
│   ├── validation.py        # purged k-fold w/ embargo, walk-forward
│   ├── models.py            # linear, tree, ensemble wrappers
│   ├── selection.py         # purged-CV model comparison (selection only)
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

## Fundamental data (point-in-time)

Quarterly statements carry both their fiscal `period_end` and the
`publication_date` they became usable; factors only ever see statements
already published — no-lookahead stays structural. Where true filing dates
are unavailable (Yahoo), a conservative 45-day publication lag is assumed
and recorded in the dataset metadata.

```python
from quantlab.data import SyntheticFundamentalSource
from quantlab.research import EarningsYield

# Deterministic offline statements; fetch 6-8 quarters before your first
# evaluation date so trailing-twelve-month windows have history.
source = SyntheticFundamentalSource(n_assets=20, seed=7, inception="2016-01-02")
fundamentals = source.fetch(source.symbols, "2016-01-02", "2023-12-29")

signal = EarningsYield(fundamentals).compute(ds)  # standard wide signal frame
```

Live fundamentals come from `CachedFundamentalSource(YahooFundamentalSource())`
(requires the `data` extra). Yahoo serves only the trailing ~4-6 quarters and
no filing dates; the cache accumulates quarters across periodic fetches, which
is the honest way to build point-in-time history from a snapshot vendor. See
`docs/architecture.md` §4.9 and `docs/research_roadmap.md` for what paid
sources unlock.

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
pins the interfaces each layer exposes. Complete and tested: data
infrastructure, factor research (factors, signals, evaluation, experiment
tracking), the event-driven backtest engine, performance analytics,
portfolio construction (covariance estimation, optimizers, constraints,
rebalancing), risk monitoring (tail metrics, drawdowns, exposures,
limits, reporting), the ML alpha pipeline (supervised panels,
purged/walk-forward validation, model wrappers, OOS signal training,
model registry, multi-factor feature assembly, purged-CV model
selection), performance attribution (asset and factor),
visualization (performance/factor/portfolio charts, tearsheets), and the
point-in-time fundamental data layer with value/quality/growth factors.
All subsystems in `docs/architecture.md` are implemented; see
`docs/research_roadmap.md` for the staged research plan from here.

## Disclaimer

quantlab is a research framework. Nothing in this repository is investment
advice, and simulated results do not represent live trading.
