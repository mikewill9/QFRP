# QuantResearchAI Project Handoff

## Project Overview

QuantResearchAI is a Python quantitative finance research framework.

The goal is to build a serious research environment for exploring quantitative investing ideas.

This project is NOT intended to be a magic trading bot.

The purpose is to build a platform for:

- financial data research
- factor development
- signal generation
- portfolio construction
- strategy backtesting
- risk analysis
- learning quantitative finance methodology

The long-term goal is to demonstrate:

- programming ability
- statistical thinking
- research discipline
- software engineering practices


# Current Repository State

## Git History

Completed checkpoints:

- Checkpoint 9: point-in-time universe expansion (roadmap Stage 4)
- Checkpoint 8: multi-factor ML combination (roadmap Stage 3)
- Checkpoint 7: point-in-time fundamental data layer and fundamental factors
- Checkpoint 6: visualization and attribution
- Checkpoint 5: ML alpha pipeline
- Checkpoint 4: portfolio and risk layer
- Checkpoint 3: backtesting engine
- Checkpoint 2: research subsystem
- Checkpoint 1: initial quantlab framework

Latest checkpoint commit:

8ec5f2a Checkpoint 9: point-in-time universe expansion (Stage 4)

The repository is the source of truth.

Do not rely on previous chat history.

## Environment

- Python 3.14 virtual environment at `.venv` (not committed)
- Install with: `pip install -e ".[dev]"`
- Validate with: `python -m pytest` (459 tests), `ruff check src tests examples`, `mypy`
- Everything runs offline on the synthetic market; no downloads needed
- Test suite runs matplotlib headless (Agg backend, wired in tests/conftest.py)


# Completed Work


## Checkpoint 1: Initial Framework + Data Infrastructure

Completed:

- Python package structure
- configuration system
- logging system
- data architecture
- data source interfaces
- data caching
- data validation

This created the foundation for future research modules.


## Checkpoint 2: Research Subsystem

Completed:

- factor framework (momentum, mean reversion, volatility, liquidity, statistical)
- signal generation, transformations, pipelines
- research evaluation (IC analysis, quantile analysis, turnover)
- experiment tracking


## Checkpoint 3: Backtesting Engine

Completed:

- event-driven architecture (market, order, fill events)
- execution simulation (costs, commission, slippage, fill timing)
- portfolio accounting (positions, trades, equity)
- performance analytics (returns, Sharpe, Sortino, drawdown, CAGR, turnover)

Design goals: avoid lookahead bias, include realistic costs,
separate research from execution, maintain test coverage.


## Checkpoint 4: Portfolio and Risk Layer

Completed:

### Portfolio (src/quantlab/portfolio/)

- covariance: sample, EWMA, Ledoit-Wolf constant-correlation shrinkage
- constraints: bounds/leverage/net caps with approximate projection
  and exact SLSQP hooks
- optimizers: inverse volatility, equal risk contribution,
  mean-variance, max Sharpe, hierarchical risk parity,
  volatility targeting wrapper
- rebalancing: no-trade bands and turnover caps

Optimizers raise on singular/ill-conditioned covariance rather than
silently regularizing. Covariance is per-period; annualization happens
at the point of use.

### Risk (src/quantlab/risk/)

- metrics: vol, VaR and Expected Shortfall (historical, normal,
  Cornish-Fisher), beta
- drawdowns: series, episode table, ulcer index
- exposures: gross/net, sector, factor (OLS)
- limits: declarative RiskLimit framework with RiskMonitor
- reporting: RiskReport.from_backtest


## Checkpoint 5: ML Alpha Pipeline

Completed (src/quantlab/ml/):

- datasets: wide factors -> supervised (date, symbol) panel with
  forward-return targets; the standard alignment is causal by
  construction
- validation: purged k-fold with embargo; walk-forward with a purge
  gap defaulting to the label horizon (strictly past -> future)
- models: ridge, random forest, gradient boosting (lightgbm optional,
  chosen at fit time, never imported at module load), ensembles
- training: walk-forward trainer producing an out-of-sample signal
  wide frame plus per-fold audit table
- evaluation: regression metrics, OOS rank IC (same IC machinery as
  the research layer), permutation importance
- persistence: versioned model registry with JSON manifests


## Checkpoint 6: Visualization and Attribution

Completed:

### Attribution (src/quantlab/analytics/attribution.py)

- asset_contributions(weights, asset_returns): prior-bar weight x
  realized return, per asset per bar
- factor_attribution(returns, factor_returns) -> FactorAttribution:
  OLS alpha, betas, R^2, plus a per-period contributions frame
  (alpha, one column per factor, residual) that sums exactly back to
  the input returns

Both use the standard linear-attribution approximation (no compounding
cross-terms, no costs); collinear factors raise rather than returning
an arbitrary least-squares solution.

### Visualization (src/quantlab/viz/)

- style.py: apply_style(), PALETTE
- performance.py: plot_equity_and_drawdown, plot_rolling_stats,
  plot_monthly_heatmap
- factors.py: plot_ic, plot_quantile_returns, plot_factor_correlation
- portfolio.py: plot_weights, plot_contributions
- tearsheet.py: tearsheet(result, risk_report=None) -> Figure

Every function accepts an optional ax/axes and returns a Figure; a
function only calls fig.tight_layout() when it created its own figure,
so panels compose cleanly inside tearsheet's GridSpec. Nothing calls
plt.show(). risk_report is optional so a plain backtest result
tearsheets without a hard dependency on quantlab.risk.

**All subsystems specified in docs/architecture.md are now implemented.**


## Checkpoint 7: Point-in-Time Fundamental Data Layer and Fundamental Factors

Completed:

### Fundamentals (src/quantlab/data/fundamentals.py)

- FundamentalDataset: statement panel where every observation carries
  both its fiscal period_end AND the publication_date it became usable;
  as_of(date) accessor; restatement versions supported
- alignment kernels: pit_align (latest published value per date; a
  late-restated old quarter never displaces a newer public one),
  trailing_stat (TTM sums / stability stds, usable at the window's max
  publication date), trailing_change (YoY with |base| denominator),
  combine_events
- sources: FundamentalSource ABC; SyntheticFundamentalSource
  (deterministic, window-independent, internally consistent statements —
  all tests and examples run offline); YahooFundamentalSource (behind
  the data extra; assumed 45-day publication lag, documented as
  optimistic for fiscal Q4 10-Ks); CachedFundamentalSource (parquet
  cache that accumulates quarters snapshot vendors later drop)
- validation: FundamentalDataValidator plus five checks (missing
  statements, non-positive values, accounting identity, stale values,
  restatement detection) in data/validation.py

### Fundamental factors (src/quantlab/research/factors/fundamental.py)

- value: EarningsYield, BookToMarket, FCFYield
- quality: ReturnOnEquity, GrossMarginStability, Accruals
- growth: RevenueGrowth, EarningsGrowth (TTM or quarterly YoY)
- EstimateRevision: interface only — raises DataError until a genuine
  point-in-time estimates source is connected (yfinance has no
  estimate history; backtesting current snapshots would be lookahead)

Governing constraint, pinned by publication-straddle tests: a factor
value at date t changes only when information published on or before t
changes.

Also delivered: examples/06_fundamental_factors.py (offline),
architecture.md section 4.9, and docs/research_roadmap.md — the staged
research plan with mechanical implementation specs for stages 3 and 4.

Known limits with free data (documented in docstrings and the roadmap):
Yahoo serves only ~4-6 trailing quarters, no filing dates, silent
restatements, survivorship-biased universe. Real multi-year fundamental
research needs cache accumulation over time or a point-in-time vendor.


## Checkpoint 8: Multi-Factor ML Combination (Roadmap Stage 3)

Completed (executed exactly per docs/research_roadmap.md Appendix B):

### Feature assembly (src/quantlab/ml/features.py)

- FeatureSpec(name, factor, transform=None): one feature — a factor
  plus optional post-compute normalization
- build_features(dataset, specs) -> dict[str, WideFrame]: computes
  each factor and applies its transform, producing the mapping
  PanelDataset.from_factors consumes; validates non-empty specs and
  unique names, preserves order. Fundamental factors carry their own
  FundamentalDataset from construction, so mixed price/fundamental
  feature sets need nothing extra.

### Model selection (src/quantlab/ml/selection.py)

- compare_models(panel, model_factories, splitter) -> DataFrame:
  one row per candidate (mean_r2, std_r2, mean_mse, mean_rank_ic,
  n_folds), every candidate scored on identical folds, fresh model
  per (model, fold), sorted by mean_rank_ic. SELECTION ONLY — its
  scores are interior-fold numbers; walk-forward remains the only
  output causal enough to quote or backtest.
- Dates where predictions (or targets) are constant are skipped when
  averaging rank IC: Spearman is undefined there, not zero.

### Panel targets (src/quantlab/ml/datasets.py)

- PanelDataset.from_factors gained target_transform: "raw" (default,
  exact backward compatibility), "demeaned" (per-date cross-sectional
  demeaning — the recommended setting; the label is relative return
  and the model never fits the market level), "rank" (centered
  percentile ranks in (-0.5, 0.5)).

Also delivered: examples/07_multifactor_ml.py (offline: six normalized
price+fundamental features -> demeaned panel -> purged model comparison
-> one walk-forward -> costed long/short backtest), 12 new tests
(tests/test_ml_features.py, tests/test_ml_selection.py, extended
tests/test_ml_datasets.py), architecture.md section 4.6 and README
updates.

Sanity result pinned by the example: on the synthetic market ridge won
selection and the ML blend's OOS rank IC (+0.06) sat below raw
momentum's (+0.09), with fundamental features contributing ~nothing —
correct by construction; a blend that beat everything on synthetic data
would indicate a leak.


## Checkpoint 9: Point-in-Time Universe Expansion (Roadmap Stage 4)

Completed (executed exactly per docs/research_roadmap.md Appendix C):

### Universe (src/quantlab/data/universe.py)

- rolling_universe(dataset, top_n=100, window=63, min_price=5.0,
  rebalance_every=21): boolean dates x symbols membership mask. At each
  rebalance bar (first at bar index window-1), symbols are ranked by
  trailing mean dollar volume computed THROUGH that bar only; the top-N
  with trailing median close >= min_price are members until the next
  rebalance. NaN trailing volume excludes; rows before the first
  rebalance are all False; top_n above the eligible count keeps all;
  ties resolve by pandas nlargest order (documented).
- apply_universe(frame, universe): NaNs cells outside membership,
  reindexing the mask to the frame with missing entries as False.
  Masks only — never renormalizes.
- US_LARGE_CAP_150 / US_LARGE_CAP_SECTORS: 150 tickers with GICS-style
  sectors; the comment block states the survivorship warning (survivors
  only, results overstate historical returns).
- 8 tests in tests/test_data_universe.py, including the mandatory PIT
  lookahead pin: volume perturbed strictly AFTER a rebalance bar leaves
  membership at that bar unchanged (mirror of the momentum-skip test).

Also delivered: examples/08_large_universe.py (offline, 150 synthetic
names -> 100-name PIT universe -> sector-neutralized momentum -> breadth
comparison -> costed long/short backtest), architecture.md section 4.1
bullet, README updates.

Sanity result pinned by the example: per-date IC std fell 0.207 -> 0.108
moving from 30 to 100 names (almost exactly the sqrt-of-breadth
shrinkage). The 30-name subset drew a HIGHER ic_mean (0.048 vs 0.018) by
luck, so the example prints ic_std and explains that a small universe's
IC level is dominated by the draw — rather than asserting the originally
planned "IR rises with breadth" claim, which this draw contradicted.

Known limits: membership TIMING is point-in-time by construction;
membership COMPOSITION is survivorship-biased with free data (the seed
list holds today's survivors; no free source carries delisted names).
State the caveat on every real-data result (roadmap Appendix A).


# Current Development Stage

There is no active checkpoint. The architecture.md contract (sections
4.1 through 4.9: data, research, backtest, portfolio, risk, ml,
analytics, viz, fundamentals) is fully built, tested, and documented.

docs/research_roadmap.md is the plan of record for future work:

- Stage 5 (next): sentiment / NLP — outlined only, DATA-GATED: it needs
  a genuinely point-in-time text source (SEC EDGAR full text is the free
  option; Tiingo/Polygon news are paid). No mechanical appendix spec
  exists yet — write one (in the style of Appendices B/C) and agree it
  with the user before implementing anything.
- Stage 6 (macro regime overlays): outlined only, data-gated (ALFRED
  vintages for point-in-time macro).

Before starting any stage: inspect the repository, read README.md,
docs/architecture.md, and docs/research_roadmap.md (its appendices are
written to be executed as specified), and confirm scope with the user.


# Development Rules

When continuing:

1. Inspect the repository first.

2. Read:

- README.md
- docs/architecture.md
- existing tests

3. Do not redesign the architecture without a strong reason.

4. Preserve existing interfaces.

5. Prioritize:

- correct mathematics
- clean code
- tests
- documentation

over unnecessary complexity.


# Validation Requirements

Before completing any future checkpoint:

- run the test suite
- verify existing functionality still works
- add tests for new functionality
- document important assumptions
