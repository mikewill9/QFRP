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

- Checkpoint 6: visualization and attribution
- Checkpoint 5: ML alpha pipeline
- Checkpoint 4: portfolio and risk layer
- Checkpoint 3: backtesting engine
- Checkpoint 2: research subsystem
- Checkpoint 1: initial quantlab framework

Current HEAD:

306f123 Checkpoint 6: visualization and attribution

The repository is the source of truth.

Do not rely on previous chat history.

## Environment

- Python 3.14 virtual environment at `.venv` (not committed)
- Install with: `pip install -e ".[dev]"`
- Validate with: `python -m pytest` (408 tests), `ruff check src tests examples`, `mypy`
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


# Current Development Stage

There is no active checkpoint. The architecture.md contract (sections
4.1 through 4.8: data, research, backtest, portfolio, risk, ml,
analytics, viz) is fully built, tested, and documented.

Possible directions for future work (none currently planned/approved):

- CLI or notebook-driven research workflow examples beyond examples/01-05
- additional data sources / live execution adapters
- deeper ML model families (e.g. neural nets) behind the same
  ModelWrapper protocol
- packaging/release polish (versioning, PyPI-style distribution)

Before starting any new direction: inspect the repository, read
README.md and docs/architecture.md, and confirm scope with the user
rather than assuming one of the above.


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
