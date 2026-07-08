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

- Checkpoint 5: ML alpha pipeline
- Checkpoint 4: portfolio and risk layer
- Checkpoint 3: backtesting engine
- Checkpoint 2: research subsystem
- Checkpoint 1: initial quantlab framework

Current HEAD:

0656e5b Checkpoint 5: ML alpha pipeline

The repository is the source of truth.

Do not rely on previous chat history.

## Environment

- Python 3.14 virtual environment at `.venv` (not committed)
- Install with: `pip install -e ".[dev]"`
- Validate with: `python -m pytest` (340 tests), `ruff check src tests examples`, `mypy`
- Everything runs offline on the synthetic market; no downloads needed


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

Note: analytics/attribution.py (architecture.md section 4.7) was
deferred from Checkpoint 4 and remains unimplemented.


# Current Development Stage

## Checkpoint 6: Visualization + Analytics Attribution

This is the next milestone.


# Visualization Module

Expected location:

src/quantlab/viz/

Contracts are pinned in docs/architecture.md section 4.8.

Expected components:

- style.py: apply_style()
- performance.py: plot_equity_and_drawdown, plot_rolling_stats,
  plot_monthly_heatmap
- factors.py: plot_ic, plot_quantile_returns, plot_factor_correlation
- portfolio.py: plot_weights, plot_contributions
- tearsheet.py: tearsheet(result) -> Figure

Rules:

- all functions accept and return matplotlib objects
- nothing calls plt.show()
- viz may import every other subsystem


# Attribution Module

Expected location:

src/quantlab/analytics/attribution.py

Contract is pinned in docs/architecture.md section 4.7:

- asset_contributions(weights, asset_returns)
- factor_attribution(returns, factor_returns) with alpha, betas, R^2


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

Before completing a checkpoint:

- run the test suite
- verify existing functionality still works
- add tests for new functionality
- document important assumptions


# Next Milestone

Complete:

1. Visualization layer (style, performance, factors, portfolio, tearsheet)

2. Analytics attribution

3. Tests (matplotlib figures can be tested headless via the Agg backend)

4. Documentation

Then create a Git checkpoint:

Checkpoint 6: visualization and attribution
