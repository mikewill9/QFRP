# quantlab architecture

This document pins the conventions and public interfaces of every subsystem.
Later stages of the codebase are built against these contracts; anything
listed here is stable API within the 0.x series.

## 1. Layering

```
core (config, log, types, exceptions)
  └── data
        └── research
              └── backtest ── analytics
                    └── portfolio, risk
                          └── ml
                                └── viz
```

Rules:

- A layer may import from layers above it in this diagram (its dependencies),
  never below. `viz` may import everything; `data` imports only core.
- `analytics` depends only on core and plain pandas objects (return series,
  weight frames), so both `backtest` and `ml` can use it without cycles.
- Optional third-party packages (`yfinance`, `lightgbm`) are imported inside
  functions, never at module top level.

## 2. Data conventions (framework-wide)

- **Wide frame:** `pd.DataFrame`, rows = tz-naive sorted unique
  `DatetimeIndex`, columns = string symbols. Used for prices, returns,
  factor scores, signals, weights, positions.
- **Long frame:** columns `["date", "symbol", ...]`; IO and ML boundaries
  only.
- **Returns:** `"log"` = `ln(p_t/p_{t-k})` (research default);
  `"simple"` = `p_t/p_{t-k} - 1` (portfolio accounting).
- **Prices are adjusted** (splits, dividends) at the source. The framework
  performs no corporate-action handling.
- **Signals** are wide frames of real-valued scores where higher = more
  attractive. A signal value indexed at date `t` may use information up to
  and including the close of `t`.
- **Weights** are wide frames; row sums are the net exposure, absolute row
  sums the gross exposure. Cash is implicit (`1 - net` for long-only
  conventions is NOT assumed; engines track cash explicitly).

## 3. No-lookahead policy

- Factors/signals at date `t` see data through close of `t` only. Factor
  implementations must not use `t+1` information; the evaluation and ML
  layers align signals at `t` with **forward** returns over `(t, t+h]`.
- The backtester's default fill timing is `NEXT_OPEN`: orders generated from
  the close of `t` fill at the open of `t+1` plus slippage. `SAME_CLOSE`
  exists for research convenience and is explicitly labeled as optimistic.
- Rolling estimators (covariance, PCA, ML training windows) use trailing
  windows ending at the decision date.

## 4. Subsystem contracts

### 4.1 data (complete)

- `MarketDataset` (frozen dataclass): fields `open/high/low/close/volume`,
  `frequency`, `metadata`; key methods `returns`, `select`, `between`,
  `symbol_frame`, `from_symbol_frames`, `from_long`, `to_long`,
  `dollar_volume`, `with_metadata`, `equals`.
- `DataSource.fetch(symbols, start, end, frequency) -> MarketDataset`.
  Implementations: `SyntheticSource` (deterministic factor-model market;
  `sector_map()`, `beta_map()`), `CSVSource`, `YahooFinanceSource`.
- `CachedSource(source, cache_dir)`: incremental parquet cache keyed by
  covered request ranges; `cached_range`, `clear`.
- `DatasetStore(root)`: `save/load/exists/list` for named datasets.
- `DataValidator(checks).validate(ds) -> ValidationReport`
  (`passed`, `raise_for_errors()`, `to_frame()`); checks are
  `ValidationCheck` subclasses with severities INFO/WARNING/ERROR.
- Preprocessing: `compute_returns`, `forward_fill_prices`,
  `resample_dataset`, `align_intersection`, `filter_universe`,
  `rolling_dollar_volume`.

### 4.2 research (complete)

- `Factor` (ABC), module `research.factors.base`:
  `compute(dataset) -> WideFrame` of scores (higher = better),
  properties `name: str`, `min_history: int` (bars required before the first
  valid score). Constructor parameters are echoed in `params: dict` for
  experiment logging.
- Concrete factors:
  - momentum: `Momentum(lookback, skip)` (12-1 style),
    `TimeSeriesMomentum(lookback)` (sign/scale of own trailing return);
  - mean reversion: `ShortTermReversal(lookback)`,
    `PriceToMovingAverage(window)` (z-score of price vs MA, sign flipped);
  - volatility: `RealizedVolatility(window, estimator=cc|parkinson|garman_klass)`
    returning LOW-vol-favoring scores via `invert=True` default,
    `EWMAVolatility(halflife)`;
  - liquidity: `AmihudIlliquidity(window)`, `DollarVolume(window)`,
    `RollSpread(window)`;
  - statistical: `PCAResidualReversal(window, n_components, stride)`,
    `MarketBeta(window)`.
- `signals` module: cross-sectional transforms, each a small class with
  `transform(frame) -> frame`: `Winsorize`, `ZScore`, `Rank`, `Demean`,
  `Neutralize(groups)`, `Decay(halflife)`, `Lag(periods)`, `Clip`; composed
  by `SignalPipeline([...]).transform(frame)`.
- `evaluation` module: `ic_series(signal, forward_returns, method)`,
  `ic_summary(...)` (mean, std, IR, Newey-West t-stat),
  `quantile_returns(signal, forward_returns, n_quantiles)`,
  `quantile_spread(...)`, `signal_turnover(signal)`,
  `ic_decay(signal, returns, horizons)`, plus
  `evaluate_signal(...) -> SignalEvaluation` aggregating all of the above.
  `forward_returns(prices, horizon)` helper defines the alignment: value at
  `t` is the return over `(t, t+h]`.
- `experiments` module: `ExperimentTracker(root)` with
  `start_run(name, config) -> Run`; `Run.log_metrics(dict)`,
  `Run.log_frame(name, df)`, `Run.finish()`; content-addressed run ids;
  `tracker.runs()`, `tracker.compare(metric)`.

### 4.3 backtest (complete)

- Events (`events.py`): `MarketEvent(timestamp)`, `OrderEvent`,
  `FillEvent(symbol, quantity, price, commission, slippage)`.
- Orders (`orders.py`): `Order(symbol, quantity, order_type=MARKET,
  timestamp, id)`; positive quantity = buy.
- `CommissionModel` ABC (`costs.py`): `PerShareCommission`, `BpsCommission`,
  `ZeroCommission`. `SlippageModel` ABC: `FixedBpsSlippage`,
  `VolumeImpactSlippage(spread_bps, impact_coeff)` implementing a
  square-root market-impact term `k * sigma * sqrt(|q| / ADV)`.
- `ExecutionSimulator(fill_timing, slippage, commission,
  max_participation)`: fills orders against bar data; participation cap
  yields partial fills; `FillTiming.NEXT_OPEN` (default) or `SAME_CLOSE`.
- `Portfolio(initial_capital)`: positions, cash, `equity`, `apply_fill`,
  `mark_to_market(prices)`, `weights()`, immutable `Ledger` of fills.
- `Strategy` ABC (`strategy.py`): `on_bar(context) -> list[Order] | None`;
  `TargetWeightStrategy` converts target-weight frames or callables into
  orders with rebalance scheduling.
- `BacktestEngine(dataset, strategy, execution, initial_capital)`;
  `run() -> BacktestResult` with `equity_curve`, `returns`, `weights`,
  `positions`, `trades`, `costs_paid`, and `summary()` (delegates to
  analytics).

### 4.4 portfolio (complete)

- `covariance.py`: `sample_cov(returns)`, `ewma_cov(returns, halflife)`,
  `ledoit_wolf_cc(returns)` (shrinkage to constant-correlation target,
  returns matrix and intensity), all annualization-free (per-period).
- `optimizers.py`: common interface
  `Optimizer.weights(mu, cov, prev_weights=None) -> pd.Series`:
  `InverseVolatility`, `EqualRiskContribution` (cyclical coordinate
  descent), `MeanVariance(risk_aversion, constraints)`, `MaxSharpe`
  (tangency; returned gross exposure is `max_gross` if capped, else 1),
  `HierarchicalRiskParity(linkage_method)`.
  `VolatilityTargeting(optimizer, target, periods_per_year, max_leverage)`
  wraps any optimizer and scales to target ex-ante annualized vol.
  `mu` is per-period expected returns supplied by the caller; risk-based
  optimizers ignore it. Optimizers that invert the covariance raise
  `OptimizationError` on singular/ill-conditioned input rather than
  silently regularizing.
- `constraints.py`: `Constraints(long_only, max_weight, min_weight,
  max_gross, max_net)` with `project(weights)` (approximate clip+rescale,
  used after heuristic optimizers), `is_satisfied(weights)`, and SLSQP
  hooks `scipy_bounds(n)` / `scipy_constraints()` (exact, used by the
  solver-based optimizers).
- `rebalance.py`: `RebalancePolicy(band, max_turnover, partial)` with
  `apply(current, target) -> traded-to weights` (per-asset no-trade bands,
  pro-rata or all-or-nothing one-way turnover cap) and
  `turnover(current, target)`.

### 4.5 risk (complete)

- `metrics.py`: `annualized_vol`, `rolling_vol`, `value_at_risk`
  (`historical|normal|cornish_fisher`), `expected_shortfall` (same three
  methods), `beta`. VaR/ES are positive loss fractions.
- `drawdown.py`: `drawdown_series`, `max_drawdown`, `drawdown_episodes`
  (start/trough/recovery table, `NaT` recovery while underwater),
  `ulcer_index`.
- `exposures.py`: `gross_net_exposure(weights)`, `sector_exposures(weights,
  sectors)`, `factor_exposures(returns, factor_returns)` via OLS.
- `limits.py`: `RiskLimit` ABC with `check(state) -> list[LimitBreach]`;
  `GrossLeverageLimit`, `PositionLimit`, `DrawdownLimit`, `VolLimit`;
  `RiskMonitor(limits)` evaluating a `PortfolioState`.
- `report.py`: `RiskReport.from_backtest(result, ...)` -> frames + text
  summary.

### 4.6 ml (complete)

- `datasets.py`: `PanelDataset.from_factors(factors: dict[str, WideFrame],
  prices, horizon, return_method, feature_lag)` -> long panel with `X`,
  `y`, `groups` (dates), MultiIndex `(date, symbol)` rows. Targets come
  from `research.evaluation.forward_returns` (label at `t` spans
  `(t, t+h]`), so the standard alignment is causal; `feature_lag` adds
  extra lag for sensitivity checks.
- `validation.py`: `PurgedKFold(n_splits, horizon, embargo)` over date
  groups (training dates whose label windows overlap a test block are
  purged on both sides, plus an embargo after);
  `WalkForward(train_periods, test_periods, step, purge)` — strictly
  past -> future, `purge` defaults to the panel's horizon. Both yield
  `(train_rows, test_rows)` positions.
- `models.py`: `ModelWrapper` protocol (`fit`, `predict`,
  `feature_importances_` may return None); `LinearModel` (ridge),
  `TreeModel` (random forest; single-threaded by default for bit-exact
  determinism), `GradientBoostingModel` (lightgbm if available else
  sklearn, chosen at fit time), `EnsembleModel(models, weights)`.
  Wrappers validate prediction columns against training columns.
- `training.py`: `WalkForwardTrainer(model_factory, splitter).run(panel) ->
  TrainedSignal` with the OOS prediction wide frame, per-fold date
  ranges, and the fitted models. Only WalkForward output is causal
  enough to backtest; purged k-fold is for model selection.
- `evaluation.py`: `regression_metrics(y_true, y_pred)`,
  `prediction_rank_ic(predictions, prices, horizon)` (reuses the
  research layer's IC machinery), `permutation_importance(model, X, y)`.
- `persistence.py`: `ModelRegistry(root)`: `save(model, name, manifest)
  -> version`, `load(name, version=None) -> (model, manifest)`,
  `list()`; auto-incrementing `vNNNN` versions, manifest written last so
  crashed saves are ignored.

### 4.7 analytics

- `performance.py`: `periods_per_year(index)`, `cagr`, `annualized_vol`,
  `sharpe_ratio(rf)`, `sortino_ratio`, `calmar_ratio`, `hit_rate`,
  `skewness`, `kurtosis`, `turnover(weights)`,
  `performance_summary(returns, weights=None) -> pd.Series`,
  `rolling_sharpe(window)`.
- `attribution.py`: `asset_contributions(weights, asset_returns)`,
  `factor_attribution(returns, factor_returns)` (alpha + beta
  contributions with R^2).

### 4.8 viz

- `style.py` (`apply_style()`), `performance.py`
  (`plot_equity_and_drawdown`, `plot_rolling_stats`,
  `plot_monthly_heatmap`), `factors.py` (`plot_ic`, `plot_quantile_returns`,
  `plot_factor_correlation`), `portfolio.py` (`plot_weights`,
  `plot_contributions`), `tearsheet.py` (`tearsheet(result) -> Figure`).
  All functions accept and return matplotlib objects; nothing calls
  `plt.show()`.

## 5. Error handling

All framework exceptions derive from `QuantLabError` (see
`quantlab/exceptions.py`). Subsystems raise the narrowest type that applies;
user-facing messages state what failed and the first concrete offending
values. Cache corruption degrades to refetching, never to a crash.

## 6. Configuration and reproducibility

- `Settings` resolves defaults -> TOML file -> `QUANTLAB_*` env vars; tests
  use `Settings().with_root(tmp_path)`.
- Every stochastic component takes an explicit `seed` (falling back to
  `Settings.random_seed`), and the synthetic market is a pure function of
  `(seed, n_assets, n_sectors, inception, date)`.

## 7. Extension points

- New data vendor: subclass `DataSource`, honor the contract in
  `sources/base.py`, wrap in `CachedSource`. Nothing else changes.
- New factor: subclass `Factor`, implement `compute`; it immediately works
  with signal pipelines, evaluation, experiments, and the ML panel builder.
- New optimizer: implement `Optimizer.weights`; it plugs into
  `VolatilityTargeting`, rebalance policies, and backtest strategies.
- New cost model: subclass `SlippageModel`/`CommissionModel`; the execution
  simulator composes them.
