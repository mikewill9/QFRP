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
- **Event-dated data (fundamentals) carry two dates**: the fiscal
  `period_end` and the `publication_date` on which the observation became
  public. All alignment onto price indexes goes through publication dates
  (`pit_align`), so a statement is invisible until filed. Where true filing
  dates are unavailable (yfinance), sources apply a conservative assumed
  lag (default 45 calendar days after period end) and stamp
  `metadata["publication_basis"] = "assumed_lag"`.

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
- Universe (`universe.py`): `rolling_universe(dataset, *, top_n, window,
  min_price, rebalance_every) -> bool WideFrame` — point-in-time membership
  by construction (every rank uses trailing dollar volume through the
  rebalance bar only); `apply_universe(frame, universe)` NaNs signals
  outside membership without renormalizing; `US_LARGE_CAP_150` /
  `US_LARGE_CAP_SECTORS` static seed list — survivors only, so results on
  it overstate historical returns (roadmap Appendix A).

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
  prices, horizon, return_method, feature_lag, target_transform)` -> long
  panel with `X`, `y`, `groups` (dates), MultiIndex `(date, symbol)` rows.
  Targets come from `research.evaluation.forward_returns` (label at `t`
  spans `(t, t+h]`), so the standard alignment is causal; `feature_lag`
  adds extra lag for sensitivity checks. `target_transform` reshapes the
  label per date: `"raw"` (default, backward compatible), `"demeaned"`
  (relative return — recommended for cross-sectional research), `"rank"`
  (centered percentile ranks in `(-0.5, 0.5)`).
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
- `features.py`: `FeatureSpec(name, factor, transform=None)` and
  `build_features(dataset, specs) -> dict[str, WideFrame]` — computes
  each factor and applies its optional normalization, producing the
  mapping `from_factors` consumes; validates non-empty specs and unique
  names, preserves spec order.
- `selection.py`: `compare_models(panel, model_factories, splitter) ->
  DataFrame` (`mean_r2`, `std_r2`, `mean_mse`, `mean_rank_ic`,
  `n_folds`; one row per model on identical folds, sorted by rank IC).
  Selection only — its scores are interior-fold numbers; walk-forward
  remains the only causal output.

### 4.7 analytics (complete)

- `performance.py`: `periods_per_year(index)`, `cagr`, `annualized_vol`,
  `sharpe_ratio(rf)`, `sortino_ratio`, `calmar_ratio`, `hit_rate`,
  `skewness`, `kurtosis`, `turnover(weights)`,
  `performance_summary(returns, weights=None) -> pd.Series`,
  `rolling_sharpe(window)`.
- `attribution.py`: `asset_contributions(weights, asset_returns)` (prior-bar
  weight x realized return, per asset per bar); `factor_attribution(returns,
  factor_returns) -> FactorAttribution` (OLS alpha + betas + R^2, plus a
  per-period `contributions` frame — alpha, one column per factor, and
  residual — that sums exactly back to the input returns). Both use the
  standard linear-attribution approximation (no compounding cross-terms,
  no costs); collinear factors raise rather than returning an arbitrary
  least-squares solution.

### 4.8 viz (complete)

- `style.py` (`apply_style()`, `PALETTE`), `performance.py`
  (`plot_equity_and_drawdown`, `plot_rolling_stats`,
  `plot_monthly_heatmap`), `factors.py` (`plot_ic`, `plot_quantile_returns`,
  `plot_factor_correlation`), `portfolio.py` (`plot_weights`,
  `plot_contributions`), `tearsheet.py`
  (`tearsheet(result, risk_report=None) -> Figure`; `risk_report` is an
  optional `RiskReport` that adds a risk-metrics section to the summary
  text without being a hard dependency of the signature).
  Every function accepts an optional `ax`/`axes` and returns a `Figure`;
  a function only calls `fig.tight_layout()` when it created its own
  figure, so panels compose cleanly inside a caller's `GridSpec` (as
  `tearsheet` does). Nothing calls `plt.show()`.

### 4.9 fundamentals (complete)

Point-in-time fundamental data and the factors built on it. The governing
constraint is that a statement is usable only from its publication date,
never from its period end (see §3).

- **Statement frame** (container storage): long frame, one row per
  `(symbol, period_end, publication_date)` with one numeric column per
  item. Multiple rows sharing `(symbol, period_end)` are restatement
  versions; the newest publication is the current view. Canonical items
  (`FUNDAMENTAL_ITEMS`): `revenue`, `gross_profit`, `net_income`,
  `total_assets`, `total_liabilities`, `total_equity`,
  `shares_outstanding`, `operating_cash_flow`, `capital_expenditure`
  (as-reported sign: negative = outflow, so FCF = OCF + capex). The
  container accepts any additional numeric items (e.g. estimate series).
- **Event frame** (alignment currency): columns
  `["symbol", "period_end", "publication_date", "value"]` — one item's
  observations, produced by `FundamentalDataset.quarterly/ttm` and consumed
  by the alignment kernels.
- `FundamentalDataset` (frozen dataclass): fields `statements`, `metadata`;
  construction normalizes (dates parsed, symbols cast to str, rows sorted)
  and raises on structural defects (`publication_date < period_end`,
  duplicates, tz-aware dates, non-numeric items). Key methods:
  `as_of(date)` (rows published on or before `date`; boundary inclusive),
  `select`, `quarterly(item)` (all versions kept), `ttm(item, quarters=4)`,
  `pit_panel(item, index, symbols, ttm=False)`, `with_metadata`, `equals`.
- Alignment kernels (module functions in `data.fundamentals`):
  - `pit_align(events, index, symbols) -> WideFrame`: value at `(t, sym)`
    comes from the row with the greatest `period_end` among rows published
    on or before `t` (ties/restatements resolve to the latest publication;
    a late-restated old quarter never displaces a newer public quarter).
    NaN before the first publication, never backfilled.
  - `trailing_stat(events, quarters, stat="sum"|"std")`: rolling per-symbol
    stats over consecutive fiscal quarters; usable at the window's **max**
    publication date; windows spanning missing quarters are dropped, and
    restatement versions collapse to each quarter's latest publication
    (conservative, never lookahead).
  - `trailing_change(events, quarters)`: relative change vs `quarters` ago
    with an absolute-value denominator (sign-stable through negative
    bases; zero bases NaN).
  - `combine_events(left, right, func)`: row-wise combination on
    `(symbol, period_end)`, usable at the later publication.
- `FundamentalSource.fetch(symbols, start, end) -> FundamentalDataset`
  (`start`/`end` filter `period_end`; callers fetch 6-8 quarters of history
  before the first evaluation date). Implementations:
  - `SyntheticFundamentalSource(n_assets, seed, inception,
    publication_lag_days=45, publication_jitter_days=10)`: deterministic,
    window-independent, internally consistent statements (accounting
    identity exact, equity evolves by retained earnings, margins bounded);
    symbol naming matches `SyntheticSource`. Independent of synthetic
    prices, so statement-only factors have zero IC by construction (yield
    ratios inherit a mild anti-momentum tilt from their price denominator).
  - `YahooFundamentalSource(publication_lag_days=45, strict=True)`: behind
    the `data` extra, imported inside `fetch`. Assumed publication lag
    (Yahoo has no filing dates), trailing ~4-6 quarters only, silent vendor
    restatements — all documented in the class docstring.
- `CachedFundamentalSource(source, cache_dir)`: per-symbol parquet + covered
  range sidecars under `{cache_dir}/{source.name}/quarterly/`, atomic
  writes, corruption degrades to refetch. Merging **accumulates** quarters
  that a snapshot vendor no longer serves — the honest way to build
  point-in-time history going forward. `cached_range`, `clear`.
- Validation (in `data.validation`, same issue/report machinery):
  `FundamentalValidationCheck` ABC and
  `FundamentalDataValidator(checks).validate(fds) -> ValidationReport` with
  `default_fundamental_checks()`: `MissingStatementCheck` (quarter gaps,
  stale coverage), `NonPositiveFundamentalCheck` (shares/assets <= 0,
  negative revenue: ERROR), `AccountingIdentityCheck` (A = L + E within
  tolerance), `StaleFundamentalValueCheck` (identical flow values across
  quarters), `RestatementCheck` (multi-version detection; INFO/WARNING).
- Factors (`research.factors.fundamental`), all standard `Factor`
  subclasses taking the `FundamentalDataset` at construction with
  `min_history = 1` (warmup is gated by publication dates, not bar count;
  uncovered symbols are NaN columns, zero overlap raises):
  - value: `EarningsYield` (TTM NI / mcap), `BookToMarket` (equity / mcap;
    negative equity scores low, not NaN), `FCFYield` (TTM (OCF + capex) /
    mcap); market cap uses point-in-time shares outstanding;
  - quality: `ReturnOnEquity` (TTM NI / equity; NaN on non-positive
    equity), `GrossMarginStability(quarters=8)` (negative trailing std of
    gross margin), `Accruals` (negative TTM (NI - OCF) / assets — low
    accruals rank high);
  - growth: `RevenueGrowth(ttm=True)`, `EarningsGrowth(ttm=True)` (YoY,
    absolute-value base; `ttm=False` compares single quarters and needs
    only 5 quarters of vendor history);
  - `EstimateRevision(estimates, estimate_item, window)`: interface only
    against free data — raises `DataError` unless given a genuinely
    point-in-time estimates panel (rows with `period_end ==
    publication_date ==` snapshot date and a rolled consensus item), since
    yfinance exposes only current estimate snapshots.

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
- New fundamental vendor: subclass `FundamentalSource`, emit a statement
  frame with real filing dates where available
  (`publication_basis="reported"`), wrap in `CachedFundamentalSource`.
  A point-in-time estimates vendor activates `EstimateRevision` by
  supplying an estimates panel — no framework changes required.
- New factor: subclass `Factor`, implement `compute`; it immediately works
  with signal pipelines, evaluation, experiments, and the ML panel builder.
- New optimizer: implement `Optimizer.weights`; it plugs into
  `VolatilityTargeting`, rebalance policies, and backtest strategies.
- New cost model: subclass `SlippageModel`/`CommissionModel`; the execution
  simulator composes them.
