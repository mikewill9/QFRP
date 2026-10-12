# QFRP Project Handoff

## Project Overview

QFRP is a Python quantitative finance research framework.

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

- Checkpoint 11: gradient boosting challenger in the factor research notebook
- Checkpoint 10: ML combination wiring and general factor research notebook
- Checkpoint 9: point-in-time universe expansion (roadmap Stage 4)
- Checkpoint 8: multi-factor ML combination (roadmap Stage 3)
- Checkpoint 7: point-in-time fundamental data layer and fundamental factors
- Checkpoint 6: visualization and attribution
- Checkpoint 5: ML alpha pipeline
- Checkpoint 4: portfolio and risk layer
- Checkpoint 3: backtesting engine
- Checkpoint 2: research subsystem
- Checkpoint 1: initial quantlab framework

Latest commit (2026-10-11, branch bp-frontend):

d60a872 README: add app launch steps

Work after Checkpoint 11 is described in the "Post-Checkpoint 11" and
"Congressional App Challenge scope" sections below.

Between checkpoints 9 and 10 a research-experiments commit (b067be5)
ran factor testing on US_LARGE_CAP_149 (real Yahoo data) in
notebooks/momentum_research.ipynb.

Work since Checkpoint 11 (commits de38eb1..bcfd643, 2026-09-13 to
2026-10-04) was NOT committed as one checkpoint. It is a run of small
commits: the permutation significance test, the web app's data layer,
the shipped results, and the disclosure documents. The first two of
those commits are labelled "CP12" in git. It is all described below
under "Post-Checkpoint 11: Permutation Testing and the QFRP App".

The repository is the source of truth.

Do not rely on previous chat history.

## Environment

- Python 3.14 virtual environment at `.venv` (not committed)
- Install with: `pip install -e ".[dev]"`
- Validate with: `python -m pytest` (561 tests), `ruff check src tests examples`, `mypy`
  (`ruff check app scripts` is also clean; mypy covers 70 source files)
- The test suite, the examples and the app's data layer all run offline;
  no downloads needed. Tests use the synthetic market, and `app/api.py`
  only reads the JSON files committed in `results/permutations/`.
- `pyproject.toml` sets pytest `pythonpath = ["."]` so tests can
  `import app.api` (`app/` is not an installed package).
- Test suite runs matplotlib headless (Agg backend, wired in tests/conftest.py)
- Real-data caches (`~/.quantlab/cache*`) are machine-local and not in git.
  The 2015-2024 price cache for US_LARGE_CAP_149 lives on the primary
  (Windows) machine; a fresh clone re-downloads on first notebook run.
- The permutation runner is the one thing that needs real prices. It
  reads a separate cache fetched fresh over 2004-01-01..2024-12-31
  (`~/.quantlab/cache_2004_2024` on the Mac, selected with the
  `QUANTLAB_CACHE_DIR` environment variable) and refuses to run if that
  cache is incomplete. Regenerating results on another machine means
  building that cache first; the numbers may then differ slightly,
  because Yahoo re-adjusts historical prices over time.
- Machine-local and gitignored: `outputs/permutations/` (the runner's
  raw output, all 55 configs) and `results/permutations_hp/` (the
  20,000-shuffle rerun). Only `results/permutations/` is tracked.


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


## Checkpoint 10: ML Combination Wiring and General Factor Research Notebook

Completed: the Checkpoint 8 ML layer wired up for real-data experiments,
plus a new primary research notebook.

### Library (small, backward-compatible)

- WalkForward gained `mode="rolling" | "expanding"` (ml/validation.py).
  Rolling is the default and byte-for-byte the old behavior. Expanding
  pins the training window at the first date and grows it by `step` per
  fold (`train_periods` is the initial size); the purge gap to the test
  block is unchanged, so expanding only ever adds older data.
- `coefficient_history(signal)` (ml/training.py): per-fold linear
  coefficients, one row per fold indexed by its `test_start`, one column
  per feature — the honest OOS picture of how combination weights
  evolved. Raises ModelError on models without `coef_`.
- `FundamentalDataset.coverage_table(item, index, symbols)`
  (data/fundamentals.py): per-date `n_covered` / `n_universe` /
  `pct_covered`, so sparse fundamental data is reported explicitly —
  panels drop rows with any missing feature, and this table is how that
  shrinkage stays visible.
- 11 new tests (470 total), architecture.md sections 4.6/4.9 and README
  updated.

### Notebook: notebooks/factor_research.ipynb (new primary)

Config-driven replacement for the single-purpose momentum notebook
(momentum_research.ipynb is kept for reference, untouched):

- Universe US_LARGE_CAP_149 (US_LARGE_CAP_150 minus MMC, Yahoo data
  issue), configurable dates, rebalance frequency, and factor subset
  from a catalog of 11 price + 8 fundamental (CP7 value/quality/growth)
  factors; selection validated before any download.
- Mode "single": one-factor long/short costed backtest (the old
  workflow, generalized to any factor).
- Mode "ml": ranked forward-return target (`target_transform="rank"`)
  -> walk-forward Ridge (`LinearModel`) -> continuous OOS signal;
  outputs OOS rank IC and costed-backtest comparisons of the blend vs
  EVERY selected single-factor baseline on identical OOS dates, plus a
  Ridge weight-evolution chart saved as its own PNG.
- Fundamental honesty: statements fetched from START minus 2 years (TTM
  warmup), `strict=False` so uncovered names drop with a warning,
  per-item coverage change-point tables, per-feature coverage summary,
  and an explicit requested-vs-covered panel date-range printout.
- Conventions kept from the old notebook: RUN_NAME, cached data reuse,
  timestamped tearsheet PNGs under outputs/.

Design note, deliberate: **Ridge is the first rung of the model
ladder** — linear, regularized, interpretable, with weight evolution as
a first-class output. Gradient boosting and stacked ensembles are
planned future comparisons, to be run only after the Ridge baseline on
real data is understood (roadmap §3 discipline: one walk-forward per
finalized configuration, every run counted).

Validation: the notebook's exact cell logic was executed end-to-end on
the synthetic market in both modes before authoring; the ML sanity
result matched Checkpoint 8 (blend OOS rank IC below raw momentum's,
fundamentals contributing ~nothing — correct on synthetic data).

Known limits: real-data caches are machine-local (see Environment); a
fresh Yahoo fundamental fetch covers only ~4-6 trailing quarters, so
multi-year fundamental ML runs wait on cache accumulation (roadmap
Appendix A); every result on US_LARGE_CAP_149 carries the survivorship
caveat.


## Checkpoint 11: Gradient Boosting Challenger

Completed: the second rung of the model ladder — gradient boosting as a
config-selectable challenger to the Ridge baseline in
notebooks/factor_research.ipynb.

Scope note: the checkpoint was specified as a new
`research/ml/gradient_boosting.py` with a `GradientBoostingAlphaModel`
mirroring a `RidgeAlphaModel`. Neither existed — the repository (source
of truth) already had `GradientBoostingModel` in `src/quantlab/ml/models.py`,
with window modes living in `WalkForward` and purged CV in `PurgedKFold`,
both model-agnostic. The checkpoint was therefore implemented as minimal
extensions to the existing architecture rather than a parallel class.

### Library (small, backward-compatible)

- `GradientBoostingModel` gained `subsample: float = 1.0` (ml/models.py):
  values below 1 enable stochastic gradient boosting (each tree fits a
  random row fraction — regularization for low signal-to-noise panels).
  The default reproduces the pre-CP11 model exactly (pinned by test);
  out-of-range values raise ModelError. Passed to sklearn's
  GradientBoostingRegressor directly; on the optional lightgbm backend it
  maps to `subsample` + `subsample_freq=1`.
- `importance_history(signal)` (ml/training.py): the tree-model analogue
  of `coefficient_history`, same frame format (one row per fold indexed
  by `test_start`, one column per feature), so the notebook's
  weight-evolution chart reads either. Importances are non-negative and
  unsigned (no direction information; sklearn normalizes to sum 1 per
  fold); raises ModelError when a fold's model has no importances.
- 14 new tests in tests/test_gradient_boosting.py (484 total): subsample
  validation and backward-compat pins, determinism under subsampling,
  challenger-hyperparameter signal recovery with finite predictions and
  normalized importances, a hand-computed constant-target case, both
  walk-forward window modes with causality assertions, PurgedKFold
  compatibility (the purge machinery is structural and model-agnostic —
  the GBM flows through it unchanged), and the importance_history
  contract.

### Notebook: notebooks/factor_research.ipynb

- Step 1 config: `ML_MODEL = "ridge" | "gbm"` plus `GBM_PARAMS`
  (challenger defaults `n_estimators=100, max_depth=3,
  learning_rate=0.05, subsample=0.8` — shallow trees, slow learning,
  row bagging; all editable in place). Invalid values raise.
- Step 8 instantiates `LinearModel(alpha=RIDGE_ALPHA)` or
  `GradientBoostingModel(**GBM_PARAMS)` from the config; the
  walk-forward loop, purge gap, IC table, and baseline comparisons are
  untouched and identical across models.
- Step 10 plots `coefficient_history` (ridge, signed) or
  `importance_history` (gbm, unsigned, sums to 1) with matching labels.
- Output filenames now carry the model name in ml mode
  (`..._ml_ridge_...` / `..._ml_gbm_...`, and the weights chart
  `{RUN_NAME}_{ML_MODEL}_weights_...`); single mode is unchanged.

Validation: the notebook's exact cell logic was executed end-to-end on
the synthetic market for BOTH `ML_MODEL` settings before authoring;
identical fold structures, finite OOS signals, and normalized GBM
importances confirmed.

Discipline reminder (roadmap §3): run the challenger only on a
configuration whose Ridge baseline is already understood, and count
every walk-forward. Small OOS gains over Ridge are within noise.


## Post-Checkpoint 11: Permutation Testing and the QFRP App

Completed: a statistical test that asks "could this backtest result be
luck?", run for every price factor and date range, with the results
shipped in the repo and served to a web app through a small, fixed
interface.

The question the whole block answers: a backtest can look good by
chance. A permutation test measures how often. Keep the strategy
exactly the same, but randomly reassign the factor's scores among the
stocks, and re-run the backtest thousands of times. If the real result
is no better than those random versions, the factor's ranking carried
no information.

### Significance functions (src/quantlab/analytics/significance.py)

Hand-written by Michael, with their tests (tests/test_significance.py).
See "AI usage" below before editing them.

- `permute_within_date(zscores, rng)`: shuffles one date's scores
  across stocks. NaN entries stay where they are, so a stock with no
  score on that date still has no score.
- `permutation_pvalue(real_sharpe, null_sharpes) -> (p, n_dropped)`:
  one-sided p-value, the share of shuffled Sharpes at least as large as
  the real one, computed as (count + 1) / (n + 1) so it is never
  exactly zero. NaN shuffles are dropped and counted; a NaN real Sharpe
  or an all-NaN null raises.
- `bonferroni_adjust(p, k)`: `min(1.0, p * k)`. The multiple-testing
  correction: looking at k results makes a lucky one k times more
  likely, so the p-value is multiplied by k. Simple and deliberately
  strict.

### Shared backtest convention (src/quantlab/backtest/pipeline.py)

Extracted from notebooks/factor_research.ipynb so the notebook and the
scripts run the same code instead of drifting copies (the notebook now
imports from it). Documented in docs/architecture.md section 4.3.

- `TICKERS`: 148 names, `US_LARGE_CAP_150` minus MMC (Yahoo data issue)
  and EA (delisted; Yahoo no longer serves its history). Survivors only.
- `DATA_START = "2004-01-01"`: prices are always loaded from here so
  every factor is warmed up by 2005. Factors are computed on the full
  history and then trimmed to the window, so a given date gets the same
  score in every period.
- `PRICE_FACTOR_BUILDERS`: the 11 price factors (momentum, ts_momentum,
  st_reversal, price_to_ma, realized_vol, ewma_vol, market_beta,
  pca_reversal, amihud_illiquidity, dollar_volume, roll_spread).
- `normalization()`: per-date winsorize (1%/99%) then z-score.
- Portfolio: long the top 20%, short the bottom 20%, equal weight per
  side, gross exposure 1.0, rebalanced every 21 bars, decisions at the
  close and fills at the next open.
- Costs at "1x": 5 bp spread, $0.005/share commission ($1 minimum),
  square-root market impact (coefficient 0.1), at most 5% of a bar's
  volume.
- `make_execution(cost_multiplier, frictionless=False)` and
  `run_ls_backtest(signal, ds, cost_multiplier=1.0, frictionless=False)`.
  The multiplier scales spread and commission but NOT market impact
  (impact models the stock's liquidity, it is not a fee).
  `frictionless=True` removes every cost and keeps the participation cap.

### Permutation runner (scripts/run_permutations.py)

For each (factor, period) configuration it runs the real backtest with
no costs (gross) and at 1x/2x/3x costs, then N shuffled backtests with
no costs, and writes one JSON.

- Periods (`PERIODS`): 2005_2024 (full sample), 2005_2009, 2010_2014,
  2015_2019, 2020_2024. The full sample overlaps the other four, so the
  five are not independent tests.
- 11 factors x 5 periods = 55 configurations.
- The shuffle is applied only on rebalance dates, and those dates are
  read off `signal_to_weights`' own output rather than re-coded, so the
  shuffle hits exactly the rows the portfolio trades on.
- **Significance is tested gross, on both sides.** Shuffled portfolios
  trade about twice as much as the real one, so charging costs would
  push the random versions below zero and make any real factor look
  more significant than it is. Michael caught this bias in the first
  version. Whether an edge survives costs is a separate question,
  answered by the 1x/2x/3x cost stress on the real strategy.
- Seeds: each config gets its own `SeedSequence` from (base seed,
  SHA-256 of "factor|period"). Configs never share random draws and the
  same command reproduces the same nulls on any machine. Python's
  built-in `hash()` is not used because it changes per process.
- Safe to interrupt: finished configs are skipped on restart, files are
  written atomically, and an existing JSON made with a different N or
  base seed stops the run instead of being silently reused.
- Never downloads: `check_cache_complete()` exits if any ticker's cache
  is short of the data window. Stitching fresh Yahoo prices onto an old
  cache creates fake jumps, because the dividend adjustment basis differs.
- Flags: `--n` (default 1000), `--factor`, `--period`, `--base-seed`
  (default 20260927), `--out-dir` (default `outputs/permutations/`),
  `--workers` (default: all cores but 2), and `--config`.
- **`--config FACTOR:PERIOD`** (repeatable): run only the listed
  configurations. Added so a few borderline results could be rerun at a
  higher N into a separate directory. It cannot be combined with
  `--factor`/`--period`, rejects unknown names, and drops repeats while
  keeping order (tests/test_run_permutations_cli.py).

```
QUANTLAB_CACHE_DIR=~/.quantlab/cache_2004_2024 \
    python scripts/run_permutations.py --n 2000
```

Each JSON stores: the raw p-value, the gross Sharpe, every shuffled
gross Sharpe (unbinned), Sharpe at 1x/2x/3x, total return, max drawdown,
hit rate and turnover at 1x, the shuffles' average turnover, the 1x
equity curve, names with a score per year, every setting used, and
provenance (git commit, cache directory, timestamp, runtime).

### Shipped results (results/permutations/)

- 50 JSON files, `{factor}_{period}.json`, about 6.3 MB, tracked in git.
  They are unchanged copies of the runner's output for the 10 factors
  the app offers (the 5 `roll_spread` files are left out).
- Every file: N = 2,000 shuffles, base seed 20260927, 148 tickers, and
  zero shuffles dropped as NaN.
- Provenance: 45 files were generated at commit fa74751 and 5 (the
  2020_2024 files for momentum, ts_momentum, st_reversal, price_to_ma,
  realized_vol) at cd2de31. The commits between those two changed only
  documents, app/api.py and its tests, not any backtest code.
- `.gitignore` ignores `results/*` except `results/permutations/`.
- `tests/test_api.py::test_shipped_results_cover_every_menu_config`
  fails if a menu configuration has no shipped file.

Verdicts for a single test (k=1), computed from the shipped files
through app/api.py on 2026-10-04:

- 5 PASS: amihud_illiquidity in 2005_2024, 2005_2009, 2010_2014 and
  2015_2019, and price_to_ma in 2015_2019.
- 5 INCONCLUSIVE: st_reversal 2015_2019 and 2020_2024, price_to_ma
  2005_2024 and 2020_2024, pca_reversal 2005_2009.
- 40 FAIL, of which 9 are "reversed" (significantly WORSE than random):
  realized_vol (2005_2024, 2005_2009, 2015_2019), ewma_vol (2005_2024,
  2005_2009), market_beta (2005_2024) and dollar_volume (2005_2024,
  2005_2009, 2010_2014).
- Momentum fails in all five periods.
- If a user has viewed all 50 (k=50), only two results still PASS:
  amihud_illiquidity 2005_2024 and 2005_2009.

**The main finding is the Amihud result, and it is a caution, not a
discovery.** The illiquidity factor passes strongly in every period
except the most recent (gross Sharpe 1.63 in 2005-2009 falling to 0.22
in 2020-2024). The stock list was chosen in 2026, so a stock that was
illiquid in 2005 and is on the list had to survive and grow large. The
shuffle test checks for luck; it cannot check whether the universe was
fair. docs/amihud_survivorship.md explains this, and its takeaway line
is "Passing the test means the signal beat luck, not that the data was
fair."

### 20,000-shuffle rerun (docs/rerun_20k_comparison.md)

Purpose: a p-value estimated from 2,000 shuffles has its own sampling
error, so a result sitting near a cutoff could land on the other side
with different random draws. The borderline results were rerun with ten
times the shuffles and an independent seed to check.

- 14 configurations at N = 20,000, base seed 20261002, at commit
  bb11683, written to `results/permutations_hp/` (gitignored).
- scripts/compare_permutations.py builds the comparison table. It gets
  its verdicts from `app.api` (switching directories with
  `QFRP_RESULTS_DIR`), so cutoffs and the reversed rule are never
  re-implemented. It also checks the real gross Sharpe is identical in
  both files; if not, the backtest code changed and the runs are not
  comparable. `SELECTED_FOR` in that script records why each config was
  picked, decided from the N=2,000 results before any rerun.
- Result: **no verdict changed at k=1.** One changed at k=2 only:
  price_to_ma 2020_2024 goes from FAIL to INCONCLUSIVE (adjusted p
  0.2059 vs 0.1972, either side of the 0.20 cutoff).
- The largest gaps between the two estimates were z = -2.5 (st_reversal
  2015_2019) and z = -2.0 (ewma_vol 2010_2014), where z is the gap in
  combined Monte Carlo standard errors.
- docs/rerun_20k_comparison.md is the committed evidence. The 20k JSONs
  themselves are not in git and exist only on the Mac.
- **The app still serves the 2,000-shuffle files.** The rerun confirms
  them; it does not replace them.
- Not rerun: market_beta 2005_2024 (about 11 hours; noted in
  LIMITATIONS.md) and amihud_illiquidity 2005_2024 (listed in
  `SELECTED_FOR` but no rerun file exists).
- Cost of a rerun: a 5-year config at 20,000 shuffles took about 3 hours.

### App data layer (app/api.py)

The only code the web app calls. It reads the precomputed JSONs and
never runs a backtest, so every call is instant. Nothing in it is made up.

- `list_factors()`: the 10-factor menu, taken from
  `PRICE_FACTOR_BUILDERS` minus `EXCLUDED_FACTORS`.
- `list_periods()`: the 5 periods, read from `PERIODS` in the runner
  script (loaded with importlib because `scripts/` is not a package).
- `get_analysis(factor_key, period_key, k=1) -> AnalysisResult`: net
  performance at 1x costs, the equity curve, the gross significance
  numbers, cost stress, breadth per year, and limitation strings to
  display verbatim.
- `verdict_for(result, n_configs_viewed) -> Verdict`: PASS when the
  Bonferroni-adjusted p is below 0.05, FAIL above 0.20, INCONCLUSIVE in
  between (`PASS_BELOW`, `FAIL_ABOVE`).

Design decisions worth knowing:

- **The verdict is computed at display time, not stored**, because the
  adjustment depends on how many configurations the user has viewed
  this session.
- **Reversed FAIL.** The test is one-sided, so a strategy far worse
  than random would also read as a plain FAIL ("no signal"). Every
  verdict therefore runs the test in the other direction too. If that
  passes, the label stays FAIL but `reversed` is True and the headline
  is "Consistently worse than random".
- **`roll_spread` is hidden** (`EXCLUDED_FACTORS`). Its estimator
  clamps at zero, so about 54 of 148 stocks tie for the best score and
  the sort's tie-breaking, not the factor, picks the long leg. That
  biases the null distribution and makes its p-values meaningless. It
  stays in the catalog and the runner still computes it.
- **`SURVIVORSHIP_NOTE`** is appended to every PASS explanation.
- Two Sharpes, two questions: `sharpe_gross` is what the p-value tests;
  `sharpe_1x` is whether it was worth trading.
- An import-time check raises if the menu text drifts out of sync with
  the factor catalog.
- Errors: unknown or excluded key raises `KeyError`; a valid key with
  no file raises `ResultNotComputedError` (a `KeyError` subclass); bad
  `k` raises `ValueError` or `TypeError`.
- `QFRP_RESULTS_DIR` overrides the results directory (used by the tests
  and by compare_permutations.py).

### Documents added

- **CONTRACT.md** (v2): the boundary between the two team members.
  Michael owns everything behind `app/api.py` (`src/quantlab/`,
  `scripts/`, `results/`); BP owns the frontend in `app/ui/`. BP calls
  the four functions and reads named fields, and never imports
  `quantlab` directly. Changes to CONTRACT.md need both to agree. It
  defines every field, the verdict bands, error-to-HTTP mapping and
  deployment needs.
- **LIMITATIONS.md**: the honest list. Data (survivorship bias, the
  universe growing from 126 to 148 names, price factors only, known
  Yahoo errors such as DHR on 2016-07-05), method (gross significance,
  the shuffle design, Bonferroni being stricter than needed because
  periods overlap, minimum p of about 0.0005, not every borderline
  config rerun) and app (the viewed-count resets on page refresh; no
  short borrow costs). The "hold" shuffle comparison it mentions was a
  one-off check and is not reproducible from the repo.
- **AI_USAGE.md**: who wrote what. See "AI usage" below.
- **docs/amihud_survivorship.md**: the explainer for the main finding.
- **docs/rerun_20k_comparison.md**: the rerun tables.
- **README.md**: gained a "Quick start" section, a "Running the app"
  section with the launch steps, a short "Permutation test results"
  paragraph and a "Libraries and tools" section. Its opening sections
  still introduce the `quantlab` library rather than the app.

Tests added: 77 (561 total) across tests/test_significance.py,
tests/test_backtest_pipeline.py, tests/test_api.py and
tests/test_run_permutations_cli.py.

### AI usage (AI_USAGE.md)

The project discloses how AI was used, and commit messages are tagged
to match ("hand-written", "Claude Code", "drafted with Claude chat").

- Hand-written by Michael: `permute_within_date`, `permutation_pvalue`,
  `bonferroni_adjust`, and their tests.
- AI-assisted (Claude Code, reviewed by Michael): pipeline.py,
  run_permutations.py including `--config`, compare_permutations.py,
  app/api.py, CONTRACT.md, the data extension to 2005 and EA removal,
  shipping the results.
- Drafted with Claude chat, edited and approved by Michael: verdict
  text, the survivorship note, LIMITATIONS.md, parts of the Amihud
  explainer.
- Methodology decisions (gross significance test, Bonferroni, the
  cutoffs, dropping roll_spread, featuring the survivorship result)
  were made by Michael.

Any AI session that changes code must keep this accurate: update
AI_USAGE.md in the same change, and do not rewrite the three
hand-written functions or their tests without Michael asking for it,
because that would make the disclosure untrue.


# Current Development Stage

The current focus is the QFRP web app, the team's entry for the
Congressional App Challenge. This replaces "no active checkpoint".

## Congressional App Challenge scope

Source note: the repository never names the competition. That the app
is for the Congressional App Challenge comes from Michael (2026-10-04).
Target submission is Oct 22, 2026; the competition deadline is Oct 26,
2026. A demo video of 1 to 3 minutes is planned and is the most
important deliverable. Other submission and judging rules are not
recorded in the repo; ask Michael rather than assuming them.

What the entry is, as built:

- A web app that lets someone pick a well-known trading factor and a
  date range, see its backtest, and then see whether the result beats
  luck. The lesson it teaches is that a good-looking backtest is easy
  to find by trying many options, and that the honest p-value has to
  account for how many were tried.
- Team: Michael (research, statistics, `app/api.py`) and BP (React
  frontend plus a FastAPI bridge in `app/ui/`).
- Fixed menu: 10 price factors x 5 periods = 50 precomputed
  configurations. A session counter of configurations viewed drives the
  Bonferroni adjustment and the verdict.
- Featured result: Amihud illiquidity passing the test while being
  explained by survivorship bias.

In scope:

- keeping `app/api.py`, CONTRACT.md and the shipped results consistent
- wording: verdicts, limitations, the survivorship explainer
- supporting BP's integration
- keeping AI_USAGE.md and LIMITATIONS.md accurate

Out of scope for the entry (do not add without Michael asking):

- letting users run arbitrary backtests or choose their own settings
- fundamental factors (free data has no real history before about 2025)
- the ML combination work (Checkpoints 8, 10, 11); the app shows
  single factors only
- new factors, new periods, or regenerating the shipped results. Any of
  these changes the 50 files and invalidates the rerun comparison.
- anything that reads as trading advice

Open items:

- The frontend is not on master yet. It lives on `origin/bp-frontend`
  (`app/ui/server.py`, `app/ui/frontend/`). As of 2026-10-11, master
  has been merged into `bp-frontend`, so that branch is ahead of master
  and is missing nothing from it. Merging `bp-frontend` back into
  master is still to do before submission.
- Benchmark curve: v1 had an equal-weight long-only basket; it is not
  precomputed. Either add a precompute step or drop it from the UI.
- Deployment: the server needs `pip install -e .` from the repo root;
  the result files ship with the checkout. No deployment is recorded
  in the repo.
- The viewed-count resets on page refresh (LIMITATIONS.md).

## Research framework

The architecture.md contract (sections 4.1 through 4.9: data, research,
backtest, portfolio, risk, ml, analytics, viz, fundamentals) is fully
built, tested, and documented.

Real-data factor experiments (single or ML-combined) run through
notebooks/factor_research.ipynb with a config-selectable model
(`ML_MODEL`: Ridge baseline or gradient boosting challenger, Checkpoint
11); the next model rung to compare there is stacked ensembles.

docs/research_roadmap.md is the plan of record for research work after
the app:

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
- for anything touching the app: CONTRACT.md, LIMITATIONS.md, AI_USAGE.md

3. Do not redesign the architecture without a strong reason.

4. Preserve existing interfaces. The names and types in CONTRACT.md are
   an agreement with BP: do not change them, or CONTRACT.md itself,
   without Michael and BP both agreeing. Do not edit `app/ui/`.

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
- if a change could alter any backtest number, the shipped results in
  `results/permutations/` no longer match the code: say so, and do not
  regenerate them without Michael asking
- update AI_USAGE.md when AI-written code is added or changed
