# quantlab research roadmap

From price-only factors toward a forward-looking multi-signal research
system.

This document does three jobs:

1. It states honestly what the platform can and cannot do today.
2. It lays out a staged progression — each stage earns its complexity by
   adding an *information source*, not just a technique — with the data it
   requires, where it plugs into the architecture, and the overfitting
   risks it introduces.
3. It carries mechanical implementation specs for the next two stages
   (Appendices B and C), written so a capable engineer — human or model —
   can execute them without re-deriving the architecture. Read
   `docs/architecture.md` first; it pins every interface referenced here.

A principle repeated throughout because it is the one most often violated:
**the ML layer predicts cross-sectional relative returns — which names do
better than which other names over the next horizon — never absolute
prices or index levels.** Absolute-level prediction on a handful of years
of daily data is a statistical dead end; relative ranking across a breadth
of names is where a small, honest edge can survive.

---

## 1. What the current system can and cannot do

### What it can do

The platform is a complete, tested research loop: deterministic synthetic
or cached Yahoo market data → factor computation → signal transforms →
IC/quantile evaluation → supervised panels → purged/walk-forward ML →
event-driven backtests with realistic costs → portfolio construction, risk
monitoring, attribution, and tearsheets. No-lookahead is structural
(signals at the close, fills at the next open, purge gaps sized to the
label horizon, publication-dated fundamentals), not a convention users
must remember.

As of Checkpoint 7 it has two information sources:

- **Prices and volume** (Checkpoints 1-6): momentum, reversal, volatility,
  liquidity, and statistical (PCA) factors.
- **Quarterly fundamental statements** (Checkpoint 7): value, quality, and
  growth factors on a point-in-time statement panel with publication-date
  alignment.

### What it cannot do — and why that matters

Be clear-eyed about what price-only factors are. They exploit
**short-horizon statistical regularities in the price process itself** —
autocorrelation (momentum), mean reversion, volatility clustering,
liquidity premia. These patterns are real but:

- they are **backward-looking by construction**: every input is a
  transformation of past prices, so the "forecast" is a bet that a
  historical regularity persists a little longer;
- they say **nothing about the business**: a price factor cannot tell a
  company growing earnings 30% a year from one in terminal decline if
  their charts happen to rhyme;
- they cannot capture **earnings quality** (are reported profits backed by
  cash?), **competitive dynamics** (is the margin structure defensible?),
  or **long-term fundamentals** (does the balance sheet survive a
  downturn?) — the things that determine multi-year returns;
- they are the **most crowded** signals in existence; whatever premium
  remains is small, decays fast, and is fought over by faster
  participants.

Checkpoint 7's fundamental factors begin to address this, with hard
limits inherited from the free data (Appendix A): ~4-6 quarters of
statement history from Yahoo, assumed publication dates, silent
restatements, and a survivorship-biased symbol universe. And fundamentals
as implemented are still *reported history* — genuinely forward-looking
inputs (estimate revisions, guidance language, positioning) arrive in
later stages, each demanding stricter validation than the last.

The platform also currently researches ~10-40 name universes. Cross-
sectional signals are statistically starved at that breadth: a 5-quantile
portfolio on 30 names holds 6 per bucket, so single-stock noise dominates
and IC estimates carry enormous error bars.

---

## 2. The staged progression

| Stage | Adds | Status |
|---|---|---|
| 1 | Price/volume factors | done (CP 2) |
| 2 | Fundamental factors, point-in-time statements | done (CP 7) |
| 3 | Multi-factor ML combination | next — spec in Appendix B |
| 4 | Universe expansion to 100+ names | spec in Appendix C |
| 5 | Sentiment / NLP | outlined below |
| 6 | Macro regime overlays for position sizing | outlined below |

Stages 3 and 4 are deliberately ordered before 5 and 6: they multiply the
statistical power of signals that already exist, whereas 5 and 6 add new,
noisier data that only pays off once the evaluation machinery has the
breadth to judge it.

### Stage 2 — fundamental factors (complete, Checkpoint 7)

**What it added.** A point-in-time fundamental data layer
(`quantlab.data.fundamentals`) where every observation carries its fiscal
`period_end` *and* the `publication_date` it became usable; alignment
kernels (`pit_align`, `trailing_stat`, `trailing_change`,
`combine_events`) that make lookahead structurally impossible; value
(earnings yield, book-to-market, FCF yield), quality (ROE, gross-margin
stability, accruals) and growth (YoY revenue/earnings) factors; a
deterministic synthetic statement generator; fundamental data-quality
checks; and caching that *accumulates* history from snapshot vendors.

**What was deliberately cut.** An estimate-revision factor against
yfinance data. Yahoo exposes only the *current* snapshot of analyst
estimates; backtesting requires what the consensus was on each historical
date, and feeding today's estimates into the past is lookahead. The
`EstimateRevision` interface exists, is tested against hand-built
point-in-time panels, and raises `DataError` until a genuine PIT estimates
source is connected.

**Honest limitations.** With Yahoo data: ~4-6 quarters of history (no
multi-year fundamental backtest until the cache accumulates or a paid
source arrives), assumed 45-day publication lag (optimistic for fiscal Q4,
which arrives via the 10-K — use 60+ days for stricter tests),
restatements invisible, universe survivorship-biased. The synthetic source
exists precisely so the *machinery* is fully tested despite all this.

### Stage 3 — multi-factor ML combination

**What it adds.** Today each factor is evaluated alone. Stage 3 feeds
price *and* fundamental factors jointly into the existing Checkpoint 5
pipeline (`PanelDataset` → `WalkForwardTrainer`) so the model learns the
weighting — including interactions a fixed-weight blend cannot express
(e.g. momentum that is only trustworthy when accruals are low).

**Data required.** None new. That is the point: Stage 3 multiplies the
value of data already flowing.

**Where it plugs in.** `quantlab.ml` gains a thin feature-assembly module
(`ml/features.py`) and a model-comparison helper (`ml/selection.py`);
`PanelDataset.from_factors` gains a `target_transform` option. Everything
else — splitters, trainers, models, registry, evaluation — already exists.
Exact spec: Appendix B.

**What the ML layer predicts.** Cross-sectionally **demeaned** (or
rank-transformed) forward returns over `(t, t+h]`, i.e. relative
attractiveness within the universe on each date. Never absolute prices,
never raw index-level returns: subtracting the per-date cross-sectional
mean removes the market component the model has no business fitting.

**Overfitting risks introduced.**

- *Feature multiplicity*: every added factor multiplies the hypothesis
  space; with ~10 features and monthly labels on a few years of data, an
  unregularized learner will fit noise with certainty. Mitigations: ridge
  and shallow GBMs only, feature counts in the low tens, purged CV for
  every selection decision.
- *Hyperparameter leakage*: tuning on walk-forward OOS results and
  re-running is training on the test set. Model/hyperparameter selection
  happens **inside** purged CV; walk-forward output is produced once per
  finalized configuration.
- *Scale leakage*: features must be cross-sectionally normalized per date
  (the existing `ZScore`/`Rank` transforms) so no feature smuggles in a
  market-level signal.

### Stage 4 — universe expansion to 100+ names

**What it adds.** Statistical breadth. Fundamental and ML signals are
cross-sectional bets; their information ratio scales roughly with the
square root of independent breadth. Moving from 30 to 150 names takes a
5-quantile portfolio from 6 to 30 names per bucket, shrinks IC error bars,
makes sector neutralization meaningful (5+ names per sector), and makes
capacity/cost modeling honest.

**Data required.** Daily OHLCV for ~150 liquid US names (yfinance through
the existing cache is adequate); fundamentals for the same universe
(accumulating via `CachedFundamentalSource`); a static seed list of
tickers and sectors, checked into the repo with its survivorship caveat
documented.

**Where it plugs in.** A new `quantlab.data.universe` module producing
**point-in-time membership masks** (top-N by trailing dollar volume, as of
each rebalance date) plus a helper that NaNs signals outside the
membership. Factors and pipelines are unchanged — breadth flows through
the existing wide-frame convention. Exact spec: Appendix C.

**What the ML layer predicts.** Unchanged (relative returns), now
estimated on 3-5x the cross-section, which is the cheapest robustness
upgrade available to the whole platform.

**Overfitting risks introduced.**

- *Survivorship bias becomes the dominant data risk*: a ticker list drawn
  from today's large caps contains only winners; historical backtests on
  it overstate returns (delisted losers are simply absent from Yahoo).
  This must be stated on every result until a survivorship-free source
  (Appendix A) arrives. Membership *timing* is handled point-in-time by
  the universe mask; membership *composition* cannot be fixed with free
  data.
- *Silent universe drift*: if membership at date `t` uses volume from
  after `t`, a subtle lookahead enters every downstream signal — the spec
  mandates a dedicated PIT test mirroring the momentum-skip test.
- *Liquidity illusion*: 150 names include thinner ones; realistic cost
  and participation settings matter more, not less.

### Stage 5 — sentiment / NLP

**What it adds.** A genuinely forward-looking, non-price information
source: news flow, filing language (10-K/10-Q risk sections, 8-K events),
earnings-call tone. These lead or explain fundamental changes weeks before
statements print.

**Data required.** This stage is *data-bound, not model-bound*. Honest
options: point-in-time news with per-article timestamps and ticker tags
(Tiingo News, Polygon News), SEC EDGAR full-text filings (free, with true
filing timestamps), GDELT (free, noisy). Scored either by simple lexicons
(Loughran-McDonald, free and auditable) or a small transformer (FinBERT)
run locally. **Every article/filing must carry the timestamp it became
public**, and scores aggregate into daily per-symbol series exactly like
publication-dated fundamentals.

**Where it plugs in.** The `FundamentalDataset`/event-frame machinery from
Checkpoint 7 generalizes directly: a sentiment observation is an event
with a publication timestamp; `pit_align` already implements the
alignment. A `SentimentSource` would mirror `FundamentalSource`; sentiment
factors subclass `Factor` like everything else and feed Stage 3's panels
as additional columns.

**What the ML layer predicts.** Unchanged. Sentiment enters as features
(level, change, abnormal volume of coverage), never as a separate
prediction target.

**Overfitting risks introduced.** The most dangerous stage so far:

- *Text pipelines are lookahead machines*: articles get re-timestamped,
  corrected, and back-filled by vendors; filing text is amended. Require
  vendor-native publish timestamps; when in doubt, lag by a full day.
- *Researcher degrees of freedom explode*: tokenization, lexicon choice,
  aggregation window, decay, entity matching — each is a knob, and knobs
  multiply false discoveries. Pre-register one construction per idea in
  the experiment tracker; evaluate once.
- *Regime fragility*: language usage drifts (COVID-era vocabularies broke
  many trained-sentiment models). Walk-forward evaluation with era
  breakdowns is mandatory, not optional.

### Stage 6 — macro regime overlays for position sizing

**What it adds.** Not stock selection — *risk conditioning*. Trend, rates,
credit spreads, and volatility regimes modulate **gross exposure and
factor tilts** (e.g. halve gross when the vol regime is stressed; damp
momentum after vol spikes, when momentum crashes historically cluster).

**Data required.** A handful of macro series: index level/vol (VIX or
realized proxy from the existing data), rates and spreads (FRED — use
**ALFRED vintages** for point-in-time values, since macro series are
heavily revised), possibly futures-based trend. Tiny data, huge revision
hazard.

**Where it plugs in.** Above signal generation, below execution: a regime
classifier produces a small state series; a sizing policy maps state →
target gross / vol target, implemented against the existing
`VolatilityTargeting` wrapper and `RebalancePolicy`. It composes with any
strategy; it never touches per-name signals directly.

**What the ML layer predicts.** Ideally nothing learned at all: 2-3
hand-specified regimes with hysteresis (e.g. realized vol above/below a
long percentile) beat fitted classifiers at this sample size. If anything
is learned, it predicts *portfolio risk states*, never returns.

**Overfitting risks introduced.** The worst per-datapoint risk in the
whole roadmap: daily data spans maybe 3-4 genuine regime cycles, so a
fitted regime model has effectively n≈4. Discipline: parameters chosen a
priori from published research, evaluated walk-forward, and the overlay
must improve *risk* (drawdown, vol stability) — treat any headline-return
improvement as suspect. Macro revisions are lookahead unless vintage data
(ALFRED) is used.

---

## 3. Validation doctrine

The machinery exists (purged k-fold with embargo, walk-forward with purge
gaps, HAC t-statistics, experiment tracking). Doctrine is about how its
use must *tighten* as features multiply.

### The rules

1. **Two-tier evaluation, never blended.** `PurgedKFold` is for model and
   hyperparameter *selection*; only `WalkForward` output may be called a
   result, quoted, or backtested. A purged-CV score is an internal number.
2. **One walk-forward per finalized configuration.** If you look at the
   walk-forward result and then change anything — a feature, a window, a
   hyperparameter — the next walk-forward is contaminated by that
   knowledge. Count how many times this loop runs per idea; more than 2-3
   means the "out-of-sample" label is fiction. Log every run in
   `ExperimentTracker` so the count is auditable.
3. **Selection lives inside the training window.** When hyperparameters
   must adapt over time, select them with purged CV *inside each
   walk-forward training window* (nested validation), never on the test
   block.
4. **Purge and embargo scale with the label.** The purge gap must be at
   least the label horizon (the default), and the embargo should grow when
   features are slow-moving: quarterly fundamentals make adjacent months
   informationally correlated far beyond the label window. For fundamental
   features, an embargo of ~21 bars on top of the horizon purge is a sane
   default.
5. **A held-out era stays sealed.** Reserve the most recent 1-2 years.
   Nothing — no selection, no eyeballing — touches it until a strategy is
   final; it is spent the moment it is used. Everything else can be
   iterated on.
6. **Feature budgets, stated in advance.** With ~1,500 daily observations
   and 21-bar labels there are ~70 non-overlapping label periods per name;
   breadth (Stage 4) is what buys statistical power, not longer feature
   lists. Keep models at ridge/shallow-GBM complexity and feature counts
   in the low tens; every feature admitted must have a written hypothesis
   *before* evaluation.
7. **Multiplicity is accounted, not ignored.** Every evaluated variant —
   including the discarded ones — raises the bar for the survivor. A
   t-statistic of 2 chosen from 20 silent trials is noise. The experiment
   tracker is the count of record; a Bonferroni-style mental discount
   (require t ≳ 3 when tens of variants were tried) is crude but honest.

### Red flags that a signal is fitted noise

Any of these should stop a promotion to "result":

- **Fragile to one bar of delay**: IC collapses when the signal is lagged
  by a single day (`Lag(1)`) or fills move from `SAME_CLOSE` to
  `NEXT_OPEN`. Real information survives a day; leakage does not.
- **Purged-CV ≫ walk-forward**: a large gap between interior-fold scores
  and strictly-causal scores is the signature of temporal leakage or
  regime fit.
- **Concentration**: the PnL lives in a handful of days or names; quantile
  monotonicity holds in-sample but not OOS; the spread is driven entirely
  by the short leg in one drawdown.
- **Sign instability**: per-fold coefficients or per-era ICs flip sign
  across walk-forward folds. A weak-but-stable IC is worth more than a
  strong-but-flippy one.
- **Turnover outruns decay**: the signal reshuffles daily (`rank_turnover`
  near 0.33) while its IC decay says the edge lasts a month — costs will
  eat it; conversely an IC that *improves* with implementation delay is a
  data error until proven otherwise.
- **Seed or ordering sensitivity**: results change with a random seed,
  row order, or universe ordering. The framework is deterministic;
  non-determinism is a bug in the research, not the world.
- **Synthetic alpha**: any strong result on the synthetic market is a bug
  by construction (statement-only fundamental factors have zero IC there;
  yield ratios inherit only a mild anti-momentum tilt from their price
  denominator).
- **Suspicious magnitude**: a fundamental factor printing a Sharpe of 3,
  an IC above ~0.10 at monthly horizons, or a backtest that never draws
  down is a lookahead or alignment bug until a specific, boring
  explanation is found. Investigate before reporting, every time.

---

## Appendix A — data quality: free sources, the point-in-time problem, and what paid data unlocks

### The point-in-time problem, precisely

A backtest is honest only if every input at date `t` was knowable at `t`.
Fundamental and macro data break this in three distinct ways:

1. **Reporting lag**: a March 31 quarter was filed weeks later. quantlab
   handles this structurally (publication dates on every observation;
   45-day assumed lag where filing dates are missing — conservative for
   10-Qs, optimistic for fiscal Q4s that arrive via 10-Ks, so use 60+ days
   for stricter tests).
2. **Restatements**: vendors silently overwrite history with corrected
   figures; today's snapshot of 2022 is not what you would have seen in
   2022. The `FundamentalDataset` supports multi-version rows and
   `RestatementCheck` surfaces them *when the vendor exposes them*; free
   vendors do not.
3. **Universe recomposition**: today's symbol list omits everything that
   was delisted, acquired, or bankrupted — the survivors' history is
   systematically rosier. No free price API fixes this.

### What each free source actually provides

| Source | Good | Broken for research |
|---|---|---|
| yfinance prices | Adjusted daily OHLCV, long history, fine through the cache | No delisted names (survivorship); occasional bad prints (the validator catches most) |
| yfinance fundamentals | Real quarterly statements, correct current values | Only trailing ~4-6 quarters; no filing dates; silently restated; no delisted names |
| yfinance estimates | Current consensus snapshot | **No history at all** — unusable for backtests; this is why `EstimateRevision` is interface-only |
| SEC EDGAR (free) | True filing timestamps, full as-filed XBRL and text, all filers incl. delisted | Raw: needs a parsing/mapping layer; effort, not money |
| FRED (free) | Every macro series | Heavily revised; **ALFRED** (also free) provides the point-in-time vintages — always prefer it |
| GDELT (free) | Global news events with timestamps | Noisy entity matching; volume ≫ signal |

The cache-accumulation design in `CachedFundamentalSource` is the honest
free-data strategy: fetch quarterly from now on, keep everything the
vendor later drops, and in 2-3 years the platform owns a small genuinely
point-in-time statement history it built itself.

### What paid sources unlock, by stage

| Source (approx. cost) | Unlocks | Stage |
|---|---|---|
| **Tiingo** (~$10-30/mo) | Clean EOD prices incl. many delisted names (softens survivorship); daily-updated fundamentals; tagged news feed | 4, 5 |
| **Polygon.io** (~$30-200/mo) | Financials parsed from SEC XBRL **with filing dates** (real publication dates for `publication_basis="reported"`); minute bars; tagged news | 2 upgrade, 4, 5 |
| **EODHD** (~$20-80/mo) | Broad EOD incl. delisted; fundamentals with decent history | 2 upgrade, 4 |
| **Sharadar / Nasdaq Data Link Core US** (~$300-500/yr) | The budget-quant standard: true point-in-time, as-reported fundamentals (ARQ dimension), delisted coverage, tickers-to-entities done properly | 2 done right; prerequisite for serious 3 |
| **IBES / Refinitiv via WRDS** (institutional/academic) | Point-in-time analyst estimate history — activates `EstimateRevision` as designed | 3+ |
| **RavenPack / Bloomberg** (institutional) | Entity-resolved, timestamped news sentiment | 5, if ever |
| **CRSP + Compustat via WRDS** (academic) | The survivorship-free gold standard for prices + fundamentals | everything |

Rule of thumb: the first paid dollar goes to **point-in-time fundamentals
with filing dates and delisted coverage** (Sharadar or Polygon), because
it converts Stage 2 from "machinery verified on synthetic data" into "real
multi-year fundamental research", and every later stage inherits that
foundation. Sentiment data is the last dollar, not the first.

---

## Appendix B — implementation spec: Stage 3 (multi-factor ML combination)

*Written for mechanical execution. Follow the development workflow in
`CLAUDE.md`: inspect, implement, test, run `python -m pytest`,
`ruff check src tests examples`, `mypy` — all green, offline,
deterministic. Do not commit; the user reviews first. If any instruction
conflicts with the repository, the repository wins — stop and say so
rather than improvising.*

### B.1 `PanelDataset.from_factors`: add `target_transform`

**File:** `src/quantlab/ml/datasets.py` (edit in place).

Add a keyword-only parameter to `from_factors`:

```python
@classmethod
def from_factors(
    cls,
    factors: dict[str, WideFrame],
    prices: WideFrame,
    horizon: int = 21,
    *,
    return_method: str = "log",
    feature_lag: int = 0,
    target_transform: str = "raw",
) -> PanelDataset:
```

Behavior, applied to the wide `target` frame immediately after
`forward_returns(...)` and before stacking:

- `"raw"`: unchanged (exact backward compatibility — default).
- `"demeaned"`: `target = target.sub(target.mean(axis=1), axis=0)` —
  removes the per-date market component so the label is relative return.
- `"rank"`: per-date centered percentile ranks in `(-0.5, 0.5)`, exactly
  the `research.signals.Rank(center=True)` construction:
  `ranked = target.rank(axis=1); n = target.notna().sum(axis=1);
  target = ranked.sub((n + 1) / 2.0, axis=0).div(n, axis=0)`.
- Any other value: `raise DataError(f"Unknown target_transform '{...}'
  (use 'raw', 'demeaned', or 'rank')")`.

Docstring must state that `"demeaned"` is the recommended setting for
cross-sectional research (the model should never fit the market level)
and that `"raw"` remains the default only for backward compatibility.

**Tests** (extend `tests/test_ml_datasets.py`, same style as the existing
tests there):

- `test_target_transform_demeaned_hand_computed`: 2 symbols, constructed
  prices with known forward returns; assert each date's two labels are
  symmetric around 0 and match hand values.
- `test_target_transform_rank_bounds`: labels lie in `(-0.5, 0.5)` and
  each date's labels are a permutation of the expected centered ranks.
- `test_target_transform_raw_is_default_and_backward_compatible`: panel
  built without the argument equals panel built with
  `target_transform="raw"` (compare `X`, `y`, `groups`).
- `test_target_transform_invalid_raises`: `pytest.raises(DataError)`.

### B.2 New module: `src/quantlab/ml/features.py`

Feature assembly — the bridge from factor objects to the
`dict[str, WideFrame]` that `PanelDataset.from_factors` consumes.

```python
"""Feature assembly for supervised panels.

Computes a set of factors on one dataset and applies per-feature
normalization, producing the ``{name: wide frame}`` mapping that
``PanelDataset.from_factors`` consumes. Cross-sectional normalization
(ZScore / Rank) is strongly recommended for every feature so that no
feature carries a market-level component or an incomparable scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.research.factors.base import Factor
from quantlab.research.signals import Transform
from quantlab.types import WideFrame

__all__ = ["FeatureSpec", "build_features"]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One feature: a factor plus optional post-compute normalization.

    ``transform=None`` emits raw scores; pass ``SignalPipeline([...])`` or
    any ``Transform`` for normalization. ``name`` becomes the panel column.
    """

    name: str
    factor: Factor
    transform: Transform | None = None


def build_features(
    dataset: MarketDataset, specs: Sequence[FeatureSpec]
) -> dict[str, WideFrame]:
    """Compute every spec on ``dataset`` -> ``{name: transformed frame}``.

    Raises DataError on empty specs or duplicate names. Order preserved.
    """
```

Implementation notes: validate `specs` non-empty and names unique
(`DataError` with the offending names); loop
`frame = spec.factor.compute(dataset)`, apply
`spec.transform.transform(frame)` when present; return a plain dict in
spec order. No caching, no cleverness. Fundamental factors already carry
their `FundamentalDataset`, so this function needs no fundamentals
parameter — that is why factors take data objects at construction.

**Tests** (new file `tests/test_ml_features.py`):

- `test_build_features_names_shapes_and_order`: two specs on `small_synth`
  (e.g. `Momentum(lookback=63, skip=5)`, `ShortTermReversal(lookback=5)`),
  assert dict keys in order, each frame has the dataset's index/columns.
- `test_build_features_applies_transform`: spec with
  `SignalPipeline([ZScore()])`; assert each non-warmup row of the output
  has mean ≈ 0 (abs tol 1e-9) while the raw factor does not.
- `test_build_features_validates`: empty specs raise `DataError`;
  duplicate names raise `DataError`.
- `test_build_features_mixed_price_and_fundamental`: include
  `EarningsYield(synthetic fundamentals)` alongside a price factor (reuse
  the fixture pattern from `tests/test_research_fundamental_factors.py`);
  assert both frames align to the dataset index/columns.

### B.3 New module: `src/quantlab/ml/selection.py`

Purged-CV model comparison, so selection has one blessed code path.

```python
"""Model selection with leak-proof cross-validation.

Compares candidate models on identical purged folds. This module is for
SELECTION ONLY: its scores are interior-fold numbers and must never be
quoted as out-of-sample results — produce those with WalkForwardTrainer
on the single chosen configuration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pandas as pd

from quantlab.exceptions import ModelError
from quantlab.ml.datasets import PanelDataset
from quantlab.ml.evaluation import regression_metrics
from quantlab.ml.models import ModelWrapper
from quantlab.ml.validation import PurgedKFold, WalkForward

__all__ = ["compare_models"]


def compare_models(
    panel: PanelDataset,
    model_factories: Mapping[str, Callable[[], ModelWrapper]],
    splitter: PurgedKFold | WalkForward,
) -> pd.DataFrame:
    """Fit each candidate per fold; return one row per model.

    Columns: ``mean_r2``, ``std_r2``, ``mean_mse``, ``mean_rank_ic``
    (Spearman correlation between fold predictions and fold targets,
    computed per test date and averaged), ``n_folds``. Index = model name,
    sorted by ``mean_rank_ic`` descending. A fresh model is constructed
    per (model, fold) so folds never share fitted state.
    """
```

Implementation notes: materialize `list(splitter.split(panel))` once and
reuse the identical folds for every model (fair comparison). Per fold:
fit, predict test rows, `regression_metrics(y_test, preds)` for r2/mse;
for the rank IC, group test rows by `panel.groups` date, Spearman-correlate
predictions vs targets within each date (reuse
`pandas.Series.corr(method="spearman")`, require ≥ 3 names per date, skip
dates below that), average over dates then over folds. Empty
`model_factories` raises `ModelError`.

**Tests** (new file `tests/test_ml_selection.py`):

- `test_compare_models_ranks_signal_over_noise`: build a panel from a
  synthetic factor frame where `y = 0.5 * feature + small noise` (plant
  the relation by constructing prices whose forward returns follow the
  factor — or simpler, construct `PanelDataset` directly from hand frames
  the way `tests/test_ml_datasets.py` does); compare
  `{"ridge": lambda: LinearModel(alpha=1.0), "mean": <dummy factory>}`
  where the dummy is a tiny local `ModelWrapper` predicting the training
  mean; assert ridge sorts first with higher `mean_rank_ic`.
- `test_compare_models_is_deterministic`: two calls produce identical
  frames (`pd.testing.assert_frame_equal`).
- `test_compare_models_frame_contract`: exact column names/order, index
  name `"model"`, `n_folds` equals the splitter's fold count.
- `test_compare_models_empty_factories_raises`: `ModelError`.

### B.4 Exports and documentation

- `src/quantlab/ml/__init__.py`: add `FeatureSpec`, `build_features`,
  `compare_models` (alphabetical, matching the existing style).
- `docs/architecture.md` §4.6: append two bullets — `features.py`
  (FeatureSpec/build_features contract) and `selection.py`
  (compare_models is selection-only; walk-forward remains the only causal
  output) — in the same terse style as the section's existing bullets.
- `README.md` status paragraph: mention multi-factor feature assembly and
  model selection.

### B.5 Example: `examples/07_multifactor_ml.py`

Offline, deterministic, mirrors the structure/tone of
`examples/05_ml_alpha.py` and `06_fundamental_factors.py`:

1. `make_synthetic_dataset(n_assets=30, start="2018-01-02",
   end="2023-12-29", n_sectors=5, seed=7)` and
   `SyntheticFundamentalSource(n_assets=30, seed=7,
   inception="2016-01-02")` fetched from 2016 (TTM warmup — copy the
   comment from example 06).
2. `build_features` with six specs, each `SignalPipeline([Winsorize(),
   ZScore()])`: momentum(252, 21), st_reversal(5), realized_vol(63),
   earnings_yield, roe, revenue_growth.
3. `PanelDataset.from_factors(features, ds.close, horizon=21,
   target_transform="demeaned")`.
4. `compare_models` with `PurgedKFold(n_splits=5, horizon=21, embargo=21)`
   over ridge (`LinearModel(alpha=1.0)`), random forest
   (`TreeModel(n_estimators=200, max_depth=3)` — check the wrapper's
   actual constructor arguments in `ml/models.py` before writing), and
   gradient boosting; print the comparison table with an explicit
   "selection only — not results" caption.
5. One `WalkForwardTrainer` run (`WalkForward(train_periods=252,
   test_periods=63)`) with the top model; print OOS
   `prediction_rank_ic` next to each single feature's rank IC on the same
   dates (the example-05 pattern).
6. `signal_to_weights` → realistic-cost backtest → summary.
7. Honest closing print: on synthetic data the ML blend should roughly
   match the best single price factor and fundamentals should contribute
   ≈ nothing — the value of the example is the *workflow*, and any
   dramatic outperformance on synthetic data means a leak.

### B.6 Definition of done

`python -m pytest` (all suites, including the new ones) green;
`ruff check src tests examples` and `mypy` clean; example 07 runs offline
end-to-end; no new hard dependencies; every new public callable has a
docstring stating its contract and its honesty caveats.

---

## Appendix C — implementation spec: Stage 4 (universe expansion to 100+ names)

*Same execution rules as Appendix B.*

### C.1 New module: `src/quantlab/data/universe.py`

Point-in-time universe membership. The module holds the mechanics plus a
static seed list; both carry survivorship documentation.

```python
"""Point-in-time universe construction.

Membership at date t is decided with data through t only (trailing
dollar-volume ranks as of the most recent rebalance bar), so membership
timing is lookahead-free. Membership COMPOSITION still inherits
survivorship bias from the symbol list itself: a list of today's large
caps contains only survivors, and free price sources do not carry
delisted names. Every backtest on such a list overstates returns; say so
when reporting results (see docs/research_roadmap.md, Appendix A).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.data.dataset import MarketDataset
from quantlab.exceptions import DataError
from quantlab.types import WideFrame

__all__ = ["US_LARGE_CAP_150", "US_LARGE_CAP_SECTORS", "apply_universe", "rolling_universe"]


def rolling_universe(
    dataset: MarketDataset,
    *,
    top_n: int = 100,
    window: int = 63,
    min_price: float | None = 5.0,
    rebalance_every: int = 21,
) -> WideFrame:
    """Boolean membership mask (dates x symbols).

    Every ``rebalance_every`` bars, starting at the first bar with a full
    ``window`` of history (bar index ``window - 1``), rank symbols by
    trailing ``window``-bar mean dollar volume computed through that bar
    (inclusive); the top ``top_n`` with trailing median close >=
    ``min_price`` are members until the next rebalance. Rows before the
    first rebalance are all False. ``top_n`` larger than the number of
    eligible symbols keeps them all (no error). Symbols with insufficient
    data at a rebalance (NaN trailing volume) are not members.
    """


def apply_universe(frame: WideFrame, universe: WideFrame) -> WideFrame:
    """NaN out ``frame`` cells outside the universe.

    ``universe`` is reindexed to ``frame``'s index/columns with missing
    entries treated as False, so a signal for a symbol the universe never
    admitted is fully masked. Values inside the universe pass through
    unchanged (this masks, it does not renormalize).
    """
```

Implementation notes for `rolling_universe`: compute
`adv = dataset.dollar_volume().rolling(window).mean()` and
`med_price = dataset.close.rolling(window).median()` once; loop over
rebalance bar positions (`range(window - 1, n_periods, rebalance_every)`);
at each, take the `adv` row, drop NaN, apply the price floor, take the
`top_n` largest by value (`nlargest` — ties resolved by pandas order is
acceptable, note it in the docstring); write True into the mask from this
bar (inclusive) until the next rebalance bar (exclusive). Return a
DataFrame of dtype `bool` with the dataset's index/columns. Validate
`top_n >= 1`, `window >= 1`, `rebalance_every >= 1` (`DataError`).

Static constants at the bottom of the module:

- `US_LARGE_CAP_150: tuple[str, ...]` — 150 large, liquid US tickers
  spread across sectors (choose from well-known current S&P 500
  constituents; correctness of the *code* never depends on the specific
  names). Comment block above it MUST state: reflects constituents as of
  the authoring date, contains survivors only, results on it overstate
  historical returns.
- `US_LARGE_CAP_SECTORS: dict[str, str]` — GICS-style sector label for
  every ticker in the list (usable directly by
  `research.signals.Neutralize`).

### C.2 Tests: new file `tests/test_data_universe.py`

Hand-computed and PIT tests in the established style:

- `test_rolling_universe_hand_computed`: 4 symbols, constant prices,
  constant volumes rigged so dollar-volume order is `A > B > C > D`;
  `top_n=2, window=3, rebalance_every=5`; assert the mask is exactly
  `{A, B}` from bar 2 onward and all-False before bar 2.
- `test_rolling_universe_membership_flips_at_rebalance_only`: volumes
  where `D` overtakes `B` mid-period; assert membership changes exactly at
  the next rebalance bar, not between rebalances.
- `test_rolling_universe_is_point_in_time`: mirror of the momentum-skip
  test — multiply `D`'s volume by 1000 only on bars strictly *after* a
  rebalance bar `t`; assert membership at `t` (and until the next
  rebalance) is identical to the unperturbed mask. This is the mandatory
  lookahead pin.
- `test_rolling_universe_price_floor_and_nan_handling`: a sub-$5 median
  name is excluded despite huge volume; a symbol with NaN trailing volume
  at the rebalance is excluded.
- `test_rolling_universe_top_n_exceeding_symbols_keeps_all`: `top_n=50`
  on 4 symbols → all eligible symbols members after warmup.
- `test_apply_universe_masks_and_passes_through`: hand frame + hand mask;
  assert NaN outside, identical values inside; a symbol missing from the
  mask columns is fully NaN.
- `test_universe_constants_sane`: `len(US_LARGE_CAP_150) >= 100`, all
  unique, all non-empty strings; `US_LARGE_CAP_SECTORS` covers exactly the
  same tickers with non-empty labels.
- `test_rolling_universe_validates_params`: bad `top_n`/`window`/
  `rebalance_every` raise `DataError`.

### C.3 Exports and documentation

- `src/quantlab/data/__init__.py`: add `US_LARGE_CAP_150`,
  `US_LARGE_CAP_SECTORS`, `apply_universe`, `rolling_universe`
  (alphabetical).
- `docs/architecture.md` §4.1: append a bullet for `universe.py`
  (rolling_universe / apply_universe contract, PIT-by-construction, the
  survivorship caveat) in the section's existing style.
- `README.md`: one sentence in the layout tree comment for
  `data/universe.py` and one in the status paragraph.

### C.4 Example: `examples/08_large_universe.py`

Offline on synthetic data (the synthetic source scales cleanly):

1. `make_synthetic_dataset(n_assets=150, start="2018-01-02",
   end="2023-12-29", n_sectors=10, seed=7)`.
2. `rolling_universe(ds, top_n=100, window=63, rebalance_every=21)`;
   print member counts over time (`mask.sum(axis=1).describe()`).
3. Momentum factor → `Winsorize -> ZScore -> Neutralize(sector_map)`
   pipeline → `apply_universe`.
4. Evaluate the same signal on the 100-name universe versus a 30-name
   subset (`ds.select(ds.symbols[:30])`): print IC mean, IC IR, and the
   Newey-West t-stat side by side — the point of the example is the error
   bars shrinking with breadth, so say exactly that in the printout.
5. Long/short backtest with the standard realistic execution settings on
   the universed signal; print the summary and a closing note that on real
   data this workflow additionally requires the survivorship disclaimer
   from the module docstring.

### C.5 Real-data path (documentation only, no code)

In the example's module docstring (not executed): with the `data` extra,
the same workflow runs on `CachedSource(YahooFinanceSource())` over
`US_LARGE_CAP_150` plus
`CachedFundamentalSource(YahooFundamentalSource())` for the fundamental
factors; first fetch is slow, subsequent runs are cache-served; results on
that list carry survivorship bias, full stop.

### C.6 Definition of done

Same bar as B.6: full suite + ruff + mypy green, example 08 offline
end-to-end, no new dependencies, survivorship caveats present in module
docstring, constants, example, and architecture docs.

---

*Maintenance note: when a stage completes, update its row in the §2 table,
record what was cut or discovered (the Stage 2 entry above is the
template), and update `PROJECT_HANDOFF.md` per the repository's checkpoint
ritual.*
